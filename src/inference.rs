use std::collections::HashMap;
use std::path::Path;

use anyhow::{Result, ensure};
use mlx_rs::Array;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::{concatenate_axis, pad};
use mlx_rs::random::normal;

use crate::audio::{SincResampler, VolumeExtractor, adjust_loudness_envelope};
use crate::contentvec::ContentVec;
use crate::diffusion::DiffusionModel;
use crate::gan::GanModel;
use crate::pitch::FcpePredictor;
use crate::slicer::Slicer;
use crate::vocoder::NsfHifiGan;
use crate::weights::Weights;

const SAMPLE_RATE: i32 = 44_100;
const ENCODER_SAMPLE_RATE: i32 = 16_000;
const HOP_SIZE: i32 = 512;

#[derive(Debug, Clone)]
pub struct InferenceOptions {
    pub pitch_shift: f32,
    pub noise_scale: f32,
    pub predict_f0: bool,
    pub shallow_diffusion: bool,
    pub diffusion_steps: i32,
    pub diffusion_speedup: i32,
    pub loudness_envelope_adjustment: f32,
    pub second_encoding: bool,
}

impl Default for InferenceOptions {
    fn default() -> Self {
        Self {
            pitch_shift: 0.0,
            noise_scale: 0.4,
            predict_f0: false,
            shallow_diffusion: false,
            diffusion_steps: 100,
            diffusion_speedup: 10,
            loudness_envelope_adjustment: 1.0,
            second_encoding: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SliceInferenceOptions {
    pub threshold_db: f32,
    pub padding_seconds: f32,
    pub clip_seconds: f32,
    pub crossfade_seconds: f32,
    pub crossfade_ratio: f32,
}

impl Default for SliceInferenceOptions {
    fn default() -> Self {
        Self {
            threshold_db: -40.0,
            padding_seconds: 0.5,
            clip_seconds: 0.0,
            crossfade_seconds: 0.0,
            crossfade_ratio: 0.75,
        }
    }
}

#[derive(Debug)]
pub struct InferenceOutput {
    pub audio: Array,
    pub gan_audio: Array,
    pub refined_mel: Option<Array>,
    pub content: Array,
    pub diffusion_content: Option<Array>,
    pub f0: Array,
    pub uv: Array,
    pub volume: Array,
}

pub struct SovitsSvc {
    contentvec: ContentVec,
    pitch: FcpePredictor,
    gan: GanModel,
    diffusion: DiffusionModel,
    vocoder: NsfHifiGan,
    volume: VolumeExtractor,
    encoder_resampler: SincResampler,
    input_resamplers: HashMap<i32, SincResampler>,
}

impl SovitsSvc {
    pub fn load(
        gan_checkpoint: impl AsRef<Path>,
        diffusion_checkpoint: impl AsRef<Path>,
        contentvec_checkpoint: impl AsRef<Path>,
        fcpe_checkpoint: impl AsRef<Path>,
        vocoder_checkpoint: impl AsRef<Path>,
    ) -> Result<Self> {
        Ok(Self {
            contentvec: ContentVec::load(contentvec_checkpoint)?,
            pitch: FcpePredictor::load(fcpe_checkpoint),
            gan: GanModel::load(gan_checkpoint)?,
            diffusion: DiffusionModel::load(Weights::load(diffusion_checkpoint)?)?,
            vocoder: NsfHifiGan::load(vocoder_checkpoint)?,
            volume: VolumeExtractor::new(HOP_SIZE)?,
            encoder_resampler: SincResampler::new(SAMPLE_RATE, ENCODER_SAMPLE_RATE, 6)?,
            input_resamplers: HashMap::new(),
        })
    }

    fn resample_input(&mut self, audio: &Array, sample_rate: i32) -> Result<Array> {
        if sample_rate == SAMPLE_RATE {
            return Ok(audio.clone());
        }
        if !self.input_resamplers.contains_key(&sample_rate) {
            self.input_resamplers.insert(
                sample_rate,
                SincResampler::new(sample_rate, SAMPLE_RATE, 6)?,
            );
        }
        Ok(self
            .input_resamplers
            .get_mut(&sample_rate)
            .expect("input resampler was inserted")
            .resample(audio)?)
    }

    pub fn infer(
        &mut self,
        audio: &Array,
        sample_rate: i32,
        options: &InferenceOptions,
    ) -> Result<InferenceOutput> {
        ensure!(
            audio.ndim() == 2 && audio.shape()[0] == 1,
            "input audio must have shape [1, samples]"
        );
        ensure!(sample_rate > 0, "input sample rate must be positive");
        ensure!(
            options.noise_scale >= 0.0,
            "GAN noise scale must be non-negative"
        );
        if options.shallow_diffusion {
            ensure!(
                options.diffusion_steps >= 2,
                "diffusion step count must be at least 2"
            );
            ensure!(
                options.diffusion_speedup > 0,
                "diffusion speedup must be positive"
            );
        }
        let resampled_audio = self.resample_input(audio, sample_rate)?;
        let frame_count = resampled_audio.shape()[1] / HOP_SIZE;
        ensure!(frame_count > 0, "input is shorter than one model frame");
        let audio = resampled_audio.index((.., ..frame_count * HOP_SIZE));

        let pitch = self
            .pitch
            .infer(&audio, SAMPLE_RATE as usize, frame_count)?;
        let pitch_factor = 2.0_f64.powf(options.pitch_shift as f64 / 12.0) as f32;
        let input_f0 = pitch.f0 * pitch_factor;
        let wav_16k = self.encoder_resampler.resample(&audio)?;
        let encoded = self.contentvec.encode(&wav_16k)?;
        let content = ContentVec::expand_nearest(&encoded, frame_count)?;
        let volume = self.volume.extract(&audio)?;
        ensure!(
            content.shape()[1] == frame_count
                && input_f0.shape()[1] == frame_count
                && pitch.uv.shape()[1] == frame_count
                && volume.shape()[1] == frame_count,
            "preprocessing produced inconsistent frame counts"
        );
        let speaker_id = Array::from_int(0).reshape(&[1, 1])?;
        let gan = self.gan.infer(
            &content,
            &input_f0,
            &pitch.uv,
            &volume,
            &speaker_id,
            options.noise_scale,
            options.predict_f0,
        )?;
        gan.audio.eval()?;

        let (output, refined_mel, diffusion_content) = if options.shallow_diffusion {
            let gan_waveform = gan.audio.squeeze_axes(&[-1])?;
            let initial_mel = self.vocoder.mel.extract(&gan_waveform)?;
            let f0 = gan.f0.expand_dims(-1)?;
            let volume_condition = volume.expand_dims(-1)?;
            let diffusion_content = if options.second_encoding {
                let wav_16k = self.encoder_resampler.resample(&gan_waveform)?;
                let encoded = self.contentvec.encode(&wav_16k)?;
                ContentVec::expand_nearest(&encoded, gan.f0.shape()[1])?
            } else {
                content.clone()
            };
            let condition = self
                .diffusion
                .condition(&diffusion_content, &f0, &volume_condition)?;
            let normalized = self
                .diffusion
                .norm_spec(&initial_mel)?
                .swap_axes(1, 2)?
                .expand_dims(1)?;
            let diffusion_noise = normal::<f32>(normalized.shape(), None, None, None)?;
            let timestep = Array::from_int(options.diffusion_steps - 1).reshape(&[1])?;
            let noisy = self
                .diffusion
                .q_sample(&normalized, &timestep, &diffusion_noise)?;
            let refined = self.diffusion.sample_dpm_solver_pp(
                &noisy,
                &condition,
                options.diffusion_steps,
                options.diffusion_speedup,
            )?;
            let refined_mel = self
                .diffusion
                .denorm_spec(&refined.squeeze_axes(&[1])?.swap_axes(1, 2)?)?;
            let vocoder_f0 = gan.f0.index((.., ..refined_mel.shape()[1]));
            let output = self.vocoder.infer(&refined_mel, &vocoder_f0)?;
            (output, Some(refined_mel), Some(diffusion_content))
        } else {
            (gan.audio.clone(), None, None)
        };
        let output = if options.loudness_envelope_adjustment != 1.0 {
            adjust_loudness_envelope(
                &resampled_audio,
                SAMPLE_RATE,
                &output,
                SAMPLE_RATE,
                options.loudness_envelope_adjustment,
            )?
            .reshape(&[1, -1, 1])?
        } else {
            output
        };
        output.eval()?;
        Ok(InferenceOutput {
            audio: output,
            gan_audio: gan.audio,
            refined_mel,
            content,
            diffusion_content,
            f0: gan.f0,
            uv: pitch.uv,
            volume,
        })
    }

    pub fn infer_sliced(
        &mut self,
        audio: &Array,
        sample_rate: i32,
        inference_options: &InferenceOptions,
        slice_options: &SliceInferenceOptions,
    ) -> Result<Array> {
        ensure!(
            audio.ndim() == 2 && audio.shape()[0] == 1,
            "input audio must have shape [1, samples]"
        );
        ensure!(sample_rate > 0, "input sample rate must be positive");
        ensure!(
            slice_options.padding_seconds >= 0.0
                && slice_options.clip_seconds >= 0.0
                && slice_options.crossfade_seconds >= 0.0,
            "slice durations must be non-negative"
        );
        ensure!(
            (0.0..=1.0).contains(&slice_options.crossfade_ratio),
            "crossfade ratio must be between zero and one"
        );
        let slices = Slicer::standard(sample_rate, slice_options.threshold_db)?.slice(audio)?;
        let clip_size = (slice_options.clip_seconds * sample_rate as f32) as i32;
        let overlap_size = (slice_options.crossfade_seconds * sample_rate as f32) as i32;
        ensure!(
            clip_size == 0 || overlap_size <= clip_size,
            "crossfade duration cannot exceed clip duration"
        );
        let overlap_mix_size = (overlap_size as f32 * slice_options.crossfade_ratio) as i32;
        let overlap_left_trim = (overlap_size - overlap_mix_size) / 2;
        let overlap_right_trim = overlap_size - overlap_mix_size - overlap_left_trim;
        let mix_weight = if overlap_mix_size > 0 {
            Some(Array::linspace::<_, f32>(
                0.0_f32,
                1.0_f32,
                overlap_mix_size,
            )?)
        } else if overlap_size != 0 {
            Some(Array::zeros::<f32>(&[0])?)
        } else {
            None
        };
        let mut accumulated: Option<Array> = None;
        for slice in slices {
            let slice_length = slice.end - slice.start;
            let converted_slice_length =
                (slice_length as f64 / sample_rate as f64 * SAMPLE_RATE as f64).ceil() as i32;
            if slice.silent {
                let silence = Array::zeros::<f32>(&[1, converted_slice_length])?;
                accumulated = Some(if let Some(previous) = accumulated {
                    concatenate_axis(&[&previous, &silence], 1)?
                } else {
                    silence
                });
                continue;
            }

            let mut pieces = Vec::new();
            if clip_size == 0 {
                pieces.push((0, slice_length));
            } else {
                let mut position = 0;
                while position < slice_length {
                    let start = if position >= overlap_size {
                        position - overlap_size
                    } else {
                        position
                    };
                    pieces.push((start, (position + clip_size).min(slice_length)));
                    position += clip_size;
                }
            }
            for (piece_index, (piece_start, piece_end)) in pieces.into_iter().enumerate() {
                let piece_length = piece_end - piece_start;
                let expected_length = if clip_size != 0 {
                    (piece_length as f64 / sample_rate as f64 * SAMPLE_RATE as f64).ceil() as i32
                } else {
                    converted_slice_length
                };
                let source = audio.index((.., slice.start + piece_start..slice.start + piece_end));
                let source_padding = (sample_rate as f32 * slice_options.padding_seconds) as i32;
                let source = pad(
                    &source,
                    &[(0, 0), (source_padding, source_padding)],
                    None,
                    None,
                )?;
                let inferred = self
                    .infer(&source, sample_rate, inference_options)?
                    .audio
                    .reshape(&[1, -1])?;
                let target_padding = (SAMPLE_RATE as f32 * slice_options.padding_seconds) as i32;
                let inferred = if target_padding == 0 {
                    inferred.index((.., 0..0))
                } else {
                    ensure!(
                        inferred.shape()[1] >= target_padding * 2,
                        "inferred chunk is shorter than its padding"
                    );
                    inferred.index((.., target_padding..inferred.shape()[1] - target_padding))
                };
                let inferred = if inferred.shape()[1] < expected_length {
                    let difference = expected_length - inferred.shape()[1];
                    let left = difference / 2;
                    pad(&inferred, &[(0, 0), (left, difference - left)], None, None)?
                } else {
                    inferred
                };

                if overlap_size != 0 && piece_index != 0 {
                    let previous = accumulated.take().expect("a previous chunk must exist");
                    ensure!(
                        previous.shape()[1] >= overlap_mix_size + overlap_right_trim
                            && inferred.shape()[1] >= overlap_left_trim + overlap_mix_size,
                        "chunk is shorter than the configured crossfade"
                    );
                    let previous_mix = if slice_options.crossfade_ratio == 1.0 {
                        previous.index((.., previous.shape()[1] - overlap_size..))
                    } else {
                        previous.index((
                            ..,
                            previous.shape()[1] - overlap_mix_size - overlap_right_trim
                                ..previous.shape()[1] - overlap_right_trim,
                        ))
                    };
                    let current_mix = if slice_options.crossfade_ratio == 1.0 {
                        inferred.index((.., ..overlap_size))
                    } else {
                        inferred
                            .index((.., overlap_left_trim..overlap_left_trim + overlap_mix_size))
                    };
                    let weight = mix_weight.as_ref().expect("crossfade weight must exist");
                    let mixed = previous_mix * (weight * -1.0_f32 + 1.0_f32) + current_mix * weight;
                    let previous_keep = if slice_options.crossfade_ratio == 1.0 {
                        previous.index((.., ..previous.shape()[1] - overlap_size))
                    } else {
                        previous.index((
                            ..,
                            ..previous.shape()[1] - overlap_mix_size - overlap_right_trim,
                        ))
                    };
                    let current_keep = if slice_options.crossfade_ratio == 1.0 {
                        inferred.index((.., overlap_size..))
                    } else {
                        inferred.index((.., overlap_left_trim + overlap_mix_size..))
                    };
                    accumulated = Some(concatenate_axis(
                        &[&previous_keep, &mixed, &current_keep],
                        1,
                    )?);
                } else {
                    accumulated = Some(if let Some(previous) = accumulated {
                        concatenate_axis(&[&previous, &inferred], 1)?
                    } else {
                        inferred
                    });
                }
            }
        }
        accumulated.ok_or_else(|| anyhow::anyhow!("slicer produced no output"))
    }
}
