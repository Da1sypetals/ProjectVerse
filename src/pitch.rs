use std::collections::HashMap;
use std::path::Path;

use anyhow::{Result, ensure};
use fcpe_mlxrs::{
    CFNaiveMelPE, build_hann_window, build_mel_filterbank, postprocess_f0, wav_to_mel,
};
use mlx_rs::Array;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::{concatenate_axis, floor};

use crate::audio::SincResampler;

const FCPE_SAMPLE_RATE: usize = 16_000;
const FCPE_HOP_SIZE: i32 = 160;

#[derive(Debug)]
pub struct FcpeTrace {
    pub resampled_audio: Array,
    pub mel: Array,
    pub latent: Array,
    pub raw_f0: Array,
    pub expanded_raw_f0: Array,
}

#[derive(Debug)]
pub struct PitchOutput {
    pub f0: Array,
    pub uv: Array,
}

pub struct FcpePredictor {
    model: CFNaiveMelPE,
    mel_basis: Array,
    hann_window: Array,
    resamplers: HashMap<usize, SincResampler>,
}

impl FcpePredictor {
    pub fn load(path: impl AsRef<Path>) -> Self {
        Self {
            model: CFNaiveMelPE::load(path),
            mel_basis: build_mel_filterbank(16_000.0, 1024, 128, 0.0, 8_000.0),
            hann_window: build_hann_window(1024),
            resamplers: HashMap::new(),
        }
    }

    fn infer_inner(
        &mut self,
        audio: &Array,
        sample_rate: usize,
        target_length: i32,
    ) -> Result<(PitchOutput, FcpeTrace)> {
        ensure!(target_length > 0, "target length must be positive");
        let audio = if audio.ndim() == 1 {
            audio.reshape(&[1, -1])?
        } else {
            ensure!(
                audio.ndim() == 2 && audio.shape()[0] == 1,
                "FCPE audio must have shape [samples] or [1, samples]"
            );
            audio.clone()
        };
        audio.eval()?;
        let input_samples = audio.shape()[1];
        let resampled_audio = if sample_rate == FCPE_SAMPLE_RATE {
            audio.clone()
        } else {
            if !self.resamplers.contains_key(&sample_rate) {
                self.resamplers.insert(
                    sample_rate,
                    SincResampler::new(sample_rate as i32, FCPE_SAMPLE_RATE as i32, 128)?,
                );
            }
            self.resamplers
                .get_mut(&sample_rate)
                .expect("FCPE resampler was inserted")
                .resample(&audio)?
        };
        self.infer_resampled_inner(resampled_audio, input_samples, target_length)
    }

    fn infer_resampled_inner(
        &mut self,
        resampled_audio: Array,
        source_sample_count: i32,
        target_length: i32,
    ) -> Result<(PitchOutput, FcpeTrace)> {
        let mut mel = wav_to_mel(&resampled_audio, &self.mel_basis, &self.hann_window);
        let source_frame_count = source_sample_count / FCPE_HOP_SIZE + 1;
        if source_frame_count > mel.shape()[1] {
            let last = mel.index((.., -1, ..)).reshape(&[1, 1, 128])?;
            mel = concatenate_axis(&[&mel, &last], 1)?;
        } else if source_frame_count < mel.shape()[1] {
            mel = mel.index((.., ..source_frame_count, ..));
        }

        let latent = self.model.forward(&mel);
        let cents = self.model.latent2cents_local_decoder(&latent, 0.05);
        let raw_f0 = self.model.cent_to_f0(&cents);
        let raw_frame_count = raw_f0.shape()[1];
        let indices = floor(
            &(Array::arange::<_, f32>(None, target_length, None)? * raw_frame_count as f32
                / target_length as f32),
        )?
        .as_type::<i32>()?;
        let expanded_raw_f0 = raw_f0.take_axis(indices, 1)?;
        let (f0, unvoiced) = postprocess_f0(&expanded_raw_f0, 32.7, None, true);
        let f0 = f0.squeeze_axes(&[-1])?;
        let uv = (Array::from_f32(1.0) - unvoiced).squeeze_axes(&[-1])?;
        Ok((
            PitchOutput { f0, uv },
            FcpeTrace {
                resampled_audio,
                mel,
                latent,
                raw_f0,
                expanded_raw_f0,
            },
        ))
    }

    pub fn infer(
        &mut self,
        audio: &Array,
        sample_rate: usize,
        target_length: i32,
    ) -> Result<PitchOutput> {
        Ok(self.infer_inner(audio, sample_rate, target_length)?.0)
    }

    pub fn infer_traced(
        &mut self,
        audio: &Array,
        sample_rate: usize,
        target_length: i32,
    ) -> Result<(PitchOutput, FcpeTrace)> {
        self.infer_inner(audio, sample_rate, target_length)
    }

    pub fn infer_resampled_traced(
        &mut self,
        audio_16k: &Array,
        source_sample_count: i32,
        target_length: i32,
    ) -> Result<(PitchOutput, FcpeTrace)> {
        ensure!(
            audio_16k.ndim() == 2 && audio_16k.shape()[0] == 1,
            "resampled FCPE audio must have shape [1, samples]"
        );
        self.infer_resampled_inner(audio_16k.clone(), source_sample_count, target_length)
    }
}
