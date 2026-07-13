use std::f64::consts::PI;
use std::path::Path;

use anyhow::{Result, ensure};
use mlx_rs::module::Module;
use mlx_rs::nn::{self, Conv1d, ConvTranspose1d, Linear, Upsample, UpsampleMode};
use mlx_rs::ops::indexing::{IndexOp, IntoStrideBy};
use mlx_rs::ops::{
    as_strided, clip, concatenate_axis, imag, log, real, remainder, sqrt, square, tanh, zeros_like,
};
use mlx_rs::random::{normal, uniform};
use mlx_rs::{Array, Dtype, Stream};

use crate::weights::Weights;

const SAMPLE_RATE: f32 = 44_100.0;
const HARMONICS: i32 = 9;
const UPSAMPLE_FACTOR: i32 = 512;
const N_FFT: i32 = 2048;
const HOP_SIZE: i32 = 512;
const WIN_SIZE: i32 = 2048;

#[derive(Debug)]
pub struct VocoderSourceInputs {
    pub initial_phase: Array,
    pub normal: Array,
}

impl VocoderSourceInputs {
    pub fn sample(batch: i32, sample_count: i32) -> Result<Self> {
        let phase_mask = Array::arange::<_, i32>(None, HARMONICS, None)?
            .gt(Array::from_int(0))?
            .as_type::<f32>()?;
        Ok(Self {
            initial_phase: uniform::<_, f32>(0.0_f32, 1.0_f32, &[batch, HARMONICS], None)?
                * phase_mask,
            normal: normal::<f32>(&[batch, sample_count, HARMONICS], None, None, None)?,
        })
    }
}

#[derive(Debug)]
pub struct VocoderTrace {
    pub source_sine: Array,
    pub source_uv: Array,
    pub source_noise: Array,
    pub source_merged: Array,
    pub conv_pre: Array,
    pub upsampled: Vec<Array>,
    pub source_convolved: Vec<Array>,
    pub resblocks: Vec<Array>,
    pub conv_post: Array,
    pub output: Array,
}

#[derive(Debug)]
struct SourceTrace {
    sine: Array,
    uv: Array,
    noise: Array,
    merged: Array,
}

#[derive(Debug)]
struct VocoderResBlock {
    convs_1: Vec<Conv1d>,
    convs_2: Vec<Conv1d>,
}

impl VocoderResBlock {
    fn load(weights: &mut Weights, index: usize, channels: i32, kernel_size: i32) -> Result<Self> {
        let dilations = [1, 3, 5];
        Ok(Self {
            convs_1: dilations
                .into_iter()
                .enumerate()
                .map(|(layer, dilation)| {
                    weights.conv1d(
                        &format!("resblocks.{index}.convs1.{layer}"),
                        channels,
                        channels,
                        kernel_size,
                        (kernel_size * dilation - dilation) / 2,
                        dilation,
                        1,
                        1,
                    )
                })
                .collect::<Result<Vec<_>>>()?,
            convs_2: (0..3)
                .map(|layer| {
                    weights.conv1d(
                        &format!("resblocks.{index}.convs2.{layer}"),
                        channels,
                        channels,
                        kernel_size,
                        (kernel_size - 1) / 2,
                        1,
                        1,
                        1,
                    )
                })
                .collect::<Result<Vec<_>>>()?,
        })
    }

    fn forward(&mut self, input: &Array) -> Result<Array> {
        let mut hidden = input.clone();
        for (conv_1, conv_2) in self.convs_1.iter_mut().zip(&mut self.convs_2) {
            let residual = nn::leaky_relu(&hidden, 0.1)?;
            let residual = conv_1.forward(&residual)?;
            let residual = nn::leaky_relu(residual, 0.1)?;
            let residual = conv_2.forward(&residual)?;
            hidden = hidden + residual;
        }
        Ok(hidden)
    }
}

#[derive(Debug)]
struct VocoderGenerator {
    nearest_upsample: Upsample,
    linear_upsample: Upsample,
    source_linear: Linear,
    conv_pre: Conv1d,
    upsample_layers: Vec<ConvTranspose1d>,
    noise_convs: Vec<Conv1d>,
    resblocks: Vec<VocoderResBlock>,
    conv_post: Conv1d,
}

impl VocoderGenerator {
    fn load(weights: &mut Weights) -> Result<Self> {
        let rates = [8, 8, 2, 2, 2];
        let kernels = [16, 16, 4, 4, 4];
        let channels = [256, 128, 64, 32, 16];
        let noise_strides = [64, 8, 4, 2, 1];
        let noise_kernels = [128, 16, 8, 4, 1];
        let noise_padding = [32, 4, 2, 1, 0];
        let upsample_layers = rates
            .into_iter()
            .zip(kernels)
            .enumerate()
            .map(|(index, (rate, kernel))| {
                weights.conv_transpose1d(
                    &format!("ups.{index}"),
                    512 / 2_i32.pow(index as u32),
                    channels[index],
                    kernel,
                    (kernel - rate) / 2,
                    rate,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let noise_convs = channels
            .into_iter()
            .enumerate()
            .map(|(index, output_channels)| {
                weights.conv1d(
                    &format!("noise_convs.{index}"),
                    1,
                    output_channels,
                    noise_kernels[index],
                    noise_padding[index],
                    1,
                    noise_strides[index],
                    1,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let mut resblocks = Vec::with_capacity(15);
        for (stage, stage_channels) in channels.into_iter().enumerate() {
            for (kernel_index, kernel_size) in [3, 7, 11].into_iter().enumerate() {
                resblocks.push(VocoderResBlock::load(
                    weights,
                    stage * 3 + kernel_index,
                    stage_channels,
                    kernel_size,
                )?);
            }
        }
        Ok(Self {
            nearest_upsample: Upsample::new(UPSAMPLE_FACTOR as f32, UpsampleMode::Nearest),
            linear_upsample: Upsample::new(
                UPSAMPLE_FACTOR as f32,
                UpsampleMode::Linear {
                    align_corners: true,
                },
            ),
            source_linear: weights.linear("m_source.l_linear", HARMONICS, 1, true)?,
            conv_pre: weights.conv1d("conv_pre", 128, 512, 7, 3, 1, 1, 1)?,
            upsample_layers,
            noise_convs,
            resblocks,
            conv_post: weights.conv1d("conv_post", 16, 1, 7, 3, 1, 1, 1)?,
        })
    }

    fn harmonic_source(&mut self, f0: &Array, source: &VocoderSourceInputs) -> Result<SourceTrace> {
        ensure!(f0.ndim() == 2, "f0 must have shape [batch, frames]");
        ensure!(
            source.initial_phase.shape() == [f0.shape()[0], HARMONICS],
            "initial phase has invalid shape {:?}",
            source.initial_phase.shape()
        );
        ensure!(
            source.normal.shape() == [f0.shape()[0], f0.shape()[1] * UPSAMPLE_FACTOR, HARMONICS],
            "source noise has invalid shape {:?}",
            source.normal.shape()
        );
        let f0 = f0.expand_dims(-1)?;
        let harmonics = Array::arange::<_, f32>(Some(1), HARMONICS + 1, None)?;
        let frequency = &f0 * harmonics;
        let mut radians = remainder(&(frequency / SAMPLE_RATE), &Array::from_f32(1.0))?;
        let time_zero = Array::arange::<_, i32>(None, radians.shape()[1], None)?
            .eq(Array::from_int(0))?
            .as_type::<f32>()?
            .reshape(&[1, -1, 1])?;
        radians = radians + time_zero * source.initial_phase.expand_dims(1)?;

        let cpu = Stream::cpu();
        let accumulated = radians
            .as_dtype_device(Dtype::Float64, &cpu)?
            .cumsum_device(1, false, true, &cpu)?
            .as_dtype_device(Dtype::Float32, &cpu)?
            * UPSAMPLE_FACTOR as f32;
        let interpolated_accumulated = self.linear_upsample.forward(&accumulated)?;
        let radians = self.nearest_upsample.forward(&radians)?;
        let wrapped = remainder(&interpolated_accumulated, &Array::from_f32(1.0))?;
        let wrapped_difference = wrapped.index((.., 1.., ..)) - wrapped.index((.., ..-1, ..));
        let shift_tail = wrapped_difference
            .lt(Array::from_int(0))?
            .as_type::<f32>()?
            * -1.0_f32;
        let first_shift = zeros_like(&radians.index((.., 0..1, ..)))?;
        let shift = concatenate_axis(&[&first_shift, &shift_tail], 1)?;
        let phase = radians
            .as_dtype_device(Dtype::Float64, &cpu)?
            .add_device(shift.as_dtype_device(Dtype::Float64, &cpu)?, &cpu)?
            .cumsum_device(1, false, true, &cpu)?
            .multiply_device(Array::from_f64(2.0 * PI), &cpu)?;
        let waves = phase
            .sin_device(&cpu)?
            .as_dtype_device(Dtype::Float32, &cpu)?
            * 0.1_f32;
        let uv = self
            .nearest_upsample
            .forward(&f0.gt(Array::from_int(0))?.as_type::<f32>()?)?;
        let noise_amplitude = &uv * 0.003_f32 + (&uv * -1.0_f32 + 1.0_f32) / 30.0_f32;
        let noise = &source.normal * noise_amplitude;
        let sine = waves * &uv + &noise;
        let merged = tanh(&self.source_linear.forward(&sine)?)?;
        Ok(SourceTrace {
            sine,
            uv,
            noise,
            merged,
        })
    }

    fn forward_traced(
        &mut self,
        mel: &Array,
        f0: &Array,
        source: &VocoderSourceInputs,
    ) -> Result<VocoderTrace> {
        ensure!(
            mel.ndim() == 3 && mel.shape()[2] == 128,
            "mel must have shape [batch, frames, 128]"
        );
        ensure!(
            f0.shape() == [mel.shape()[0], mel.shape()[1]],
            "f0 shape {:?} does not match mel shape {:?}",
            f0.shape(),
            mel.shape()
        );
        let source_trace = self.harmonic_source(f0, source)?;
        let conv_pre = self.conv_pre.forward(mel)?;
        let mut hidden = conv_pre.clone();
        let mut upsampled = Vec::with_capacity(5);
        let mut source_convolved = Vec::with_capacity(5);
        let mut resblock_outputs = Vec::with_capacity(15);
        for stage in 0..self.upsample_layers.len() {
            hidden = nn::leaky_relu(&hidden, 0.1)?;
            let upsampled_stage = self.upsample_layers[stage].forward(&hidden)?;
            let source_stage = self.noise_convs[stage].forward(&source_trace.merged)?;
            hidden = &upsampled_stage + &source_stage;
            let first = self.resblocks[stage * 3].forward(&hidden)?;
            let second = self.resblocks[stage * 3 + 1].forward(&hidden)?;
            let third = self.resblocks[stage * 3 + 2].forward(&hidden)?;
            hidden = (&first + &second + &third) / 3.0_f32;
            upsampled.push(upsampled_stage);
            source_convolved.push(source_stage);
            resblock_outputs.push(first);
            resblock_outputs.push(second);
            resblock_outputs.push(third);
        }
        hidden = nn::leaky_relu(hidden, None)?;
        let conv_post = self.conv_post.forward(&hidden)?;
        let output = tanh(&conv_post)?;
        Ok(VocoderTrace {
            source_sine: source_trace.sine,
            source_uv: source_trace.uv,
            source_noise: source_trace.noise,
            source_merged: source_trace.merged,
            conv_pre,
            upsampled,
            source_convolved,
            resblocks: resblock_outputs,
            conv_post,
            output,
        })
    }
}

#[derive(Debug)]
pub struct MelSpectrogram {
    mel_basis: Array,
    hann_window: Array,
}

impl MelSpectrogram {
    fn load(weights: &mut Weights) -> Result<Self> {
        let mel_basis = weights.take("_buffers.mel_basis")?;
        let hann_window = weights.take("_buffers.hann_window")?;
        ensure!(
            mel_basis.shape() == [N_FFT / 2 + 1, 128],
            "invalid mel basis shape {:?}",
            mel_basis.shape()
        );
        ensure!(
            hann_window.shape() == [WIN_SIZE],
            "invalid Hann window shape {:?}",
            hann_window.shape()
        );
        Ok(Self {
            mel_basis,
            hann_window,
        })
    }

    pub fn extract(&self, audio: &Array) -> Result<Array> {
        ensure!(audio.ndim() == 2, "audio must have shape [batch, samples]");
        ensure!(
            audio.dtype() == Dtype::Float32,
            "audio must use float32, got {:?}",
            audio.dtype()
        );
        let batch = audio.shape()[0];
        let sample_count = audio.shape()[1];
        let pad_left = (WIN_SIZE - HOP_SIZE) / 2;
        let pad_right = ((WIN_SIZE - HOP_SIZE + 1) / 2).max(WIN_SIZE - sample_count - pad_left);
        let padded = if pad_right < sample_count {
            let left = audio
                .index((.., 1..pad_left + 1))
                .index((.., (..).stride_by(-1)));
            let right = audio
                .index((.., sample_count - pad_right - 1..sample_count - 1))
                .index((.., (..).stride_by(-1)));
            concatenate_axis(&[&left, audio, &right], 1)?
        } else {
            let left = Array::zeros::<f32>(&[batch, pad_left])?;
            let right = Array::zeros::<f32>(&[batch, pad_right])?;
            concatenate_axis(&[&left, audio, &right], 1)?
        };
        let padded_count = sample_count + pad_left + pad_right;
        ensure!(
            padded_count >= N_FFT,
            "padded audio is shorter than one FFT window"
        );
        let frame_count = (padded_count - N_FFT) / HOP_SIZE + 1;
        let frames = as_strided(
            &padded,
            &[batch, frame_count, N_FFT][..],
            &[padded_count as i64, HOP_SIZE as i64, 1_i64][..],
            0,
        )?;
        let spectrum = mlx_rs::fft::rfft(&(&frames * &self.hann_window), N_FFT, -1)?;
        let magnitude =
            sqrt(&(square(&real(&spectrum)?)? + square(&imag(&spectrum)?)? + 1.0e-9_f32))?;
        let mel = magnitude.matmul(&self.mel_basis)?;
        Ok(log(&clip(&mel, (1.0e-5_f32, ()))?)?)
    }
}

#[derive(Debug)]
pub struct NsfHifiGan {
    pub mel: MelSpectrogram,
    generator: VocoderGenerator,
}

impl NsfHifiGan {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let mut weights = Weights::load(path)?;
        let mel = MelSpectrogram::load(&mut weights)?;
        let generator = VocoderGenerator::load(&mut weights)?;
        weights.finish()?;
        Ok(Self { mel, generator })
    }

    pub fn infer_with_source(
        &mut self,
        mel: &Array,
        f0: &Array,
        source: &VocoderSourceInputs,
    ) -> Result<Array> {
        Ok(self.generator.forward_traced(mel, f0, source)?.output)
    }

    pub fn infer_traced(
        &mut self,
        mel: &Array,
        f0: &Array,
        source: &VocoderSourceInputs,
    ) -> Result<VocoderTrace> {
        self.generator.forward_traced(mel, f0, source)
    }

    pub fn infer(&mut self, mel: &Array, f0: &Array) -> Result<Array> {
        let source = VocoderSourceInputs::sample(mel.shape()[0], mel.shape()[1] * UPSAMPLE_FACTOR)?;
        self.infer_with_source(mel, f0, &source)
    }
}
