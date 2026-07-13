use std::f32::consts::PI;

use anyhow::{Result, ensure};
use mlx_rs::Array;
use mlx_rs::module::Module;
use mlx_rs::nn::{self, Conv1d, ConvTranspose1d, Linear, Upsample, UpsampleMode};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::{concatenate_axis, remainder, sin, tanh, zeros_like};
use mlx_rs::random::{normal, uniform};

use crate::weights::Weights;

const SAMPLE_RATE: f32 = 44_100.0;
const HARMONICS: i32 = 9;
const UPSAMPLE_FACTOR: i32 = 512;

#[derive(Debug)]
pub struct SourceInputs {
    pub initial_phase: Array,
    pub sine_noise: Array,
    pub source_noise: Array,
}

impl SourceInputs {
    pub fn sample(batch: i32, sample_count: i32) -> Result<Self> {
        let phase_mask = Array::arange::<_, i32>(None, HARMONICS, None)?
            .gt(Array::from_int(0))?
            .as_type::<f32>()?;
        Ok(Self {
            initial_phase: uniform::<_, f32>(0.0_f32, 1.0_f32, &[batch, HARMONICS], None)?
                * phase_mask,
            sine_noise: normal::<f32>(&[batch, sample_count, HARMONICS], None, None, None)?,
            source_noise: normal::<f32>(&[batch, sample_count, 1], None, None, None)?,
        })
    }
}

#[derive(Debug)]
struct ResBlock {
    convs_1: Vec<Conv1d>,
    convs_2: Vec<Conv1d>,
}

impl ResBlock {
    fn load(weights: &mut Weights, index: usize, channels: i32, kernel_size: i32) -> Result<Self> {
        let dilations = [1, 3, 5];
        Ok(Self {
            convs_1: dilations
                .into_iter()
                .enumerate()
                .map(|(layer, dilation)| {
                    weights.conv1d(
                        &format!("dec.resblocks.{index}.convs1.{layer}"),
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
                        &format!("dec.resblocks.{index}.convs2.{layer}"),
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
pub struct Generator {
    f0_upsample: Upsample,
    source_linear: Linear,
    conv_pre: Conv1d,
    condition: Conv1d,
    upsample_layers: Vec<ConvTranspose1d>,
    noise_convs: Vec<Conv1d>,
    resblocks: Vec<ResBlock>,
    conv_post: Conv1d,
}

impl Generator {
    pub fn load(weights: &mut Weights) -> Result<Self> {
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
                    &format!("dec.ups.{index}"),
                    512 / 2_i32.pow(index as u32),
                    channels[index],
                    kernel,
                    (kernel - rate + 1) / 2,
                    rate,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let noise_convs = channels
            .into_iter()
            .enumerate()
            .map(|(index, output_channels)| {
                weights.conv1d(
                    &format!("dec.noise_convs.{index}"),
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
                resblocks.push(ResBlock::load(
                    weights,
                    stage * 3 + kernel_index,
                    stage_channels,
                    kernel_size,
                )?);
            }
        }
        Ok(Self {
            f0_upsample: Upsample::new(UPSAMPLE_FACTOR as f32, UpsampleMode::Nearest),
            source_linear: weights.linear("dec.m_source.l_linear", HARMONICS, 1, true)?,
            conv_pre: weights.conv1d("dec.conv_pre", 192, 512, 7, 3, 1, 1, 1)?,
            condition: weights.conv1d("dec.cond", 768, 512, 1, 0, 1, 1, 1)?,
            upsample_layers,
            noise_convs,
            resblocks,
            conv_post: weights.conv1d("dec.conv_post", 16, 1, 7, 3, 1, 1, 1)?,
        })
    }

    fn harmonic_source(&mut self, f0: &Array, source: &SourceInputs) -> Result<Array> {
        let harmonics = Array::arange::<_, f32>(Some(1), HARMONICS + 1, None)?;
        let frequency = f0 * harmonics;
        let mut radians = remainder(&(frequency / SAMPLE_RATE), &Array::from_f32(1.0))?;
        let time_zero = Array::arange::<_, i32>(None, radians.shape()[1], None)?
            .eq(Array::from_int(0))?
            .as_type::<f32>()?
            .reshape(&[1, -1, 1])?;
        radians = radians + time_zero * source.initial_phase.expand_dims(1)?;
        let wrapped = remainder(&radians.cumsum(1, false, true)?, &Array::from_f32(1.0))?;
        let wrap_shift = wrapped.index((.., 1.., ..)) - wrapped.index((.., ..-1, ..));
        let wrap_shift = wrap_shift.lt(Array::from_int(0))?.as_type::<f32>()? * -1.0_f32;
        let first_shift = zeros_like(&radians.index((.., 0..1, ..)))?;
        let wrap_shift = concatenate_axis(&[&first_shift, &wrap_shift], 1)?;
        let waves =
            sin(&((radians + wrap_shift).cumsum(1, false, true)? * (2.0_f32 * PI)))? * 0.1_f32;
        let voiced = f0.gt(Array::from_int(0))?.as_type::<f32>()?;
        let noise_amplitude = &voiced * 0.003_f32 + (&voiced * -1.0_f32 + 1.0_f32) / 30.0_f32;
        let waves = waves * &voiced + &source.sine_noise * noise_amplitude;
        let _source_noise = &source.source_noise / 30.0_f32;
        Ok(tanh(&self.source_linear.forward(&waves)?)?)
    }

    pub fn forward_with_source(
        &mut self,
        latent: &Array,
        f0: &Array,
        speaker: &Array,
        source: &SourceInputs,
    ) -> Result<Array> {
        ensure!(
            latent.ndim() == 3,
            "latent must have shape [batch, frames, channels]"
        );
        let f0 = self.f0_upsample.forward(&f0.expand_dims(-1)?)?;
        let harmonic_source = self.harmonic_source(&f0, source)?;
        let mut hidden = self.conv_pre.forward(latent)? + self.condition.forward(speaker)?;
        for stage in 0..self.upsample_layers.len() {
            hidden = nn::leaky_relu(&hidden, 0.1)?;
            hidden = self.upsample_layers[stage].forward(&hidden)?;
            hidden = hidden + self.noise_convs[stage].forward(&harmonic_source)?;
            let first = self.resblocks[stage * 3].forward(&hidden)?;
            let second = self.resblocks[stage * 3 + 1].forward(&hidden)?;
            let third = self.resblocks[stage * 3 + 2].forward(&hidden)?;
            hidden = (first + second + third) / 3.0_f32;
        }
        hidden = nn::leaky_relu(hidden, None)?;
        Ok(tanh(&self.conv_post.forward(&hidden)?)?)
    }

    pub fn forward(&mut self, latent: &Array, f0: &Array, speaker: &Array) -> Result<Array> {
        let source = SourceInputs::sample(latent.shape()[0], latent.shape()[1] * UPSAMPLE_FACTOR)?;
        self.forward_with_source(latent, f0, speaker, &source)
    }
}
