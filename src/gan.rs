use std::path::Path;

use anyhow::{Result, ensure};
use mlx_rs::Array;
use mlx_rs::module::Module;
use mlx_rs::nn::{Conv1d, Embedding, Linear};
use mlx_rs::ops::{exp, log, r#where};
use mlx_rs::random::normal;

use crate::attention::Encoder;
use crate::flow::ResidualCouplingBlock;
use crate::weights::Weights;

pub use crate::f0_decoder::{F0Decoder, F0Prediction};
pub use crate::gan_generator::{Generator, SourceInputs};

const F0_BIN: i32 = 256;
const F0_MEL_SCALE: f32 = 0.257_435_92;
const F0_MEL_OFFSET: f32 = 19.016_922;

#[derive(Debug)]
pub struct GanCondition {
    pub speaker: Array,
    pub mask: Array,
    pub volume_embedding: Array,
    pub preprocessed: Array,
}

#[derive(Debug)]
pub struct GanLatent {
    pub speaker: Array,
    pub mask: Array,
    pub volume_embedding: Array,
    pub preprocessed: Array,
    pub mean: Array,
    pub log_scale: Array,
    pub sampled: Array,
    pub flowed: Array,
}

#[derive(Debug)]
pub struct GanCore {
    speaker_embedding: Embedding,
    volume_embedding: Linear,
    pre: Conv1d,
    uv_embedding: Embedding,
    f0_embedding: Embedding,
    encoder: Encoder,
    projection: Conv1d,
    flow: ResidualCouplingBlock,
}

impl GanCore {
    pub fn load(weights: &mut Weights) -> Result<Self> {
        Ok(Self {
            speaker_embedding: weights.embedding("emb_g", 1, 768)?,
            volume_embedding: weights.linear("emb_vol", 1, 192, true)?,
            pre: weights.conv1d("pre", 768, 192, 5, 2, 1, 1, 1)?,
            uv_embedding: weights.embedding("emb_uv", 2, 192)?,
            f0_embedding: weights.embedding("enc_p.f0_emb", 256, 192)?,
            encoder: Encoder::load(weights)?,
            projection: weights.conv1d("enc_p.proj", 192, 384, 1, 0, 1, 1, 1)?,
            flow: ResidualCouplingBlock::load(weights)?,
        })
    }

    pub fn prepare_condition(
        &mut self,
        content: &Array,
        uv: &Array,
        volume: &Array,
        speaker_id: &Array,
    ) -> Result<GanCondition> {
        let speaker = self.speaker_embedding.forward(speaker_id)?;
        let mask = Array::ones::<f32>(&[content.shape()[0], content.shape()[1], 1])?;
        let volume_embedding = self.volume_embedding.forward(&volume.expand_dims(-1)?)?;
        let preprocessed = self.pre.forward(content)? * &mask
            + self.uv_embedding.forward(&uv.as_type::<i32>()?)?
            + &volume_embedding;
        Ok(GanCondition {
            speaker,
            mask,
            volume_embedding,
            preprocessed,
        })
    }

    pub fn infer_from_condition(
        &mut self,
        condition: GanCondition,
        f0_coarse: &Array,
        encoder_noise: &Array,
        noise_scale: f32,
    ) -> Result<GanLatent> {
        let GanCondition {
            speaker,
            mask,
            volume_embedding,
            preprocessed,
        } = condition;
        let hidden = &preprocessed + self.f0_embedding.forward(f0_coarse)?;
        let hidden = self.encoder.forward(&(hidden * &mask), &mask)?;
        let stats = self.projection.forward(&hidden)? * &mask;
        let stats = stats.split(2, -1)?;
        let mean = stats[0].clone();
        let log_scale = stats[1].clone();
        let sampled = (&mean + encoder_noise * exp(&log_scale)? * noise_scale) * &mask;
        let flowed = self.flow.reverse(&sampled, &mask, &speaker)?;
        Ok(GanLatent {
            speaker,
            mask,
            volume_embedding,
            preprocessed,
            mean,
            log_scale,
            sampled,
            flowed,
        })
    }

    pub fn infer_latent(
        &mut self,
        content: &Array,
        f0_coarse: &Array,
        uv: &Array,
        volume: &Array,
        speaker_id: &Array,
        encoder_noise: &Array,
        noise_scale: f32,
    ) -> Result<GanLatent> {
        let condition = self.prepare_condition(content, uv, volume, speaker_id)?;
        self.infer_from_condition(condition, f0_coarse, encoder_noise, noise_scale)
    }
}

pub fn f0_to_coarse(f0: &Array) -> Result<Array> {
    let f0_mel = log(&(f0 / 700.0_f32 + 1.0_f32))? * 1127.0_f32;
    let f0_mel = r#where(
        &f0_mel.gt(Array::from_int(0))?,
        &(f0_mel.clone() * F0_MEL_SCALE - F0_MEL_OFFSET),
        &f0_mel,
    )?;
    let mut coarse = f0_mel.round(None)?.as_type::<i32>()?;
    coarse = &coarse * coarse.gt(Array::from_int(0))?.as_type::<i32>()?;
    coarse = &coarse + coarse.lt(Array::from_int(1))?.as_type::<i32>()?;
    coarse = &coarse * coarse.lt(Array::from_int(F0_BIN))?.as_type::<i32>()?;
    coarse = &coarse + coarse.ge(Array::from_int(F0_BIN))?.as_type::<i32>()? * (F0_BIN - 1);
    Ok(coarse)
}

#[derive(Debug)]
pub struct GanOutput {
    pub audio: Array,
    pub f0: Array,
}

#[derive(Debug)]
pub struct GanModel {
    core: GanCore,
    f0_decoder: F0Decoder,
    generator: Generator,
}

impl GanModel {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let mut weights = Weights::load(path)?;
        let model = Self {
            core: GanCore::load(&mut weights)?,
            f0_decoder: F0Decoder::load(&mut weights)?,
            generator: Generator::load(&mut weights)?,
        };
        ensure!(
            weights.discard_prefix("enc_q.") > 0,
            "enc_q tensors missing"
        );
        weights.finish()?;
        Ok(model)
    }

    pub fn infer(
        &mut self,
        content: &Array,
        f0: &Array,
        uv: &Array,
        volume: &Array,
        speaker_id: &Array,
        noise_scale: f32,
        predict_f0: bool,
    ) -> Result<GanOutput> {
        let condition = self
            .core
            .prepare_condition(content, uv, volume, speaker_id)?;
        let f0 = if predict_f0 {
            self.f0_decoder
                .predict(
                    &condition.preprocessed,
                    f0,
                    uv,
                    &condition.mask,
                    &condition.speaker,
                )?
                .f0
        } else {
            f0.clone()
        };
        let f0_coarse = f0_to_coarse(&f0)?;
        let encoder_noise = normal::<f32>(
            &[content.shape()[0], content.shape()[1], 192],
            None,
            None,
            None,
        )?;
        let latent =
            self.core
                .infer_from_condition(condition, &f0_coarse, &encoder_noise, noise_scale)?;
        let audio =
            self.generator
                .forward(&(&latent.flowed * &latent.mask), &f0, &latent.speaker)?;
        Ok(GanOutput { audio, f0 })
    }
}
