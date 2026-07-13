use anyhow::Result;
use mlx_rs::Array;
use mlx_rs::module::Module;
use mlx_rs::nn::Conv1d;
use mlx_rs::ops::{log10, power, r#where};

use crate::attention::CausalFft;
use crate::weights::Weights;

#[derive(Debug)]
pub struct F0Prediction {
    pub log_f0: Array,
    pub normalized_log_f0: Array,
    pub predicted_log_f0: Array,
    pub f0: Array,
}

#[derive(Debug)]
pub struct F0Decoder {
    prenet: Conv1d,
    decoder: CausalFft,
    projection: Conv1d,
    f0_prenet: Conv1d,
    condition: Conv1d,
}

impl F0Decoder {
    pub fn load(weights: &mut Weights) -> Result<Self> {
        Ok(Self {
            prenet: weights.conv1d("f0_decoder.prenet", 192, 192, 3, 1, 1, 1, 1)?,
            decoder: CausalFft::load(weights)?,
            projection: weights.conv1d("f0_decoder.proj", 192, 1, 1, 0, 1, 1, 1)?,
            f0_prenet: weights.conv1d("f0_decoder.f0_prenet", 1, 192, 3, 1, 1, 1, 1)?,
            condition: weights.conv1d("f0_decoder.cond", 768, 192, 1, 0, 1, 1, 1)?,
        })
    }

    pub fn forward(
        &mut self,
        input: &Array,
        normalized_log_f0: &Array,
        mask: &Array,
        speaker: &Array,
    ) -> Result<Array> {
        let hidden =
            input + self.condition.forward(speaker)? + self.f0_prenet.forward(normalized_log_f0)?;
        let hidden = self.prenet.forward(&hidden)? * mask;
        let hidden = self.decoder.forward(&(hidden * mask), mask)?;
        Ok(self.projection.forward(&hidden)? * mask)
    }

    pub fn predict(
        &mut self,
        input: &Array,
        f0: &Array,
        uv: &Array,
        mask: &Array,
        speaker: &Array,
    ) -> Result<F0Prediction> {
        let log_f0 = log10(&(f0.expand_dims(-1)? / 700.0_f32 + 1.0_f32))? * 2595.0_f32 / 500.0_f32;
        let uv_sum = uv.sum_axis(1, true)?;
        let uv_sum = r#where(
            &uv_sum.eq(Array::from_int(0))?,
            &Array::from_f32(9999.0),
            &uv_sum,
        )?;
        let means = (log_f0.squeeze_axes(&[-1])? * uv).sum_axis(1, true)? / uv_sum;
        let normalized_log_f0 = (log_f0.clone() - means.expand_dims(-1)?) * mask;
        let predicted_log_f0 = self.forward(input, &normalized_log_f0, mask, speaker)?;
        let exponent = &predicted_log_f0 * 500.0_f32 / 2595.0_f32;
        let f0 = (power(Array::from_int(10), exponent)? - 1.0_f32) * 700.0_f32;
        let f0 = f0.squeeze_axes(&[-1])?;
        Ok(F0Prediction {
            log_f0,
            normalized_log_f0,
            predicted_log_f0,
            f0,
        })
    }
}
