use std::path::Path;

use anyhow::{Result, ensure};
use mlx_rs::Array;

use super::model::Unit2Mel;
use crate::weights::Weights;

#[derive(Debug)]
pub struct ShallowDiffusionRefiner {
    model: Unit2Mel,
}

impl ShallowDiffusionRefiner {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let mut weights = Weights::load(path)?;
        let model = Unit2Mel::load(&mut weights)?;
        weights.finish()?;
        Ok(Self { model })
    }

    pub fn refine(
        &mut self,
        units: &Array,
        f0: &Array,
        volume: &Array,
        initial_mel: &Array,
        noise: &Array,
        k_step: i32,
        infer_speedup: i32,
    ) -> Result<Array> {
        ensure!(k_step >= 2, "diffusion step count must be at least 2");
        ensure!(infer_speedup > 0, "diffusion speedup must be positive");
        let condition = self.model.condition(units, f0, volume)?;
        let normalized = self
            .model
            .norm_spec(initial_mel)?
            .swap_axes(1, 2)?
            .expand_dims(1)?;
        let timestep = Array::from_int(k_step - 1).reshape(&[1])?;
        let noisy = self.model.q_sample(&normalized, &timestep, noise)?;
        let refined = self
            .model
            .sample_dpm_solver_pp(&noisy, &condition, k_step, infer_speedup)?;
        self.model
            .denorm_spec(&refined.squeeze_axes(&[1])?.swap_axes(1, 2)?)
    }
}
