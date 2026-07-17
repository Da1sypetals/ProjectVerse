use anyhow::{Result, ensure};
use mlx_rs::Array;
use mlx_rs::module::Module;
use mlx_rs::nn::{self, Conv1d, Linear};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::{
    broadcast_to, clip, concatenate_axis, cos, exp, expm1, floor, log, log1p, sin, sqrt,
    stack_axis, sum_axis, tanh,
};

use crate::weights::Weights;

const MEL_BINS: i32 = 128;
const HIDDEN_CHANNELS: i32 = 256;
const RESIDUAL_CHANNELS: i32 = 512;
const RESIDUAL_LAYERS: usize = 20;

#[derive(Debug)]
struct NoiseSchedule {
    log_alphas: Array,
    total_steps: i32,
}

impl NoiseSchedule {
    fn new(betas: &Array, total_steps: i32) -> Result<Self> {
        let betas = betas.index(0..total_steps);
        let log_alphas = log(&(betas * -1.0_f32 + 1.0_f32))?.cumsum(0, false, true)? * 0.5_f32;
        Ok(Self {
            log_alphas,
            total_steps,
        })
    }

    fn marginal_log_mean_coeff(&self, time: &Array) -> Result<Array> {
        let position = time * self.total_steps as f32 - 1.0_f32;
        let lower_float = clip(floor(&position)?, (0.0_f32, (self.total_steps - 2) as f32))?;
        let upper = (&lower_float + 1.0_f32).as_type::<i32>()?;
        let lower = lower_float.as_type::<i32>()?;
        let lower_value = self.log_alphas.take(&lower)?;
        let upper_value = self.log_alphas.take(&upper)?;
        Ok(&lower_value + (position - lower_float) * (upper_value - &lower_value))
    }

    fn marginal_std(&self, time: &Array) -> Result<Array> {
        let log_alpha = self.marginal_log_mean_coeff(time)?;
        Ok(sqrt(&(exp(&(log_alpha * 2.0_f32))? * -1.0_f32 + 1.0_f32))?)
    }

    fn marginal_lambda(&self, time: &Array) -> Result<Array> {
        let log_alpha = self.marginal_log_mean_coeff(time)?;
        let log_std = log(&(exp(&(&log_alpha * 2.0_f32))? * -1.0_f32 + 1.0_f32))? * 0.5_f32;
        Ok(log_alpha - log_std)
    }
}

#[derive(Debug)]
struct ResidualBlock {
    dilated_conv: Conv1d,
    diffusion_projection: Linear,
    conditioner_projection: Conv1d,
    output_projection: Conv1d,
}

impl ResidualBlock {
    fn load(weights: &mut Weights, index: usize) -> Result<Self> {
        let prefix = format!("decoder.denoise_fn.residual_layers.{index}");
        Ok(Self {
            dilated_conv: weights.conv1d(
                &format!("{prefix}.dilated_conv"),
                RESIDUAL_CHANNELS,
                RESIDUAL_CHANNELS * 2,
                3,
                1,
                1,
                1,
                1,
            )?,
            diffusion_projection: weights.linear(
                &format!("{prefix}.diffusion_projection"),
                RESIDUAL_CHANNELS,
                RESIDUAL_CHANNELS,
                true,
            )?,
            conditioner_projection: weights.conv1d(
                &format!("{prefix}.conditioner_projection"),
                HIDDEN_CHANNELS,
                RESIDUAL_CHANNELS * 2,
                1,
                0,
                1,
                1,
                1,
            )?,
            output_projection: weights.conv1d(
                &format!("{prefix}.output_projection"),
                RESIDUAL_CHANNELS,
                RESIDUAL_CHANNELS * 2,
                1,
                0,
                1,
                1,
                1,
            )?,
        })
    }

    fn forward(
        &mut self,
        x: &Array,
        conditioner: &Array,
        diffusion_step: &Array,
    ) -> Result<(Array, Array)> {
        let diffusion_step = self
            .diffusion_projection
            .forward(diffusion_step)?
            .expand_dims(1)?;
        let conditioner = self.conditioner_projection.forward(conditioner)?;
        let y = self.dilated_conv.forward(&(x + diffusion_step))? + conditioner;
        let gate_filter = y.split(2, -1)?;
        let y = nn::sigmoid(&gate_filter[0])? * tanh(&gate_filter[1])?;
        let y = self.output_projection.forward(&y)?;
        let residual_skip = y.split(2, -1)?;
        Ok((
            (x + &residual_skip[0]) / 2.0_f32.sqrt(),
            residual_skip[1].clone(),
        ))
    }
}

#[derive(Debug)]
pub(super) struct WaveNet {
    input_projection: Conv1d,
    diffusion_mlp_in: Linear,
    diffusion_mlp_out: Linear,
    residual_layers: Vec<ResidualBlock>,
    skip_projection: Conv1d,
    output_projection: Conv1d,
}

#[derive(Debug)]
pub struct WaveNetTrace {
    pub input_projection: Array,
    pub diffusion_embedding: Array,
    pub mlp_in: Array,
    pub mlp_mish: Array,
    pub mlp_out: Array,
    pub residuals: Vec<Array>,
    pub skips: Vec<Array>,
    pub skip_projection: Array,
    pub output_projection: Array,
    pub output: Array,
}

impl WaveNet {
    fn load(weights: &mut Weights) -> Result<Self> {
        let residual_layers = (0..RESIDUAL_LAYERS)
            .map(|index| ResidualBlock::load(weights, index))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            input_projection: weights.conv1d(
                "decoder.denoise_fn.input_projection",
                MEL_BINS,
                RESIDUAL_CHANNELS,
                1,
                0,
                1,
                1,
                1,
            )?,
            diffusion_mlp_in: weights.linear(
                "decoder.denoise_fn.mlp.0",
                RESIDUAL_CHANNELS,
                RESIDUAL_CHANNELS * 4,
                true,
            )?,
            diffusion_mlp_out: weights.linear(
                "decoder.denoise_fn.mlp.2",
                RESIDUAL_CHANNELS * 4,
                RESIDUAL_CHANNELS,
                true,
            )?,
            residual_layers,
            skip_projection: weights.conv1d(
                "decoder.denoise_fn.skip_projection",
                RESIDUAL_CHANNELS,
                RESIDUAL_CHANNELS,
                1,
                0,
                1,
                1,
                1,
            )?,
            output_projection: weights.conv1d(
                "decoder.denoise_fn.output_projection",
                RESIDUAL_CHANNELS,
                MEL_BINS,
                1,
                0,
                1,
                1,
                1,
            )?,
        })
    }

    fn run(
        &mut self,
        spec: &Array,
        diffusion_step: &Array,
        cond: &Array,
        capture_trace: bool,
    ) -> Result<(Array, Option<WaveNetTrace>)> {
        let spec = spec.squeeze_axes(&[1])?.swap_axes(1, 2)?;
        let input_projection = self.input_projection.forward(&spec)?;
        let mut x = nn::relu(&input_projection)?;

        let half_dim = RESIDUAL_CHANNELS / 2;
        let frequencies = exp(&(Array::arange::<_, f32>(None, half_dim, None)?
            * (-10000.0_f32.ln() / (half_dim - 1) as f32)))?;
        let phase =
            diffusion_step.as_type::<f32>()?.expand_dims(-1)? * frequencies.expand_dims(0)?;
        let diffusion_embedding = concatenate_axis(&[sin(&phase)?, cos(&phase)?], -1)?;
        let mlp_in = self.diffusion_mlp_in.forward(&diffusion_embedding)?;
        let mlp_mish = nn::mish(&mlp_in)?;
        let embedding = self.diffusion_mlp_out.forward(&mlp_mish)?;

        let mut skips = Vec::with_capacity(self.residual_layers.len());
        let mut residuals = Vec::with_capacity(self.residual_layers.len());
        for layer in &mut self.residual_layers {
            let (next_x, skip) = layer.forward(&x, cond, &embedding)?;
            x = next_x;
            if capture_trace {
                residuals.push(x.clone());
            }
            skips.push(skip);
        }
        let skip_refs = skips.iter().collect::<Vec<_>>();
        x = sum_axis(stack_axis(&skip_refs, 0)?, 0, false)?
            / (self.residual_layers.len() as f32).sqrt();
        let skip_projection = self.skip_projection.forward(&x)?;
        x = nn::relu(&skip_projection)?;
        let output_projection = self.output_projection.forward(&x)?;
        let output = output_projection.swap_axes(1, 2)?.expand_dims(1)?;
        let trace = capture_trace.then(|| WaveNetTrace {
            input_projection,
            diffusion_embedding,
            mlp_in,
            mlp_mish,
            mlp_out: embedding,
            residuals,
            skips,
            skip_projection,
            output_projection,
            output: output.clone(),
        });
        Ok((output, trace))
    }

    pub fn forward(&mut self, spec: &Array, diffusion_step: &Array, cond: &Array) -> Result<Array> {
        Ok(self.run(spec, diffusion_step, cond, false)?.0)
    }

    pub fn forward_with_trace(
        &mut self,
        spec: &Array,
        diffusion_step: &Array,
        cond: &Array,
    ) -> Result<WaveNetTrace> {
        Ok(self
            .run(spec, diffusion_step, cond, true)?
            .1
            .expect("WaveNet trace capture must return a trace"))
    }
}

#[derive(Debug)]
pub(super) struct DiffusionSchedule {
    pub betas: Array,
    pub sqrt_alphas_cumprod: Array,
    pub sqrt_one_minus_alphas_cumprod: Array,
    pub spec_min: Array,
    pub spec_max: Array,
}

impl DiffusionSchedule {
    fn load(weights: &mut Weights) -> Result<Self> {
        weights.take("decoder.alphas_cumprod")?;
        weights.take("decoder.alphas_cumprod_prev")?;
        weights.take("decoder.log_one_minus_alphas_cumprod")?;
        weights.take("decoder.sqrt_recip_alphas_cumprod")?;
        weights.take("decoder.sqrt_recipm1_alphas_cumprod")?;
        weights.take("decoder.posterior_variance")?;
        weights.take("decoder.posterior_log_variance_clipped")?;
        weights.take("decoder.posterior_mean_coef1")?;
        weights.take("decoder.posterior_mean_coef2")?;
        Ok(Self {
            betas: weights.take("decoder.betas")?,
            sqrt_alphas_cumprod: weights.take("decoder.sqrt_alphas_cumprod")?,
            sqrt_one_minus_alphas_cumprod: weights.take("decoder.sqrt_one_minus_alphas_cumprod")?,
            spec_min: weights.take("decoder.spec_min")?,
            spec_max: weights.take("decoder.spec_max")?,
        })
    }
}

#[derive(Debug)]
pub(super) struct Unit2Mel {
    unit_embed: Linear,
    f0_embed: Linear,
    volume_embed: Linear,
    _aug_shift_embed: Linear,
    pub schedule: DiffusionSchedule,
    pub denoiser: WaveNet,
}

impl Unit2Mel {
    pub(super) fn load(weights: &mut Weights) -> Result<Self> {
        Ok(Self {
            unit_embed: weights.linear("unit_embed", 768, HIDDEN_CHANNELS, true)?,
            f0_embed: weights.linear("f0_embed", 1, HIDDEN_CHANNELS, true)?,
            volume_embed: weights.linear("volume_embed", 1, HIDDEN_CHANNELS, true)?,
            _aug_shift_embed: weights.linear("aug_shift_embed", 1, HIDDEN_CHANNELS, false)?,
            schedule: DiffusionSchedule::load(weights)?,
            denoiser: WaveNet::load(weights)?,
        })
    }

    pub fn condition(&mut self, units: &Array, f0: &Array, volume: &Array) -> Result<Array> {
        let units = self.unit_embed.forward(units)?;
        let f0 = self.f0_embed.forward(&log1p(&(f0 / 700.0_f32))?)?;
        let volume = self.volume_embed.forward(volume)?;
        Ok(units + f0 + volume)
    }

    pub fn q_sample(&self, x_start: &Array, timestep: &Array, noise: &Array) -> Result<Array> {
        let alpha = self
            .schedule
            .sqrt_alphas_cumprod
            .take(timestep)?
            .reshape(&[-1, 1, 1, 1])?;
        let sigma = self
            .schedule
            .sqrt_one_minus_alphas_cumprod
            .take(timestep)?
            .reshape(&[-1, 1, 1, 1])?;
        Ok(alpha * x_start + sigma * noise)
    }

    pub fn norm_spec(&self, spec: &Array) -> Result<Array> {
        Ok(
            (spec - &self.schedule.spec_min) / (&self.schedule.spec_max - &self.schedule.spec_min)
                * 2.0_f32
                - 1.0_f32,
        )
    }

    pub fn denorm_spec(&self, spec: &Array) -> Result<Array> {
        Ok(
            (spec + 1.0_f32) / 2.0_f32 * (&self.schedule.spec_max - &self.schedule.spec_min)
                + &self.schedule.spec_min,
        )
    }

    fn dpm_data_prediction(
        &mut self,
        schedule: &NoiseSchedule,
        x: &Array,
        time: &Array,
        cond: &Array,
    ) -> Result<Array> {
        let batch_time = broadcast_to(time, &[x.shape()[0]])?;
        let model_time =
            (batch_time - 1.0_f32 / schedule.total_steps as f32) * schedule.total_steps as f32;
        let noise = self.denoiser.forward(x, &model_time, cond)?;
        let alpha = exp(&schedule.marginal_log_mean_coeff(time)?)?;
        let sigma = schedule.marginal_std(time)?;
        Ok((x - sigma * noise) / alpha)
    }

    fn dpm_first_update(
        schedule: &NoiseSchedule,
        x: &Array,
        time_start: &Array,
        time_end: &Array,
        model_start: &Array,
    ) -> Result<Array> {
        let lambda_start = schedule.marginal_lambda(time_start)?;
        let lambda_end = schedule.marginal_lambda(time_end)?;
        let phi = expm1(&(-(lambda_end - lambda_start)))?;
        let sigma_start = schedule.marginal_std(time_start)?;
        let sigma_end = schedule.marginal_std(time_end)?;
        let alpha_end = exp(&schedule.marginal_log_mean_coeff(time_end)?)?;
        Ok(sigma_end / sigma_start * x - alpha_end * phi * model_start)
    }

    fn dpm_second_update(
        schedule: &NoiseSchedule,
        x: &Array,
        model_older: &Array,
        model_current: &Array,
        time_older: &Array,
        time_current: &Array,
        time_end: &Array,
    ) -> Result<Array> {
        let lambda_older = schedule.marginal_lambda(time_older)?;
        let lambda_current = schedule.marginal_lambda(time_current)?;
        let lambda_end = schedule.marginal_lambda(time_end)?;
        let previous_step = &lambda_current - lambda_older;
        let step = &lambda_end - &lambda_current;
        let derivative = &step / previous_step * (model_current - model_older);
        let phi = expm1(&(-step))?;
        let sigma_current = schedule.marginal_std(time_current)?;
        let sigma_end = schedule.marginal_std(time_end)?;
        let alpha_end = exp(&schedule.marginal_log_mean_coeff(time_end)?)?;
        Ok(sigma_end / sigma_current * x
            - &alpha_end * &phi * model_current
            - alpha_end * phi * derivative * 0.5_f32)
    }

    pub fn sample_dpm_solver_pp(
        &mut self,
        x: &Array,
        cond: &Array,
        k_step: i32,
        infer_speedup: i32,
    ) -> Result<Array> {
        ensure!(k_step >= 2, "k_step must be at least 2");
        ensure!(
            k_step <= self.schedule.betas.shape()[0],
            "k_step {k_step} exceeds diffusion schedule length {}",
            self.schedule.betas.shape()[0]
        );
        ensure!(infer_speedup > 0, "infer_speedup must be positive");
        let steps = k_step / infer_speedup;
        ensure!(steps >= 2, "DPM-Solver++ order 2 needs at least 2 steps");

        let schedule = NoiseSchedule::new(&self.schedule.betas, k_step)?;
        let timesteps = Array::linspace::<_, f32>(1.0_f32, 1.0_f32 / k_step as f32, steps + 1)?;
        let mut time_older = timesteps.index(0).expand_dims(0)?;
        let mut model_older = self.dpm_data_prediction(&schedule, x, &time_older, cond)?;
        model_older.eval()?;

        let mut time_current = timesteps.index(1).expand_dims(0)?;
        let mut current =
            Self::dpm_first_update(&schedule, x, &time_older, &time_current, &model_older)?;
        current.eval()?;
        let mut model_current =
            self.dpm_data_prediction(&schedule, &current, &time_current, cond)?;
        model_current.eval()?;

        for step_index in 2..=steps {
            let time_end = timesteps.index(step_index).expand_dims(0)?;
            current = Self::dpm_second_update(
                &schedule,
                &current,
                &model_older,
                &model_current,
                &time_older,
                &time_current,
                &time_end,
            )?;
            current.eval()?;
            if step_index < steps {
                model_older = model_current;
                model_current = self.dpm_data_prediction(&schedule, &current, &time_end, cond)?;
                model_current.eval()?;
                time_older = time_current;
                time_current = time_end;
            }
        }
        Ok(current)
    }
}
