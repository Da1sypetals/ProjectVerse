use std::path::Path;

use anyhow::{Result, ensure};
use mlx_rs::Array;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::{broadcast_to, clip, sqrt, sum_axis};
use mlx_rs::random::normal;

use super::model::{Unit2Mel, WaveNetTrace};
use crate::weights::Weights;

const TIMESTEPS: i32 = 1000;
#[derive(Debug)]
pub struct FlowMatchingRefiner {
    model: Unit2Mel,
    t_k: Array,
    t_eps: f32,
    alphas_cumprod_ascending: Array,
}

#[derive(Debug)]
pub struct FlowMatchingTrace {
    pub condition: Array,
    pub normalized: Array,
    pub entry: Array,
    pub first_wavenet: WaveNetTrace,
    pub times: Vec<Array>,
    pub time_steps: Vec<Array>,
    pub states: Vec<Array>,
    pub d: Vec<Array>,
    pub x_vp: Vec<Array>,
    pub k: Vec<Array>,
    pub epsilon: Vec<Array>,
    pub endpoint: Vec<Array>,
    pub velocity: Vec<Array>,
    pub final_state: Array,
    pub final_d: Array,
    pub final_x_vp: Array,
    pub final_k: Array,
    pub final_epsilon: Array,
    pub final_endpoint: Array,
    pub mel: Array,
}

struct EndpointPrediction {
    d: Array,
    x_vp: Array,
    k: Array,
    wavenet: Option<WaveNetTrace>,
    epsilon: Array,
    endpoint: Array,
}

impl FlowMatchingRefiner {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let mut weights = Weights::load(path)?;
        let a_k = weights.take("a_k")?;
        let b_k = weights.take("b_k")?;
        let t_k = weights.take("t_k")?;
        let alphas_cumprod_ascending = weights.take("_flow.alphas_cumprod_ascending")?;
        let timesteps = weights.take("_flow.timesteps")?;
        let t_eps = weights.take("_flow.t_eps")?;
        let default_ode_steps = weights.take("_flow.default_ode_steps")?;
        let model = Unit2Mel::load(&mut weights)?;
        weights.finish()?;
        ensure!(a_k.size() == 1, "a_k must be scalar");
        ensure!(b_k.size() == 1, "b_k must be scalar");
        ensure!(t_k.size() == 1, "t_k must be scalar");
        let expected_t_k =
            a_k.try_item::<f32>()? / (a_k.try_item::<f32>()? + b_k.try_item::<f32>()?);
        ensure!(
            (t_k.try_item::<f32>()? - expected_t_k).abs() <= f32::EPSILON,
            "t_k does not match a_k / (a_k + b_k)"
        );
        ensure!(
            alphas_cumprod_ascending.shape() == [TIMESTEPS],
            "invalid ascending alpha schedule shape {:?}",
            alphas_cumprod_ascending.shape()
        );
        ensure!(
            timesteps.try_item::<i32>()? == TIMESTEPS,
            "Flow Matching checkpoint timesteps do not match the model"
        );
        ensure!(
            default_ode_steps.try_item::<i32>()? >= 1,
            "Flow Matching default ODE step count must be positive"
        );
        let t_eps = t_eps.try_item::<f32>()?;
        ensure!(
            t_eps > 0.0 && t_eps < 1.0,
            "Flow Matching t_eps must be between zero and one"
        );
        Ok(Self {
            model,
            t_k,
            t_eps,
            alphas_cumprod_ascending,
        })
    }

    fn predict_endpoint(
        &mut self,
        x: &Array,
        time: &Array,
        condition: &Array,
        capture_trace: bool,
    ) -> Result<EndpointPrediction> {
        let one_minus_time = time * -1.0_f32 + 1.0_f32;
        let denominator_squared = time * time + &one_minus_time * &one_minus_time;
        let denominator = sqrt(&denominator_squared)?;
        let x_vp = x / denominator.reshape(&[-1, 1, 1, 1])?;
        let bar_alpha = time * time / denominator_squared;
        let positions = sum_axis(
            self.alphas_cumprod_ascending
                .expand_dims(0)?
                .lt(bar_alpha.expand_dims(-1)?)?
                .as_type::<f32>()?,
            -1,
            false,
        )?;
        let positions = clip(positions, (1.0_f32, (TIMESTEPS - 1) as f32))?;
        let lower = (&positions - 1.0_f32).as_type::<i32>()?;
        let upper = positions.as_type::<i32>()?;
        let alpha_lower = self.alphas_cumprod_ascending.take(&lower)?;
        let alpha_upper = self.alphas_cumprod_ascending.take(&upper)?;
        let weight =
            (&bar_alpha - &alpha_lower) / clip(alpha_upper - &alpha_lower, (1.0e-12_f32, ()))?;
        let k_lower = (Array::from_int(TIMESTEPS - 1) - lower).as_type::<f32>()?;
        let k_upper = (Array::from_int(TIMESTEPS - 1) - upper).as_type::<f32>()?;
        let diffusion_step = &k_lower + weight * (k_upper - &k_lower);
        let wavenet = if capture_trace {
            Some(
                self.model
                    .denoiser
                    .forward_with_trace(&x_vp, &diffusion_step, condition)?,
            )
        } else {
            None
        };
        let epsilon = if let Some(trace) = &wavenet {
            trace.output.clone()
        } else {
            self.model
                .denoiser
                .forward(&x_vp, &diffusion_step, condition)?
        };
        let endpoint = (x - one_minus_time.reshape(&[-1, 1, 1, 1])? * &epsilon)
            / clip(time, (1.0e-5_f32, ()))?.reshape(&[-1, 1, 1, 1])?;
        Ok(EndpointPrediction {
            d: denominator,
            x_vp,
            k: diffusion_step,
            wavenet,
            epsilon,
            endpoint,
        })
    }

    fn run_with_noise(
        &mut self,
        units: &Array,
        f0: &Array,
        volume: &Array,
        gan_mel: &Array,
        noise: &Array,
        n_steps: i32,
        capture_trace: bool,
    ) -> Result<(Array, Option<FlowMatchingTrace>)> {
        ensure!(n_steps >= 1, "flow matching step count must be positive");
        let condition = self.model.condition(units, f0, volume)?;
        let normalized = self
            .model
            .norm_spec(gan_mel)?
            .swap_axes(1, 2)?
            .expand_dims(1)?;
        ensure!(
            noise.shape() == normalized.shape(),
            "flow matching noise shape {:?} does not match normalized mel {:?}",
            noise.shape(),
            normalized.shape()
        );
        let t_k = self.t_k.reshape(&[1, 1, 1, 1])?;
        let mut current = &t_k * &normalized + (t_k * -1.0_f32 + 1.0_f32) * noise;
        let entry = current.clone();
        let t_start = self.t_k.try_item::<f32>()?;
        let t_end = 1.0_f32 - self.t_eps;
        ensure!(
            t_end > t_start,
            "Flow Matching end time must be greater than t_k"
        );
        let timesteps = Array::linspace::<_, f32>(t_start, t_end, n_steps + 1)?;
        let mut first_wavenet = None;
        let mut times = Vec::with_capacity(n_steps as usize);
        let mut time_steps = Vec::with_capacity(n_steps as usize);
        let mut states = Vec::with_capacity(n_steps as usize + 1);
        let mut d = Vec::with_capacity(n_steps as usize);
        let mut x_vp = Vec::with_capacity(n_steps as usize);
        let mut k = Vec::with_capacity(n_steps as usize);
        let mut epsilon = Vec::with_capacity(n_steps as usize);
        let mut endpoint = Vec::with_capacity(n_steps as usize);
        let mut velocities = Vec::with_capacity(n_steps as usize);
        if capture_trace {
            states.push(entry.clone());
        }
        for step in 0..n_steps {
            let time = broadcast_to(&timesteps.index(step).reshape(&[1])?, &[current.shape()[0]])?;
            let mut prediction =
                self.predict_endpoint(&current, &time, &condition, capture_trace && step == 0)?;
            let velocity = (&prediction.endpoint - &current)
                / clip(&time * -1.0_f32 + 1.0_f32, (1.0e-5_f32, ()))?.reshape(&[-1, 1, 1, 1])?;
            let time_step = timesteps.index(step + 1) - timesteps.index(step);
            if capture_trace && step == 0 {
                first_wavenet = prediction.wavenet.take();
            }
            current = current + &velocity * &time_step;
            current.eval()?;
            if capture_trace {
                times.push(time);
                time_steps.push(time_step);
                d.push(prediction.d);
                x_vp.push(prediction.x_vp);
                k.push(prediction.k);
                epsilon.push(prediction.epsilon);
                endpoint.push(prediction.endpoint);
                velocities.push(velocity);
                states.push(current.clone());
            }
        }
        let final_state = current.clone();
        let last_time = broadcast_to(
            &timesteps.index(n_steps).reshape(&[1])?,
            &[current.shape()[0]],
        )?;
        let final_prediction = self.predict_endpoint(&current, &last_time, &condition, false)?;
        let mel = self.model.denorm_spec(
            &final_prediction
                .endpoint
                .squeeze_axes(&[1])?
                .swap_axes(1, 2)?,
        )?;
        let trace = if capture_trace {
            Some(FlowMatchingTrace {
                condition,
                normalized,
                entry,
                first_wavenet: first_wavenet
                    .expect("the first traced prediction must contain a WaveNet trace"),
                times,
                time_steps,
                states,
                d,
                x_vp,
                k,
                epsilon,
                endpoint,
                velocity: velocities,
                final_state,
                final_d: final_prediction.d,
                final_x_vp: final_prediction.x_vp,
                final_k: final_prediction.k,
                final_epsilon: final_prediction.epsilon,
                final_endpoint: final_prediction.endpoint,
                mel: mel.clone(),
            })
        } else {
            None
        };
        Ok((mel, trace))
    }

    pub fn refine_with_noise(
        &mut self,
        units: &Array,
        f0: &Array,
        volume: &Array,
        gan_mel: &Array,
        noise: &Array,
        n_steps: i32,
    ) -> Result<Array> {
        Ok(self
            .run_with_noise(units, f0, volume, gan_mel, noise, n_steps, false)?
            .0)
    }

    pub fn trace_with_noise(
        &mut self,
        units: &Array,
        f0: &Array,
        volume: &Array,
        gan_mel: &Array,
        noise: &Array,
        n_steps: i32,
    ) -> Result<FlowMatchingTrace> {
        Ok(self
            .run_with_noise(units, f0, volume, gan_mel, noise, n_steps, true)?
            .1
            .expect("trace capture must return a trace"))
    }

    pub fn refine(
        &mut self,
        units: &Array,
        f0: &Array,
        volume: &Array,
        gan_mel: &Array,
        n_steps: i32,
    ) -> Result<Array> {
        let normalized_shape = [
            gan_mel.shape()[0],
            1,
            gan_mel.shape()[2],
            gan_mel.shape()[1],
        ];
        let noise = normal::<f32>(&normalized_shape, None, None, None)?;
        self.refine_with_noise(units, f0, volume, gan_mel, &noise, n_steps)
    }
}
