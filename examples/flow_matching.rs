use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use clap::{Args, Parser, Subcommand};
use mlx_rs::Array;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::{abs, max};
use sovits_svc_mlx::audio::{load_audio_first_channel, write_wav_float};
use sovits_svc_mlx::inference::{InferenceOptions, Refiner, SovitsSvc};
use sovits_svc_mlx::refine::FlowMatchingRefiner;

#[derive(Debug, Parser)]
#[command(
    version,
    about = "Run and verify so-vits-svc Flow Matching inference with MLX"
)]
struct Arguments {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Infer(InferArguments),
    VerifyReference(VerifyReferenceArguments),
}

#[derive(Debug, Args)]
struct InferArguments {
    /// Input audio supported by Babycat, sampled at 44.1 or 48 kHz.
    input: PathBuf,

    /// Converted opencpop GAN checkpoint.
    #[arg(long, default_value = "../ckpt/mlx/gan/opencpop/model.safetensors")]
    gan_checkpoint: PathBuf,

    /// Converted shallow-diffusion checkpoint loaded for the shared runtime.
    #[arg(
        long,
        default_value = "../ckpt/mlx/refine/shallow_diffusion/guan/model.safetensors"
    )]
    shallow_diffusion_checkpoint: PathBuf,

    /// Converted Flow Matching checkpoint.
    #[arg(
        long,
        default_value = "../ckpt/mlx/refine/flow_matching/opencpop/model.safetensors"
    )]
    flow_matching_checkpoint: PathBuf,

    /// Converted ContentVec checkpoint.
    #[arg(
        long,
        default_value = "../ckpt/mlx/encoder/contentvec/model.safetensors"
    )]
    contentvec_checkpoint: PathBuf,

    /// Converted FCPE checkpoint.
    #[arg(long, default_value = "../ckpt/mlx/pitch/fcpe/model.safetensors")]
    fcpe_checkpoint: PathBuf,

    /// Converted NSF-HiFiGAN checkpoint.
    #[arg(
        long,
        default_value = "../ckpt/mlx/vocoder/nsf_hifigan/model.safetensors"
    )]
    vocoder_checkpoint: PathBuf,

    /// Pitch shift in semitones.
    #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
    pitch_shift: f32,

    /// Input gain in dB.
    #[arg(
        long = "input-gain",
        default_value_t = 0,
        allow_hyphen_values = true,
        value_parser = clap::value_parser!(i32).range(-12..=12)
    )]
    input_gain_db: i32,

    /// GAN latent noise scale.
    #[arg(long, default_value_t = 0.4)]
    noise_scale: f32,

    /// Use the GAN automatic F0 decoder.
    #[arg(long)]
    predict_f0: bool,

    /// Number of Flow Matching Euler integration steps.
    #[arg(long, default_value_t = 50)]
    flow_matching_steps: i32,

    /// Output loudness-envelope adjustment strength.
    #[arg(long, default_value_t = 1.0)]
    loudness_envelope_adjustment: f32,
}

#[derive(Debug, Args)]
struct VerifyReferenceArguments {
    /// Converted Flow Matching checkpoint.
    #[arg(
        long,
        default_value = "../ckpt/mlx/refine/flow_matching/opencpop/model.safetensors"
    )]
    flow_matching_checkpoint: PathBuf,

    /// Converted deterministic PyTorch reference tensors.
    #[arg(
        long,
        default_value = "../ckpt/mlx/references/opencpop_flow_matching/tensors.safetensors"
    )]
    reference_tensors: PathBuf,

    /// PyTorch WaveNet stage tensors for the first ODE step.
    #[arg(
        long,
        default_value = "../ckpt/mlx/references/opencpop_flow_matching/wavenet_stages.safetensors"
    )]
    wavenet_stages: PathBuf,

    /// Maximum accepted absolute error in the normalized Flow Matching state.
    #[arg(long, default_value_t = 1.0e-3)]
    max_absolute_error: f32,

    /// Maximum accepted absolute error against the PyTorch CPU WaveNet stages.
    #[arg(long, default_value_t = 1.0e-4)]
    wavenet_max_absolute_error: f32,
}

fn output_path(input: &Path) -> Result<PathBuf> {
    let stem = input
        .file_stem()
        .context("input audio filename has no stem")?;
    let mut filename = stem.to_os_string();
    filename.push("-flow-matching.wav");
    Ok(input.with_file_name(filename))
}

fn infer(arguments: InferArguments) -> Result<()> {
    ensure!(
        arguments.flow_matching_steps >= 1,
        "flow matching step count must be positive"
    );
    let output_path = output_path(&arguments.input)?;
    let input = load_audio_first_channel(&arguments.input)?;
    let mut model = SovitsSvc::load(
        arguments.gan_checkpoint,
        arguments.shallow_diffusion_checkpoint,
        arguments.flow_matching_checkpoint,
        arguments.contentvec_checkpoint,
        arguments.fcpe_checkpoint,
        arguments.vocoder_checkpoint,
    )?;
    let started = Instant::now();
    let output = model.infer(
        &input.samples,
        input.sample_rate as i32,
        &InferenceOptions {
            input_gain_db: arguments.input_gain_db,
            pitch_shift: arguments.pitch_shift,
            noise_scale: arguments.noise_scale,
            predict_f0: arguments.predict_f0,
            refiner: Refiner::FlowMatching,
            diffusion_steps: 100,
            diffusion_speedup: 10,
            flow_matching_steps: arguments.flow_matching_steps,
            loudness_envelope_adjustment: arguments.loudness_envelope_adjustment,
            second_encoding: false,
        },
    )?;
    write_wav_float(&output_path, &output.audio, 44_100)?;
    println!(
        "Flow Matching inference completed in {:.3}s: frames={}, samples={}, output={}",
        started.elapsed().as_secs_f64(),
        output.f0.shape()[1],
        output.audio.shape()[1],
        output_path.display()
    );
    Ok(())
}

fn verify_reference(arguments: VerifyReferenceArguments) -> Result<()> {
    ensure!(
        arguments.max_absolute_error > 0.0,
        "maximum absolute error must be positive"
    );
    ensure!(
        arguments.wavenet_max_absolute_error > 0.0,
        "WaveNet maximum absolute error must be positive"
    );
    let mut reference = Array::load_safetensors(&arguments.reference_tensors)
        .with_context(|| format!("failed to load {}", arguments.reference_tensors.display()))?;
    let mut wavenet_reference = Array::load_safetensors(&arguments.wavenet_stages)
        .with_context(|| format!("failed to load {}", arguments.wavenet_stages.display()))?;
    let content = reference
        .remove("features.c")
        .context("reference is missing features.c")?
        .swap_axes(1, 2)?;
    let f0 = reference
        .remove("features.f0")
        .context("reference is missing features.f0")?
        .expand_dims(-1)?;
    let volume = reference
        .remove("features.vol")
        .context("reference is missing features.vol")?
        .expand_dims(-1)?;
    let gan_mel = reference
        .remove("gan.mel")
        .context("reference is missing gan.mel")?;
    let noise = reference
        .remove("fm.eps")
        .context("reference is missing fm.eps")?;
    let mut refiner = FlowMatchingRefiner::load(arguments.flow_matching_checkpoint)?;
    let trace = refiner.trace_with_noise(&content, &f0, &volume, &gan_mel, &noise, 50)?;
    let wavenet = trace.first_wavenet;
    let mut wavenet_checks = vec![
        (
            "wavenet.input_projection".to_owned(),
            wavenet.input_projection,
            "input_projection".to_owned(),
        ),
        (
            "wavenet.diffusion_embedding".to_owned(),
            wavenet.diffusion_embedding,
            "diffusion_embedding".to_owned(),
        ),
        (
            "wavenet.mlp.0".to_owned(),
            wavenet.mlp_in,
            "mlp.0".to_owned(),
        ),
        (
            "wavenet.mlp.1".to_owned(),
            wavenet.mlp_mish,
            "mlp.1".to_owned(),
        ),
        (
            "wavenet.mlp.2".to_owned(),
            wavenet.mlp_out,
            "mlp.2".to_owned(),
        ),
    ];
    for (index, (residual, skip)) in wavenet.residuals.into_iter().zip(wavenet.skips).enumerate() {
        wavenet_checks.push((
            format!("wavenet.residual_layers.{index}.residual"),
            residual,
            format!("residual_layers.{index}.residual"),
        ));
        wavenet_checks.push((
            format!("wavenet.residual_layers.{index}.skip"),
            skip,
            format!("residual_layers.{index}.skip"),
        ));
    }
    wavenet_checks.push((
        "wavenet.skip_projection".to_owned(),
        wavenet.skip_projection,
        "skip_projection".to_owned(),
    ));
    wavenet_checks.push((
        "wavenet.output_projection".to_owned(),
        wavenet.output_projection,
        "output_projection".to_owned(),
    ));
    wavenet_checks.push((
        "wavenet.output".to_owned(),
        wavenet.output,
        "output".to_owned(),
    ));
    let expected_timesteps = reference
        .remove("fm.ts")
        .context("reference is missing fm.ts")?;
    ensure!(
        trace.times.len() == 50
            && trace.time_steps.len() == 50
            && trace.states.len() == 51
            && trace.d.len() == 50
            && trace.x_vp.len() == 50
            && trace.k.len() == 50
            && trace.epsilon.len() == 50
            && trace.endpoint.len() == 50
            && trace.velocity.len() == 50,
        "Flow Matching trace has inconsistent step counts"
    );
    let checks = [
        (
            "condition",
            trace.condition.swap_axes(1, 2)?,
            "fm.cond",
            Some(1.0e-4_f32),
        ),
        (
            "normalized",
            trace.normalized,
            "fm.s_n",
            Some(arguments.max_absolute_error),
        ),
        (
            "entry",
            trace.entry,
            "fm.z_k",
            Some(arguments.max_absolute_error),
        ),
        (
            "final_d",
            trace.final_d,
            "fm.d_last",
            Some(arguments.max_absolute_error),
        ),
        (
            "final_x_vp",
            trace.final_x_vp,
            "fm.x_vp_last",
            Some(arguments.max_absolute_error),
        ),
        (
            "final_k",
            trace.final_k,
            "fm.k_last",
            Some(arguments.max_absolute_error),
        ),
        (
            "final_epsilon",
            trace.final_epsilon,
            "fm.eps_hat_last",
            None,
        ),
        (
            "final_endpoint",
            trace.final_endpoint,
            "fm.y_out_n",
            Some(arguments.max_absolute_error),
        ),
        (
            "mel",
            trace.mel,
            "fm.mel_hat",
            Some(arguments.max_absolute_error * 7.0_f32),
        ),
    ];
    let mut maximum_wavenet_error = 0.0_f32;
    let mut maximum_checked_error = 0.0_f32;
    let mut maximum_diagnostic_error = 0.0_f32;
    let mut mel_error = 0.0_f32;
    for (name, actual, expected_name) in wavenet_checks {
        let expected = wavenet_reference
            .remove(&expected_name)
            .with_context(|| format!("WaveNet reference is missing {expected_name}"))?;
        let error = max(abs(actual - expected)?, false)?.try_item::<f32>()?;
        maximum_wavenet_error = maximum_wavenet_error.max(error);
        println!("{name}: max_abs={error:.8}");
    }
    ensure!(
        maximum_wavenet_error <= arguments.wavenet_max_absolute_error,
        "WaveNet max_abs={maximum_wavenet_error:.8} exceeds {:.8}",
        arguments.wavenet_max_absolute_error
    );
    let initial_state_error = max(
        abs(trace.states[0].clone()
            - reference
                .remove("fm.ode_x.0")
                .context("reference is missing fm.ode_x.0")?)?,
        false,
    )?
    .try_item::<f32>()?;
    maximum_checked_error = maximum_checked_error.max(initial_state_error);
    println!("ode.0.state: max_abs={initial_state_error:.8}");
    ensure!(
        initial_state_error <= arguments.max_absolute_error,
        "ode.0.state max_abs={initial_state_error:.8} exceeds {:.8}",
        arguments.max_absolute_error
    );
    for step in 0..50 {
        let expected_time = expected_timesteps.index(step).reshape(&[1])?;
        let expected_time_step =
            expected_timesteps.index(step + 1) - expected_timesteps.index(step);
        let step_checks = [
            (
                format!("ode.{step}.time"),
                trace.times[step as usize].clone(),
                expected_time,
                Some(1.0e-6_f32),
            ),
            (
                format!("ode.{step}.dt"),
                trace.time_steps[step as usize].clone(),
                expected_time_step,
                Some(1.0e-6_f32),
            ),
            (
                format!("ode.{step}.d"),
                trace.d[step as usize].clone(),
                reference
                    .remove(&format!("fm.ode_d.{step}"))
                    .with_context(|| format!("reference is missing fm.ode_d.{step}"))?,
                Some(arguments.max_absolute_error),
            ),
            (
                format!("ode.{step}.x_vp"),
                trace.x_vp[step as usize].clone(),
                reference
                    .remove(&format!("fm.ode_x_vp.{step}"))
                    .with_context(|| format!("reference is missing fm.ode_x_vp.{step}"))?,
                Some(arguments.max_absolute_error),
            ),
            (
                format!("ode.{step}.k"),
                trace.k[step as usize].clone(),
                reference
                    .remove(&format!("fm.ode_k.{step}"))
                    .with_context(|| format!("reference is missing fm.ode_k.{step}"))?,
                Some(arguments.max_absolute_error),
            ),
            (
                format!("ode.{step}.epsilon"),
                trace.epsilon[step as usize].clone(),
                reference
                    .remove(&format!("fm.ode_eps_hat.{step}"))
                    .with_context(|| format!("reference is missing fm.ode_eps_hat.{step}"))?,
                None,
            ),
            (
                format!("ode.{step}.endpoint"),
                trace.endpoint[step as usize].clone(),
                reference
                    .remove(&format!("fm.ode_y_hat.{step}"))
                    .with_context(|| format!("reference is missing fm.ode_y_hat.{step}"))?,
                Some(arguments.max_absolute_error),
            ),
            (
                format!("ode.{step}.velocity"),
                trace.velocity[step as usize].clone(),
                reference
                    .remove(&format!("fm.ode_v.{step}"))
                    .with_context(|| format!("reference is missing fm.ode_v.{step}"))?,
                None,
            ),
            (
                format!("ode.{step}.updated"),
                trace.states[step as usize + 1].clone(),
                reference
                    .remove(&format!("fm.ode_x.{}", step + 1))
                    .with_context(|| format!("reference is missing fm.ode_x.{}", step + 1))?,
                Some(arguments.max_absolute_error),
            ),
        ];
        for (name, actual, expected, tolerance) in step_checks {
            let error = max(abs(actual - expected)?, false)?.try_item::<f32>()?;
            println!("{name}: max_abs={error:.8}");
            if let Some(tolerance) = tolerance {
                maximum_checked_error = maximum_checked_error.max(error);
                ensure!(
                    error <= tolerance,
                    "{name} max_abs={error:.8} exceeds {tolerance:.8}"
                );
            } else {
                maximum_diagnostic_error = maximum_diagnostic_error.max(error);
            }
        }
    }
    for (name, actual, expected_name, tolerance) in checks {
        let expected = reference
            .remove(expected_name)
            .with_context(|| format!("reference is missing {expected_name}"))?;
        let error = max(abs(actual - expected)?, false)?.try_item::<f32>()?;
        println!("{name}: max_abs={error:.8}");
        if expected_name == "fm.mel_hat" {
            mel_error = error;
            let tolerance = tolerance.context("mel check is missing its tolerance")?;
            ensure!(
                error <= tolerance,
                "{name} max_abs={error:.8} exceeds {tolerance:.8}"
            );
        } else if let Some(tolerance) = tolerance {
            maximum_checked_error = maximum_checked_error.max(error);
            ensure!(
                error <= tolerance,
                "{name} max_abs={error:.8} exceeds {tolerance:.8}"
            );
        } else {
            maximum_diagnostic_error = maximum_diagnostic_error.max(error);
        }
    }
    println!(
        "Flow Matching reference aligned: checked_max_abs={maximum_checked_error:.8}, \
         mel_max_abs={mel_error:.8}, wavenet_max_abs={maximum_wavenet_error:.8}, \
         diagnostic_max_abs={maximum_diagnostic_error:.8}"
    );
    Ok(())
}

fn main() -> Result<()> {
    match Arguments::parse().command {
        Command::Infer(arguments) => infer(arguments),
        Command::VerifyReference(arguments) => verify_reference(arguments),
    }
}
