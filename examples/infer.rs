use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use sovits_svc_mlx::audio::{load_audio, write_wav_float};
use sovits_svc_mlx::inference::{InferenceOptions, Refiner, SliceInferenceOptions, SovitsSvc};

#[derive(Debug, Parser)]
#[command(
    version,
    about = "Run so-vits-svc GAN inference with optional MLX refiners"
)]
struct Arguments {
    /// Input audio file supported by the statically linked FFmpeg build, sampled at 44.1 or 48 kHz.
    input: PathBuf,

    /// Converted GAN checkpoint.
    #[arg(long, default_value = "../ckpt/mlx/gan/e83_s2400/model.safetensors")]
    gan_checkpoint: PathBuf,

    /// Converted shallow-diffusion checkpoint.
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

    /// FCPE MLX safetensors checkpoint.
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

    /// Refine the GAN output with shallow diffusion.
    #[arg(long, conflicts_with = "flow_matching")]
    shallow_diffusion: bool,

    /// Refine the GAN output with Flow Matching.
    #[arg(long, conflicts_with = "shallow_diffusion")]
    flow_matching: bool,

    /// Number of shallow-diffusion steps from the training schedule.
    #[arg(long, default_value_t = 100)]
    diffusion_steps: i32,

    /// DPM-Solver++ inference speedup divisor.
    #[arg(long, default_value_t = 10)]
    diffusion_speedup: i32,

    /// Number of Flow Matching Euler integration steps.
    #[arg(long, default_value_t = 50)]
    flow_matching_steps: i32,

    /// Output loudness-envelope adjustment strength.
    #[arg(long, default_value_t = 1.0)]
    loudness_envelope_adjustment: f32,

    /// Re-encode the GAN waveform before diffusion.
    #[arg(long)]
    second_encoding: bool,

    /// Disable the silence slicing used by the original Python inference entry point.
    #[arg(
        long = "no-slicing",
        action = clap::ArgAction::SetFalse,
        default_value_t = true
    )]
    slicing: bool,

    /// Silence threshold in dB.
    #[arg(long, default_value_t = -40.0, allow_hyphen_values = true)]
    threshold_db: f32,

    /// Context padding around each non-silent slice in seconds.
    #[arg(long, default_value_t = 0.5)]
    padding_seconds: f32,

    /// Maximum chunk duration in seconds; zero disables chunk splitting.
    #[arg(long, default_value_t = 0.0)]
    clip_seconds: f32,

    /// Chunk overlap duration in seconds.
    #[arg(long, default_value_t = 0.0)]
    crossfade_seconds: f32,

    /// Fraction of the overlap used for the linear crossfade.
    #[arg(long, default_value_t = 0.75)]
    crossfade_ratio: f32,
}

fn output_path(input: &Path) -> Result<PathBuf> {
    let stem = input
        .file_stem()
        .context("input audio filename has no stem")?;
    let mut filename = stem.to_os_string();
    filename.push("-converted.wav");
    Ok(input.with_file_name(filename))
}

fn main() -> Result<()> {
    let arguments = Arguments::parse();
    let output_path = output_path(&arguments.input)?;
    let input = load_audio(&arguments.input)?;
    let mut model = SovitsSvc::load(
        &arguments.gan_checkpoint,
        &arguments.shallow_diffusion_checkpoint,
        &arguments.flow_matching_checkpoint,
        &arguments.contentvec_checkpoint,
        &arguments.fcpe_checkpoint,
        &arguments.vocoder_checkpoint,
    )?;
    let refiner = if arguments.shallow_diffusion {
        Refiner::ShallowDiffusion
    } else if arguments.flow_matching {
        Refiner::FlowMatching
    } else {
        Refiner::None
    };
    let inference_options = InferenceOptions {
        input_gain_db: arguments.input_gain_db,
        pitch_shift: arguments.pitch_shift,
        noise_scale: arguments.noise_scale,
        predict_f0: arguments.predict_f0,
        refiner,
        diffusion_steps: arguments.diffusion_steps,
        diffusion_speedup: arguments.diffusion_speedup,
        flow_matching_steps: arguments.flow_matching_steps,
        loudness_envelope_adjustment: arguments.loudness_envelope_adjustment,
        second_encoding: arguments.second_encoding,
    };
    let slice_options = SliceInferenceOptions {
        threshold_db: arguments.threshold_db,
        padding_seconds: arguments.padding_seconds,
        clip_seconds: arguments.clip_seconds,
        crossfade_seconds: arguments.crossfade_seconds,
        crossfade_ratio: arguments.crossfade_ratio,
    };

    let started = Instant::now();
    let (output, frame_count) = if arguments.slicing {
        (
            model.infer_sliced(
                &input.samples,
                input.sample_rate as i32,
                &inference_options,
                &slice_options,
            )?,
            None,
        )
    } else {
        let inferred = model.infer(&input.samples, input.sample_rate as i32, &inference_options)?;
        let frame_count = inferred.f0.shape()[1];
        (inferred.audio, Some(frame_count))
    };
    let sample_count = output.shape()[1];
    write_wav_float(&output_path, &output, 44_100)?;
    if let Some(frame_count) = frame_count {
        println!(
            "inference completed in {:.3}s: frames={frame_count}, samples={sample_count}, output={}",
            started.elapsed().as_secs_f64(),
            output_path.display()
        );
    } else {
        println!(
            "sliced inference completed in {:.3}s: samples={sample_count}, output={}",
            started.elapsed().as_secs_f64(),
            output_path.display()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_enable_slicing_and_derive_output_path() {
        let arguments = Arguments::try_parse_from(["infer", "voice.demo.flac"]).unwrap();
        assert!(arguments.slicing);
        assert_eq!(arguments.input_gain_db, 0);
        assert!(!arguments.shallow_diffusion);
        assert!(!arguments.flow_matching);
        assert_eq!(
            arguments.gan_checkpoint,
            PathBuf::from("../ckpt/mlx/gan/e83_s2400/model.safetensors")
        );
        assert_eq!(
            output_path(&arguments.input).unwrap(),
            PathBuf::from("voice.demo-converted.wav")
        );
    }

    #[test]
    fn no_slicing_flag_disables_slicing() {
        let arguments = Arguments::try_parse_from(["infer", "voice.wav", "--no-slicing"]).unwrap();
        assert!(!arguments.slicing);
    }

    #[test]
    fn input_gain_accepts_only_integer_values_in_range() {
        assert!(Arguments::try_parse_from(["infer", "voice.wav", "--input-gain", "-12"]).is_ok());
        assert!(Arguments::try_parse_from(["infer", "voice.wav", "--input-gain", "12"]).is_ok());
        assert!(Arguments::try_parse_from(["infer", "voice.wav", "--input-gain", "-13"]).is_err());
        assert!(Arguments::try_parse_from(["infer", "voice.wav", "--input-gain", "13"]).is_err());
        assert!(Arguments::try_parse_from(["infer", "voice.wav", "--input-gain", "1.5"]).is_err());
    }

    #[test]
    fn shallow_diffusion_flag_enables_diffusion() {
        let arguments =
            Arguments::try_parse_from(["infer", "voice.wav", "--shallow-diffusion"]).unwrap();
        assert!(arguments.shallow_diffusion);
    }

    #[test]
    fn flow_matching_flag_enables_flow_matching() {
        let arguments =
            Arguments::try_parse_from(["infer", "voice.wav", "--flow-matching"]).unwrap();
        assert!(arguments.flow_matching);
    }
}
