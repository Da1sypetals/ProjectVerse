use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use sovits_svc_mlx::audio::{load_audio_first_channel, write_wav_float};
use sovits_svc_mlx::inference::{InferenceOptions, SliceInferenceOptions, SovitsSvc};

#[derive(Debug, Parser)]
#[command(
    version,
    about = "Run so-vits-svc GAN and shallow-diffusion inference with MLX"
)]
struct Arguments {
    /// Input audio file supported by Babycat, sampled at 44.1 or 48 kHz.
    input: PathBuf,

    /// Directory containing the converted GAN, diffusion, ContentVec, and vocoder weights.
    #[arg(long, default_value = "artifacts")]
    artifact_dir: PathBuf,

    /// FCPE MLX safetensors checkpoint.
    #[arg(long, default_value = "../ckpt/fcpe.safetensors")]
    fcpe_checkpoint: PathBuf,

    /// Pitch shift in semitones.
    #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
    pitch_shift: f32,

    /// GAN latent noise scale.
    #[arg(long, default_value_t = 0.4)]
    noise_scale: f32,

    /// Use the GAN automatic F0 decoder.
    #[arg(long)]
    predict_f0: bool,

    /// Number of shallow-diffusion steps from the training schedule.
    #[arg(long, default_value_t = 100)]
    diffusion_steps: i32,

    /// DPM-Solver++ inference speedup divisor.
    #[arg(long, default_value_t = 10)]
    diffusion_speedup: i32,

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
    let input = load_audio_first_channel(&arguments.input)?;
    let mut model = SovitsSvc::load(
        arguments.artifact_dir.join("gan.safetensors"),
        arguments.artifact_dir.join("diffusion.safetensors"),
        arguments.artifact_dir.join("contentvec.safetensors"),
        &arguments.fcpe_checkpoint,
        arguments.artifact_dir.join("vocoder.safetensors"),
    )?;
    let inference_options = InferenceOptions {
        pitch_shift: arguments.pitch_shift,
        noise_scale: arguments.noise_scale,
        predict_f0: arguments.predict_f0,
        diffusion_steps: arguments.diffusion_steps,
        diffusion_speedup: arguments.diffusion_speedup,
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
}
