use std::io::Cursor;
use std::path::Path;

use anyhow::{Context, Result, ensure};
use hound::{SampleFormat, WavSpec, WavWriter};
use mlx_rs::module::Module;
use mlx_rs::nn::Conv1d;
use mlx_rs::ops::indexing::{IndexOp, IntoStrideBy};
use mlx_rs::ops::{
    as_strided, broadcast_to, ceil, clip, clip_device, concatenate_axis, floor, maximum, mean_axis,
    pad, power, sqrt, square, r#where_device,
};
use mlx_rs::{Array, Dtype, Stream};

mod avio;
mod decode;

#[derive(Debug)]
pub struct Audio {
    pub samples: Array,
    pub sample_rate: u32,
}

pub fn load_audio(path: impl AsRef<Path>) -> Result<Audio> {
    let path = path.as_ref();
    decode::from_file(path)
        .with_context(|| format!("failed to decode input audio {}", path.display()))
}

pub fn load_audio_bytes(bytes: &[u8], file_extension: &str, mime_type: &str) -> Result<Audio> {
    decode::from_bytes(bytes, file_extension, mime_type).context("failed to decode uploaded audio")
}

pub fn write_wav_float(path: impl AsRef<Path>, audio: &Array, sample_rate: u32) -> Result<()> {
    let path = path.as_ref();
    let audio = match audio.ndim() {
        1 => audio.clone(),
        2 if audio.shape()[0] == 1 => audio.reshape(&[-1])?,
        3 if audio.shape()[0] == 1 && audio.shape()[2] == 1 => audio.reshape(&[-1])?,
        _ => anyhow::bail!("output audio must contain exactly one waveform"),
    };
    audio.eval()?;
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };
    let mut writer = WavWriter::create(path, spec)
        .with_context(|| format!("failed to create output WAV {}", path.display()))?;
    for &sample in audio.as_slice::<f32>() {
        writer.write_sample(sample)?;
    }
    writer.finalize()?;
    Ok(())
}

pub fn wav_float_bytes(audio: &Array, sample_rate: u32) -> Result<Vec<u8>> {
    let audio = match audio.ndim() {
        1 => audio.clone(),
        2 if audio.shape()[0] == 1 => audio.reshape(&[-1])?,
        3 if audio.shape()[0] == 1 && audio.shape()[2] == 1 => audio.reshape(&[-1])?,
        _ => anyhow::bail!("output audio must contain exactly one waveform"),
    };
    audio.eval()?;
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };
    let mut bytes = Vec::new();
    {
        let cursor = Cursor::new(&mut bytes);
        let mut writer = WavWriter::new(cursor, spec)?;
        for &sample in audio.as_slice::<f32>() {
            writer.write_sample(sample)?;
        }
        writer.finalize()?;
    }
    Ok(bytes)
}

fn centered_rms(audio: &Array, frame_length: i32, hop_length: i32) -> Result<Array> {
    let padding = frame_length / 2;
    let padded = pad(audio, &[(0, 0), (padding, padding)], None, None)?;
    let frame_count = (padded.shape()[1] - frame_length) / hop_length + 1;
    let frames = as_strided(
        &padded,
        &[audio.shape()[0], frame_count, frame_length][..],
        &[padded.shape()[1] as i64, hop_length as i64, 1_i64][..],
        0,
    )?;
    Ok(sqrt(&mean_axis(&square(&frames)?, -1, false)?)?)
}

fn interpolate_linear_1d_to_length(input: &Array, target_length: i32) -> Result<Array> {
    ensure!(input.ndim() == 2, "linear interpolation input must be 2D");
    ensure!(input.shape()[1] > 0, "linear interpolation input is empty");
    ensure!(target_length > 0, "linear interpolation target is empty");

    let input_length = input.shape()[1];
    if input_length == target_length {
        return Ok(input.clone());
    }
    if input_length == 1 {
        return Ok(broadcast_to(input, &[input.shape()[0], target_length])?);
    }

    let step = input_length as f32 / target_length as f32;
    let start = (1.0_f32 - step) / 2.0_f32;
    let positions = Array::arange::<_, f32>(Some(0), target_length, None)? * step - start;
    let positions = clip(&positions, (0, input_length - 1))?;
    let left_positions = floor(&positions)?;
    let right_positions = ceil(&positions)?;
    let weights = &positions - &left_positions;
    let left_indices = left_positions.as_type::<i32>()?;
    let right_indices = right_positions.as_type::<i32>()?;
    let left_values = input.take_axis(&left_indices, 1)?;
    let right_values = input.take_axis(&right_indices, 1)?;
    let interpolated = left_values * (Array::from_f32(1.0) - &weights) + right_values * weights;
    ensure!(
        interpolated.shape() == [input.shape()[0], target_length],
        "linear interpolation produced an incorrect shape"
    );
    Ok(interpolated)
}

pub fn adjust_loudness_envelope(
    source: &Array,
    source_sample_rate: i32,
    output: &Array,
    output_sample_rate: i32,
    output_rate: f32,
) -> Result<Array> {
    ensure!(
        source.ndim() == 2 && source.shape()[0] == 1,
        "loudness source must have shape [1, samples]"
    );
    let output = match output.ndim() {
        1 => output.reshape(&[1, -1])?,
        2 if output.shape()[0] == 1 => output.clone(),
        3 if output.shape()[0] == 1 && output.shape()[2] == 1 => output.reshape(&[1, -1])?,
        _ => anyhow::bail!("loudness output must contain one waveform"),
    };
    ensure!(
        source_sample_rate > 0 && output_sample_rate > 0,
        "sample rates must be positive"
    );
    let source_rms = centered_rms(source, source_sample_rate / 2 * 2, source_sample_rate / 2)?;
    let output_rms = centered_rms(&output, output_sample_rate / 2 * 2, output_sample_rate / 2)?;
    let target_length = output.shape()[1];
    let source_rms = interpolate_linear_1d_to_length(&source_rms, target_length)?;
    let output_rms = interpolate_linear_1d_to_length(&output_rms, target_length)?;
    let output_rms = maximum(&output_rms, &Array::from_f32(1.0e-6))?;
    let source_gain = power(&source_rms, &Array::from_f32(1.0 - output_rate))?;
    let output_gain = power(&output_rms, &Array::from_f32(output_rate - 1.0))?;
    Ok(output * source_gain * output_gain)
}

fn greatest_common_divisor(mut left: i32, mut right: i32) -> i32 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left.abs()
}

#[derive(Debug)]
pub struct SincResampler {
    original_rate: i32,
    target_rate: i32,
    reduced_original_rate: i32,
    reduced_target_rate: i32,
    width: i32,
    convolution: Option<Conv1d>,
}

impl SincResampler {
    pub fn new(original_rate: i32, target_rate: i32, lowpass_filter_width: i32) -> Result<Self> {
        ensure!(original_rate > 0, "original sample rate must be positive");
        ensure!(target_rate > 0, "target sample rate must be positive");
        ensure!(
            lowpass_filter_width > 0,
            "low-pass filter width must be positive"
        );
        if original_rate == target_rate {
            return Ok(Self {
                original_rate,
                target_rate,
                reduced_original_rate: 1,
                reduced_target_rate: 1,
                width: 0,
                convolution: None,
            });
        }

        let divisor = greatest_common_divisor(original_rate, target_rate);
        let reduced_original_rate = original_rate / divisor;
        let reduced_target_rate = target_rate / divisor;
        let base_frequency = reduced_original_rate.min(reduced_target_rate) as f64 * 0.99_f64;
        let width = (lowpass_filter_width as f64 * reduced_original_rate as f64 / base_frequency)
            .ceil() as i32;
        let cpu = Stream::cpu();
        let index = Array::arange_device::<_, f64>(
            Some(-width),
            width + reduced_original_rate,
            None,
            &cpu,
        )?
        .divide_device(Array::from_f64(reduced_original_rate as f64), &cpu)?
        .reshape_device(&[1, -1, 1], &cpu)?;
        let offset = Array::arange_device::<_, f64>(Some(0), -reduced_target_rate, Some(-1), &cpu)?
            .divide_device(Array::from_f64(reduced_target_rate as f64), &cpu)?
            .reshape_device(&[-1, 1, 1], &cpu)?;
        let time = offset
            .add_device(index, &cpu)?
            .multiply_device(Array::from_f64(base_frequency), &cpu)?;
        let time = clip_device(
            &time,
            (
                Array::from_f64(-lowpass_filter_width as f64),
                Array::from_f64(lowpass_filter_width as f64),
            ),
            &cpu,
        )?;
        let window = time
            .multiply_device(
                Array::from_f64(std::f64::consts::PI / lowpass_filter_width as f64 / 2.0),
                &cpu,
            )?
            .cos_device(&cpu)?
            .square_device(&cpu)?;
        let time_pi = time.multiply_device(Array::from_f64(std::f64::consts::PI), &cpu)?;
        let sinc = r#where_device(
            time_pi.eq_device(Array::from_f64(0.0), &cpu)?,
            Array::from_f64(1.0),
            time_pi.sin_device(&cpu)?.divide_device(&time_pi, &cpu)?,
            &cpu,
        )?;
        let kernel = sinc
            .multiply_device(window, &cpu)?
            .multiply_device(
                Array::from_f64(base_frequency / reduced_original_rate as f64),
                &cpu,
            )?
            .as_dtype_device(Dtype::Float32, &cpu)?;
        let kernel_size = kernel.shape()[1];
        let mut convolution = Conv1d::new(1, reduced_target_rate, kernel_size)?;
        convolution.weight.value = kernel;
        convolution.bias.value = None;
        convolution.stride = reduced_original_rate;
        Ok(Self {
            original_rate,
            target_rate,
            reduced_original_rate,
            reduced_target_rate,
            width,
            convolution: Some(convolution),
        })
    }

    pub fn resample(&mut self, waveform: &Array) -> Result<Array> {
        ensure!(
            waveform.ndim() == 2,
            "waveform must have shape [batch, samples]"
        );
        ensure!(
            waveform.dtype() == Dtype::Float32,
            "waveform must use float32"
        );
        let Some(convolution) = self.convolution.as_mut() else {
            return Ok(waveform.clone());
        };
        let source_length = waveform.shape()[1];
        let padded = pad(
            &waveform.expand_dims(-1)?,
            &[
                (0, 0),
                (self.width, self.width + self.reduced_original_rate),
                (0, 0),
            ],
            None,
            None,
        )?;
        let output = convolution
            .forward(&padded)?
            .reshape(&[waveform.shape()[0], -1])?;
        let target_length = ((self.reduced_target_rate as f64 * source_length as f64)
            / self.reduced_original_rate as f64)
            .ceil() as i32;
        ensure!(
            output.shape()[1] >= target_length,
            "resampler produced too few samples"
        );
        Ok(output.index((.., ..target_length)))
    }

    pub fn original_rate(&self) -> i32 {
        self.original_rate
    }

    pub fn target_rate(&self) -> i32 {
        self.target_rate
    }
}

#[derive(Debug)]
pub struct VolumeExtractor {
    hop_size: i32,
}

impl VolumeExtractor {
    pub fn new(hop_size: i32) -> Result<Self> {
        ensure!(hop_size > 0, "volume hop size must be positive");
        Ok(Self { hop_size })
    }

    pub fn extract(&self, audio: &Array) -> Result<Array> {
        ensure!(audio.ndim() == 2, "audio must have shape [batch, samples]");
        let frame_count = audio.shape()[1] / self.hop_size;
        ensure!(frame_count > 0, "audio is shorter than one volume frame");
        let squared = square(audio)?;
        let left = self.hop_size / 2;
        let right = (self.hop_size + 1) / 2;
        ensure!(
            left < audio.shape()[1] && right < audio.shape()[1],
            "audio is too short for reflect padding"
        );
        let reflected_left = squared
            .index((.., 1..left + 1))
            .index((.., (..).stride_by(-1)));
        let reflected_right = squared
            .index((.., audio.shape()[1] - right - 1..audio.shape()[1] - 1))
            .index((.., (..).stride_by(-1)));
        let padded = concatenate_axis(&[&reflected_left, &squared, &reflected_right], 1)?;
        let windows = as_strided(
            &padded,
            &[audio.shape()[0], frame_count, self.hop_size][..],
            &[padded.shape()[1] as i64, self.hop_size as i64, 1_i64][..],
            0,
        )?;
        Ok(sqrt(&mean_axis(&windows, -1, false)?)?)
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use super::interpolate_linear_1d_to_length;
    use mlx_rs::Array;

    #[test]
    fn linear_interpolation_uses_exact_requested_length() -> Result<()> {
        let input = Array::from_slice(&[0.0_f32, 10.0, 20.0], &[1, 3]);
        let output = interpolate_linear_1d_to_length(&input, 5)?;
        output.eval()?;

        assert_eq!(output.shape(), &[1, 5]);
        assert_eq!(
            output.as_slice::<f32>(),
            &[0.0, 4.000_000_5, 10.0, 16.000_002, 20.0]
        );
        Ok(())
    }

    #[test]
    fn linear_interpolation_matches_pytorch_half_pixel_coordinates() -> Result<()> {
        let input = Array::from_slice(&[0.0_f32, 10.0, 20.0, 30.0], &[1, 4]);
        let output = interpolate_linear_1d_to_length(&input, 2)?;
        output.eval()?;
        assert_eq!(output.as_slice::<f32>(), &[5.0, 25.0]);

        let single = Array::from_slice(&[3.25_f32], &[1, 1]);
        let broadcast = interpolate_linear_1d_to_length(&single, 4)?;
        let broadcast_sum = mlx_rs::ops::sum(&broadcast, false)?;
        assert_eq!(broadcast_sum.item::<f32>(), 13.0);
        Ok(())
    }
}
