use anyhow::{Result, ensure};
use mlx_rs::Array;
use mlx_rs::ops::{as_strided, mean_axis, pad, sqrt, square};

#[derive(Debug, Clone, Copy)]
pub struct AudioSlice {
    pub silent: bool,
    pub start: i32,
    pub end: i32,
}

#[derive(Debug, Clone)]
pub struct Slicer {
    threshold: f32,
    hop_size: i32,
    window_size: i32,
    minimum_length: i32,
    minimum_interval: i32,
    maximum_silence_kept: i32,
}

impl Slicer {
    pub fn new(
        sample_rate: i32,
        threshold_db: f32,
        minimum_length_ms: i32,
        minimum_interval_ms: i32,
        hop_size_ms: i32,
        maximum_silence_kept_ms: i32,
    ) -> Result<Self> {
        ensure!(
            minimum_length_ms >= minimum_interval_ms && minimum_interval_ms >= hop_size_ms,
            "minimum_length_ms >= minimum_interval_ms >= hop_size_ms is required"
        );
        ensure!(
            maximum_silence_kept_ms >= hop_size_ms,
            "maximum_silence_kept_ms must be at least hop_size_ms"
        );
        ensure!(sample_rate > 0, "sample rate must be positive");
        let minimum_interval_samples = sample_rate as f64 * minimum_interval_ms as f64 / 1000.0;
        let hop_size = (sample_rate as f64 * hop_size_ms as f64 / 1000.0).round() as i32;
        let window_size = (minimum_interval_samples.round() as i32).min(4 * hop_size);
        Ok(Self {
            threshold: 10.0_f32.powf(threshold_db / 20.0),
            hop_size,
            window_size,
            minimum_length: (sample_rate as f64 * minimum_length_ms as f64
                / 1000.0
                / hop_size as f64)
                .round() as i32,
            minimum_interval: (minimum_interval_samples / hop_size as f64).round() as i32,
            maximum_silence_kept: (sample_rate as f64 * maximum_silence_kept_ms as f64
                / 1000.0
                / hop_size as f64)
                .round() as i32,
        })
    }

    pub fn standard(sample_rate: i32, threshold_db: f32) -> Result<Self> {
        Self::new(sample_rate, threshold_db, 5_000, 300, 20, 5_000)
    }

    fn minimum_position(values: &[f32], start: usize, end: usize) -> usize {
        let mut position = start;
        let mut minimum = values[start];
        for (offset, &value) in values[start + 1..end].iter().enumerate() {
            if value < minimum {
                minimum = value;
                position = start + offset + 1;
            }
        }
        position
    }

    pub fn slice(&self, waveform: &Array) -> Result<Vec<AudioSlice>> {
        ensure!(
            waveform.ndim() == 2 && waveform.shape()[0] == 1,
            "slicer input must have shape [1, samples]"
        );
        let sample_count = waveform.shape()[1];
        if sample_count <= self.minimum_length {
            return Ok(vec![AudioSlice {
                silent: false,
                start: 0,
                end: sample_count,
            }]);
        }
        let half_window = self.window_size / 2;
        let padded = pad(waveform, &[(0, 0), (half_window, half_window)], None, None)?;
        let frame_count = (padded.shape()[1] - self.window_size) / self.hop_size + 1;
        let frames = as_strided(
            &padded,
            &[1, frame_count, self.window_size][..],
            &[padded.shape()[1] as i64, self.hop_size as i64, 1_i64][..],
            0,
        )?;
        let rms = sqrt(&mean_axis(&square(&frames)?, -1, false)?)?.reshape(&[-1])?;
        rms.eval()?;
        let rms = rms.as_slice::<f32>();
        let mut silence_tags = Vec::new();
        let mut silence_start = None;
        let mut clip_start = 0_i32;
        for (index, &value) in rms.iter().enumerate() {
            let index = index as i32;
            if value < self.threshold {
                if silence_start.is_none() {
                    silence_start = Some(index);
                }
                continue;
            }
            let Some(start) = silence_start else {
                continue;
            };
            let leading = start == 0 && index > self.maximum_silence_kept;
            let middle =
                index - start >= self.minimum_interval && index - clip_start >= self.minimum_length;
            if !leading && !middle {
                silence_start = None;
                continue;
            }
            if index - start <= self.maximum_silence_kept {
                let position =
                    Self::minimum_position(rms, start as usize, index as usize + 1) as i32;
                if start == 0 {
                    silence_tags.push((0, position));
                } else {
                    silence_tags.push((position, position));
                }
                clip_start = position;
            } else if index - start <= self.maximum_silence_kept * 2 {
                let position = Self::minimum_position(
                    rms,
                    (index - self.maximum_silence_kept) as usize,
                    (start + self.maximum_silence_kept + 1) as usize,
                ) as i32;
                let left = Self::minimum_position(
                    rms,
                    start as usize,
                    (start + self.maximum_silence_kept + 1) as usize,
                ) as i32;
                let right = Self::minimum_position(
                    rms,
                    (index - self.maximum_silence_kept) as usize,
                    index as usize + 1,
                ) as i32;
                if start == 0 {
                    silence_tags.push((0, right));
                    clip_start = right;
                } else {
                    silence_tags.push((left.min(position), right.max(position)));
                    clip_start = right.max(position);
                }
            } else {
                let left = Self::minimum_position(
                    rms,
                    start as usize,
                    (start + self.maximum_silence_kept + 1) as usize,
                ) as i32;
                let right = Self::minimum_position(
                    rms,
                    (index - self.maximum_silence_kept) as usize,
                    index as usize + 1,
                ) as i32;
                if start == 0 {
                    silence_tags.push((0, right));
                } else {
                    silence_tags.push((left, right));
                }
                clip_start = right;
            }
            silence_start = None;
        }
        if let Some(start) = silence_start {
            if frame_count - start >= self.minimum_interval {
                let silence_end = frame_count.min(start + self.maximum_silence_kept);
                let position = Self::minimum_position(
                    rms,
                    start as usize,
                    (silence_end as usize + 1).min(rms.len()),
                ) as i32;
                silence_tags.push((position, frame_count + 1));
            }
        }
        if silence_tags.is_empty() {
            return Ok(vec![AudioSlice {
                silent: false,
                start: 0,
                end: sample_count,
            }]);
        }

        let mut slices = Vec::with_capacity(silence_tags.len() * 2 + 1);
        if silence_tags[0].0 != 0 {
            slices.push(AudioSlice {
                silent: false,
                start: 0,
                end: sample_count.min(silence_tags[0].0 * self.hop_size),
            });
        }
        for index in 0..silence_tags.len() {
            if index > 0 {
                slices.push(AudioSlice {
                    silent: false,
                    start: silence_tags[index - 1].1 * self.hop_size,
                    end: sample_count.min(silence_tags[index].0 * self.hop_size),
                });
            }
            slices.push(AudioSlice {
                silent: true,
                start: silence_tags[index].0 * self.hop_size,
                end: sample_count.min(silence_tags[index].1 * self.hop_size),
            });
        }
        let trailing = silence_tags[silence_tags.len() - 1].1 * self.hop_size;
        if trailing < sample_count {
            slices.push(AudioSlice {
                silent: false,
                start: trailing,
                end: sample_count,
            });
        }
        slices.retain(|slice| slice.start != slice.end);
        Ok(slices)
    }
}
