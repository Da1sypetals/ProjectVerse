use std::mem::size_of;
use std::path::Path;

use anyhow::{Context, Result, ensure};
use bytemuck::try_cast_slice;
use ffmpeg_the_third as ffmpeg;
use ffmpeg_the_third::ChannelLayout;
use ffmpeg_the_third::codec;
use ffmpeg_the_third::format;
use ffmpeg_the_third::frame;
use ffmpeg_the_third::media;
use ffmpeg_the_third::software::resampling;
use mlx_rs::Array;
use mlx_rs::ops::{concatenate_axis, mean_axis};

use super::Audio;
use super::avio::MemoryInput;

pub(super) fn from_file(path: &Path) -> Result<Audio> {
    ffmpeg::init().context("failed to initialize FFmpeg")?;
    let mut input = format::input(path)
        .with_context(|| format!("failed to open input audio {}", path.display()))?;
    decode_input(&mut input, &path.display().to_string())
}

pub(super) fn from_bytes(bytes: &[u8], file_extension: &str, mime_type: &str) -> Result<Audio> {
    ffmpeg::init().context("failed to initialize FFmpeg")?;
    let mut input = MemoryInput::open(bytes, file_extension, mime_type)?;
    decode_input(input.input_mut(), "uploaded audio")
}

fn decode_input(input: &mut format::context::Input, label: &str) -> Result<Audio> {
    let (stream_index, decoder_context) = {
        let audio_stream = input
            .streams()
            .best(media::Type::Audio)
            .with_context(|| format!("{label} contains no decodable audio stream"))?;
        let stream_index = audio_stream.index();
        let decoder_context =
            codec::context::Context::from_parameters(audio_stream.parameters())
                .with_context(|| format!("failed to create decoder context for {label}"))?;
        (stream_index, decoder_context)
    };
    let mut decoder = decoder_context
        .decoder()
        .audio()
        .with_context(|| format!("failed to open audio decoder for {label}"))?;

    let mut converter = None;
    let mut decoded_audio = DecodedAudio::default();
    for packet in input.packets() {
        let (stream, packet) =
            packet.with_context(|| format!("failed to read an audio packet from {label}"))?;
        if stream.index() != stream_index {
            continue;
        }
        decoder.send_packet(&packet).with_context(|| {
            format!("failed to send an audio packet to the decoder for {label}")
        })?;
        receive_frames(
            &mut decoder,
            &mut converter,
            &mut decoded_audio,
            label,
            false,
        )?;
    }

    decoder
        .send_eof()
        .with_context(|| format!("failed to flush the audio decoder for {label}"))?;
    receive_frames(
        &mut decoder,
        &mut converter,
        &mut decoded_audio,
        label,
        true,
    )?;
    if let Some(mut converter) = converter {
        converter.flush(&mut decoded_audio, label)?;
    }
    decoded_audio.finish(label)
}

fn receive_frames(
    decoder: &mut ffmpeg::decoder::Audio,
    converter: &mut Option<FrameConverter>,
    decoded_audio: &mut DecodedAudio,
    label: &str,
    draining: bool,
) -> Result<()> {
    loop {
        let mut decoded = frame::Audio::empty();
        match decoder.receive_frame(&mut decoded) {
            Ok(()) => convert_frame(decoded, converter, decoded_audio, label)?,
            Err(ffmpeg::Error::Other { errno }) if errno == libc::EAGAIN => {
                ensure!(
                    !draining,
                    "audio decoder for {label} requested more input after EOF"
                );
                return Ok(());
            }
            Err(ffmpeg::Error::Eof) => {
                ensure!(
                    draining,
                    "audio decoder for {label} reached EOF prematurely"
                );
                return Ok(());
            }
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("failed to receive a decoded frame from {label}"));
            }
        }
    }
}

fn convert_frame(
    mut decoded: frame::Audio,
    converter: &mut Option<FrameConverter>,
    decoded_audio: &mut DecodedAudio,
    label: &str,
) -> Result<()> {
    let channels = decoded.ch_layout().channels();
    ensure!(
        channels > 0,
        "decoded audio frame from {label} has no channels"
    );
    ensure!(
        decoded.rate() > 0,
        "decoded audio frame from {label} has no sample rate"
    );
    ensure!(
        decoded.samples() > 0,
        "decoded audio frame from {label} has no samples"
    );

    let mut channel_layout = ChannelLayout::from(decoded.ch_layout().clone().into_owned());
    ensure!(
        channel_layout.is_valid(),
        "decoded audio frame from {label} has an invalid channel layout"
    );
    if channel_layout.mask().is_none() {
        channel_layout = ChannelLayout::default_for_channels(channels);
        decoded.set_ch_layout(channel_layout.clone());
    }

    let needs_new_converter = converter.as_ref().is_none_or(|converter| {
        !converter.accepts(decoded.format(), decoded.rate(), &channel_layout)
    });
    if needs_new_converter {
        if let Some(mut previous) = converter.take() {
            previous.flush(decoded_audio, label)?;
        }
        *converter = Some(FrameConverter::new(
            decoded.format(),
            decoded.rate(),
            channel_layout,
        )?);
    }

    converter
        .as_mut()
        .expect("frame converter was initialized")
        .convert(&decoded, decoded_audio, label)
}

struct FrameConverter {
    input_format: format::Sample,
    input_rate: u32,
    channel_layout: ChannelLayout<'static>,
    context: resampling::Context,
}

impl FrameConverter {
    fn new(
        input_format: format::Sample,
        input_rate: u32,
        channel_layout: ChannelLayout<'static>,
    ) -> Result<Self> {
        let context = resampling::Context::get2(
            input_format,
            channel_layout.clone(),
            input_rate,
            format::Sample::F32(format::sample::Type::Packed),
            channel_layout.clone(),
            input_rate,
        )
        .context("failed to create FFmpeg audio sample-format converter")?;
        Ok(Self {
            input_format,
            input_rate,
            channel_layout,
            context,
        })
    }

    fn accepts(
        &self,
        input_format: format::Sample,
        input_rate: u32,
        channel_layout: &ChannelLayout<'_>,
    ) -> bool {
        self.input_format == input_format
            && self.input_rate == input_rate
            && self.channel_layout == *channel_layout
    }

    fn convert(
        &mut self,
        decoded: &frame::Audio,
        decoded_audio: &mut DecodedAudio,
        label: &str,
    ) -> Result<()> {
        let mut converted = frame::Audio::empty();
        self.context
            .run(decoded, &mut converted)
            .with_context(|| format!("failed to convert a decoded audio frame from {label}"))?;
        decoded_audio.append_frame(&converted, label)
    }

    fn flush(&mut self, decoded_audio: &mut DecodedAudio, label: &str) -> Result<()> {
        while let Some(delay) = self.context.delay() {
            let output_samples = usize::try_from(delay.output)
                .context("FFmpeg sample-format converter delay exceeds platform limits")?;
            ensure!(
                output_samples > 0,
                "FFmpeg sample-format converter reported an invalid delay"
            );
            let output = *self.context.output();
            let mut converted =
                frame::Audio::new(output.format, output_samples, output.channel_layout);
            let remaining = self
                .context
                .flush(&mut converted)
                .with_context(|| format!("failed to flush converted audio samples from {label}"))?;
            decoded_audio.append_frame(&converted, label)?;
            if remaining.is_none() {
                break;
            }
        }
        Ok(())
    }
}

#[derive(Default)]
struct DecodedAudio {
    frames: Vec<Array>,
    sample_rate: Option<u32>,
}

impl DecodedAudio {
    fn append_frame(&mut self, converted: &frame::Audio, label: &str) -> Result<()> {
        if converted.samples() == 0 {
            return Ok(());
        }
        ensure!(
            converted.format() == format::Sample::F32(format::sample::Type::Packed),
            "FFmpeg returned an unexpected sample format for {label}"
        );
        let channel_count = converted.ch_layout().channels();
        ensure!(
            channel_count > 0,
            "converted audio frame from {label} has no channels"
        );
        let sample_rate = converted.rate();
        ensure!(
            matches!(sample_rate, 44_100 | 48_000),
            "{label} has unsupported sample rate {sample_rate} Hz; expected 44100 or 48000 Hz"
        );
        if let Some(expected_rate) = self.sample_rate {
            ensure!(
                sample_rate == expected_rate,
                "{label} changes sample rate from {expected_rate} Hz to {sample_rate} Hz"
            );
        } else {
            self.sample_rate = Some(sample_rate);
        }

        let frame_samples = i32::try_from(converted.samples())
            .context("decoded audio frame contains too many samples")?;
        let channels = i32::try_from(channel_count)
            .context("decoded audio frame contains too many channels")?;
        let element_count = converted
            .samples()
            .checked_mul(channel_count as usize)
            .context("decoded audio frame sample count overflowed")?;
        let byte_count = element_count
            .checked_mul(size_of::<f32>())
            .context("decoded audio frame byte count overflowed")?;
        let bytes = converted.data(0);
        ensure!(
            bytes.len() >= byte_count,
            "FFmpeg returned a truncated packed audio frame for {label}"
        );
        let samples = try_cast_slice::<u8, f32>(&bytes[..byte_count]).map_err(|error| {
            anyhow::anyhow!("FFmpeg returned an invalid packed audio frame: {error}")
        })?;
        let interleaved = Array::from_slice(samples, &[frame_samples, channels]);
        let mono = mean_axis(&interleaved, 1, false)?.reshape(&[1, frame_samples])?;
        mono.eval()?;
        self.frames.push(mono);
        Ok(())
    }

    fn finish(mut self, label: &str) -> Result<Audio> {
        let sample_rate = self
            .sample_rate
            .with_context(|| format!("{label} contains no decoded audio samples"))?;
        let samples = if self.frames.len() == 1 {
            self.frames.pop().expect("one decoded frame is available")
        } else {
            concatenate_axis(&self.frames, 1)?
        };
        ensure!(
            samples.ndim() == 2 && samples.shape()[0] == 1 && samples.shape()[1] > 0,
            "decoded audio for {label} has an invalid shape"
        );
        samples.eval()?;
        Ok(Audio {
            samples,
            sample_rate,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::Cursor;

    use anyhow::Result;
    use hound::{SampleFormat, WavSpec, WavWriter};

    use super::{from_bytes, from_file};

    const INTERLEAVED_STEREO: [f32; 8] = [0.25, 0.75, -0.5, 0.5, 1.0, -1.0, 0.75, 0.25];
    const EXPECTED_MONO: [f32; 4] = [0.5, 0.0, 0.0, 0.5];

    fn stereo_wav(sample_rate: u32) -> Result<Vec<u8>> {
        let mut cursor = Cursor::new(Vec::new());
        {
            let mut writer = WavWriter::new(
                &mut cursor,
                WavSpec {
                    channels: 2,
                    sample_rate,
                    bits_per_sample: 32,
                    sample_format: SampleFormat::Float,
                },
            )?;
            for sample in INTERLEAVED_STEREO {
                writer.write_sample(sample)?;
            }
            writer.finalize()?;
        }
        Ok(cursor.into_inner())
    }

    #[test]
    fn file_and_memory_inputs_average_every_channel() -> Result<()> {
        let bytes_48k = stereo_wav(48_000)?;
        let memory_audio_48k = from_bytes(&bytes_48k, "wav", "audio/wav")?;
        let bytes_44k = stereo_wav(44_100)?;
        let memory_audio_44k = from_bytes(&bytes_44k, "", "")?;

        let directory = std::path::Path::new("target/audio-loading-tests");
        fs::create_dir_all(directory)?;
        let path = directory.join(format!("stereo-{}.wav", std::process::id()));
        fs::write(&path, &bytes_48k)?;
        let file_audio = from_file(&path)?;
        fs::remove_file(path)?;

        assert_eq!(memory_audio_48k.sample_rate, 48_000);
        assert_eq!(memory_audio_44k.sample_rate, 44_100);
        assert_eq!(file_audio.sample_rate, 48_000);
        assert_eq!(memory_audio_48k.samples.shape(), &[1, 4]);
        assert_eq!(memory_audio_44k.samples.shape(), &[1, 4]);
        assert_eq!(file_audio.samples.shape(), &[1, 4]);
        assert_eq!(memory_audio_48k.samples.as_slice::<f32>(), EXPECTED_MONO);
        assert_eq!(memory_audio_44k.samples.as_slice::<f32>(), EXPECTED_MONO);
        assert_eq!(file_audio.samples.as_slice::<f32>(), EXPECTED_MONO);
        assert_eq!(
            memory_audio_48k.samples.as_slice::<f32>(),
            file_audio.samples.as_slice::<f32>()
        );
        Ok(())
    }
}
