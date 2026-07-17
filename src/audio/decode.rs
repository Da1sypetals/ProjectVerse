use std::fmt;
use std::fs::File;
use std::io::Cursor;
use std::path::Path;

use anyhow::{Context, Result};
use babycat::{Signal, Source, Waveform, WaveformArgs};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{Decoder, DecoderOptions};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::{FormatOptions, FormatReader};
use symphonia::core::io::{MediaSource, MediaSourceStream};
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

struct PacketBuffer {
    samples: Vec<f32>,
    sample_rate: u32,
    channel_count: u16,
}

struct SymphoniaSource {
    decoder: Box<dyn Decoder>,
    reader: Box<dyn FormatReader>,
    track_id: u32,
    sample_rate: u32,
    channel_count: u16,
    remaining_samples: Option<usize>,
    packet: Option<Vec<f32>>,
    packet_index: usize,
}

impl SymphoniaSource {
    fn new(source: Box<dyn MediaSource>, file_extension: &str, mime_type: &str) -> Result<Self> {
        let mut hint = Hint::new();
        if !file_extension.is_empty() {
            hint.with_extension(file_extension);
        }
        if !mime_type.is_empty() {
            hint.mime_type(mime_type);
        }

        let source = MediaSourceStream::new(source, Default::default());
        let probed = symphonia::default::get_probe()
            .format(
                &hint,
                source,
                &FormatOptions {
                    enable_gapless: true,
                    ..Default::default()
                },
                &MetadataOptions::default(),
            )
            .context("failed to identify input audio encoding")?;
        let reader = probed.format;
        let track = reader
            .default_track()
            .context("input contains no decodable audio track")?;
        let track_id = track.id;
        let frame_count = track.codec_params.n_frames;
        let decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &DecoderOptions { verify: false })
            .context("failed to create input audio decoder")?;

        let mut source = Self {
            decoder,
            reader,
            track_id,
            sample_rate: 0,
            channel_count: 0,
            remaining_samples: None,
            packet: None,
            packet_index: 0,
        };
        let first_packet = source
            .next_packet_buffer()?
            .context("input contains no decoded audio samples")?;
        source.sample_rate = first_packet.sample_rate;
        source.channel_count = first_packet.channel_count;
        source.remaining_samples = match frame_count {
            Some(frames) => Some(
                usize::try_from(frames)
                    .context("input audio frame count exceeds platform limits")?
                    .checked_mul(usize::from(source.channel_count))
                    .context("input audio sample count exceeds platform limits")?,
            ),
            None => None,
        };
        source.packet = Some(first_packet.samples);
        Ok(source)
    }

    fn next_packet_buffer(&mut self) -> Result<Option<PacketBuffer>> {
        loop {
            let packet = match self.reader.next_packet() {
                Ok(packet) => packet,
                Err(SymphoniaError::IoError(error))
                    if error.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    return Ok(None);
                }
                Err(error) => return Err(error).context("failed to read input audio packet"),
            };
            if packet.track_id() != self.track_id {
                continue;
            }

            let decoded = match self.decoder.decode(&packet) {
                Ok(decoded) => decoded,
                Err(SymphoniaError::DecodeError(_)) => continue,
                Err(error) => return Err(error).context("failed to decode input audio packet"),
            };
            let spec = *decoded.spec();
            let channel_count = u16::try_from(spec.channels.count())
                .context("input has too many audio channels")?;
            let mut buffer = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
            buffer.copy_interleaved_ref(decoded);
            return Ok(Some(PacketBuffer {
                samples: buffer.samples().to_owned(),
                sample_rate: spec.rate,
                channel_count,
            }));
        }
    }
}

impl fmt::Debug for SymphoniaSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SymphoniaSource")
            .field("sample_rate", &self.sample_rate)
            .field("channel_count", &self.channel_count)
            .field("remaining_samples", &self.remaining_samples)
            .finish()
    }
}

impl Signal for SymphoniaSource {
    fn frame_rate_hz(&self) -> u32 {
        self.sample_rate
    }

    fn num_channels(&self) -> u16 {
        self.channel_count
    }

    fn num_frames_estimate(&self) -> Option<usize> {
        self.remaining_samples
            .map(|samples| samples / usize::from(self.channel_count))
    }
}

impl Source for SymphoniaSource {}

impl Iterator for SymphoniaSource {
    type Item = f32;

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self.remaining_samples {
            Some(samples) => (samples, None),
            None => (0, None),
        }
    }

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let packet = self.packet.as_ref()?;
            if self.packet_index < packet.len() {
                let sample = packet[self.packet_index];
                self.packet_index += 1;
                self.remaining_samples = self
                    .remaining_samples
                    .map(|remaining| remaining.saturating_sub(1));
                return Some(sample);
            }
            self.packet = self
                .next_packet_buffer()
                .ok()
                .flatten()
                .map(|packet| packet.samples);
            self.packet_index = 0;
        }
    }
}

pub(super) fn from_file(path: &Path) -> Result<Waveform> {
    let file = File::open(path)
        .with_context(|| format!("failed to open input audio {}", path.display()))?;
    let file_extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or("");
    from_media_source(Box::new(file), file_extension, "")
}

pub(super) fn from_bytes(bytes: &[u8], file_extension: &str, mime_type: &str) -> Result<Waveform> {
    from_media_source(
        Box::new(Cursor::new(bytes.to_owned())),
        file_extension,
        mime_type,
    )
}

fn from_media_source(
    source: Box<dyn MediaSource>,
    file_extension: &str,
    mime_type: &str,
) -> Result<Waveform> {
    let source = SymphoniaSource::new(source, file_extension, mime_type)?;
    Waveform::from_source(
        WaveformArgs {
            num_channels: 1,
            ..Default::default()
        },
        Box::new(source),
    )
    .context("failed to process decoded audio with Babycat")
}
