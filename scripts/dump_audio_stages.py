import argparse
from pathlib import Path

import librosa
import torch
import torchaudio
from safetensors.torch import load_file, save_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def volume(audio, hop_size):
    frame_count = audio.size(-1) // hop_size
    squared = torch.nn.functional.pad(
        audio.square(),
        (hop_size // 2, (hop_size + 1) // 2),
        mode="reflect",
    )
    return torch.nn.functional.unfold(
        squared[:, None, None, :],
        (1, hop_size),
        stride=hop_size,
    )[:, :, :frame_count].mean(dim=1).sqrt()


def change_rms(source, source_rate, output, output_rate, rate):
    rms_source = librosa.feature.rms(
        y=source.numpy(),
        frame_length=source_rate // 2 * 2,
        hop_length=source_rate // 2,
    )
    rms_output = librosa.feature.rms(
        y=output.numpy(),
        frame_length=output_rate // 2 * 2,
        hop_length=output_rate // 2,
    )
    rms_source = torch.nn.functional.interpolate(
        torch.from_numpy(rms_source).unsqueeze(0),
        size=output.shape[0],
        mode="linear",
    ).squeeze()
    rms_output = torch.nn.functional.interpolate(
        torch.from_numpy(rms_output).unsqueeze(0),
        size=output.shape[0],
        mode="linear",
    ).squeeze()
    rms_output = torch.maximum(rms_output, torch.zeros_like(rms_output) + 1e-6)
    return output * rms_source.pow(1 - rate) * rms_output.pow(rate - 1)


def main():
    args = parse_args()
    assert args.reference.is_file(), args.reference
    reference = load_file(args.reference)
    audio = reference["input.wav_44k"].reshape(1, -1)
    content_resampler = torchaudio.transforms.Resample(44_100, 16_000)
    fcpe_resampler = torchaudio.transforms.Resample(
        44_100,
        16_000,
        lowpass_filter_width=128,
    )
    stages = {
        "input": audio.contiguous(),
        "resample.contentvec": content_resampler(audio).contiguous(),
        "resample.fcpe": fcpe_resampler(audio).contiguous(),
        "resample.functional": torchaudio.functional.resample(audio, 44_100, 16_000).contiguous(),
        "volume": volume(audio, 512).contiguous(),
        "loudness.input": reference["vocoder.audio"].reshape(-1).contiguous(),
        "loudness.rate_0_5": change_rms(
            audio.reshape(-1),
            44_100,
            reference["vocoder.audio"].reshape(-1),
            44_100,
            0.5,
        ).contiguous(),
    }
    save_file(stages, args.output)


if __name__ == "__main__":
    main()
