import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


class RandomCapture:
    def __enter__(self):
        self.uniform = []
        self.normal = []
        self.original_rand = torch.rand
        self.original_randn_like = torch.randn_like

        def rand(*args, **kwargs):
            value = self.original_rand(*args, **kwargs)
            self.uniform.append(value.detach().clone())
            return value

        def randn_like(*args, **kwargs):
            value = self.original_randn_like(*args, **kwargs)
            self.normal.append(value.detach().clone())
            return value

        torch.rand = rand
        torch.randn_like = randn_like
        return self

    def __exit__(self, *_exc):
        torch.rand = self.original_rand
        torch.randn_like = self.original_randn_like


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def channels_last(tensor):
    return tensor.transpose(1, 2).contiguous()


def main():
    args = parse_args()
    assert args.source_dir.is_dir(), args.source_dir
    assert args.checkpoint.is_file(), args.checkpoint
    assert args.reference.is_file(), args.reference
    sys.path.insert(0, str(args.source_dir.resolve()))

    from vdecoder.nsf_hifigan.models import load_model
    from vdecoder.nsf_hifigan.nvSTFT import STFT

    reference = load_file(args.reference)
    model, config = load_model(str(args.checkpoint.resolve()), device="cpu")
    stft = STFT(
        config.sampling_rate,
        config.num_mels,
        config.n_fft,
        config.win_size,
        config.hop_size,
        config.fmin,
        config.fmax,
    )
    gan_audio = reference["gan.audio"].unsqueeze(0)
    mel = reference["diffusion.mel_refined"]
    f0 = reference["diffusion.f0"][:, : mel.size(1), 0]
    stages = {}
    handles = []

    def capture_conv(name):
        def hook(_module, _inputs, output):
            stages[name] = channels_last(output.detach())

        return hook

    def capture_source(_module, _inputs, output):
        sine, uv, noise = output
        stages["source.sine"] = sine.detach().contiguous()
        stages["source.uv"] = uv.detach().contiguous()
        stages["source.noise"] = noise.detach().contiguous()

    def capture_merge(_module, _inputs, output):
        stages["source.merged"] = output.detach().contiguous()

    handles.append(model.m_source.l_sin_gen.register_forward_hook(capture_source))
    handles.append(model.m_source.register_forward_hook(capture_merge))
    handles.append(model.conv_pre.register_forward_hook(capture_conv("generator.conv_pre")))
    handles.append(model.conv_post.register_forward_hook(capture_conv("generator.conv_post")))
    for index, layer in enumerate(model.ups):
        handles.append(layer.register_forward_hook(capture_conv(f"generator.up.{index}")))
    for index, layer in enumerate(model.noise_convs):
        handles.append(layer.register_forward_hook(capture_conv(f"generator.source_conv.{index}")))
    for index, layer in enumerate(model.resblocks):
        handles.append(layer.register_forward_hook(capture_conv(f"generator.resblock.{index}")))

    with torch.no_grad():
        audio_mel_from_gan = stft.get_mel(gan_audio).transpose(1, 2)
        with RandomCapture() as random_capture:
            audio = model(mel.transpose(1, 2), f0)

    assert len(random_capture.uniform) == 1, len(random_capture.uniform)
    assert len(random_capture.normal) == 1, len(random_capture.normal)
    initial_phase = random_capture.uniform[0]
    initial_phase[:, 0] = 0
    stages["input.mel"] = mel.contiguous()
    stages["input.f0"] = f0.contiguous()
    stages["mel.audio"] = gan_audio.contiguous()
    stages["mel.output"] = audio_mel_from_gan.contiguous()
    stages["mel.cuda"] = reference["diffusion.audio_mel_from_gan"].contiguous()
    stages["source.initial_phase"] = initial_phase.contiguous()
    stages["source.normal"] = random_capture.normal[0].contiguous()
    stages["generator.output"] = channels_last(audio)
    stages["generator.cuda"] = reference["vocoder.audio"].reshape(1, -1, 1).contiguous()
    for handle in handles:
        handle.remove()
    save_file(stages, args.output)


if __name__ == "__main__":
    main()
