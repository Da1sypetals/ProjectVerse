import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file


class RandomCapture:
    def __init__(self):
        self.uniform = []
        self.normal = []

    def __enter__(self):
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
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def channels_last(tensor):
    if tensor.ndim == 3:
        return tensor.transpose(1, 2).contiguous()
    return tensor.contiguous()


def main():
    args = parse_args()
    assert args.source_dir.is_dir(), args.source_dir
    assert args.checkpoint.is_file(), args.checkpoint
    assert args.config.is_file(), args.config
    assert args.reference.is_file(), args.reference
    sys.path.insert(0, str(args.source_dir.resolve()))

    import modules.commons as commons
    import utils
    from models import SynthesizerTrn

    hps = utils.get_hparams_from_file(str(args.config.resolve()), True)
    model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
    ).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    reference = torch.load(args.reference, map_location="cpu", weights_only=False)
    content = reference["features"]["c"]
    f0 = reference["features"]["f0"]
    uv = reference["features"]["uv"]
    volume = reference["features"]["vol"]
    speaker_id = reference["features"]["sid"]
    encoder_noise = reference["gan"]["enc_p_eps"]
    f0_coarse = reference["gan"]["f0_coarse"]
    stages = {}
    handles = []

    def capture(name):
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            stages[name] = channels_last(value.detach())

        return hook

    for index, layer in enumerate(model.enc_p.enc_.attn_layers):
        handles.append(layer.register_forward_hook(capture(f"encoder.attention.{index}")))
    for index, layer in enumerate(model.enc_p.enc_.ffn_layers):
        handles.append(layer.register_forward_hook(capture(f"encoder.ffn.{index}")))
    for index, flow in enumerate(model.flow.flows):
        handles.append(flow.register_forward_hook(capture(f"flow.{index}")))
    for index, layer in enumerate(model.f0_decoder.decoder.self_attn_layers):
        handles.append(layer.register_forward_hook(capture(f"f0_decoder.attention.{index}")))
    for index, layer in enumerate(model.f0_decoder.decoder.ffn_layers):
        handles.append(layer.register_forward_hook(capture(f"f0_decoder.ffn.{index}")))

    with torch.no_grad():
        speaker = model.emb_g(speaker_id).transpose(1, 2)
        lengths = (torch.ones(content.size(0)) * content.size(-1)).long()
        mask = torch.unsqueeze(commons.sequence_mask(lengths, content.size(2)), 1).to(content.dtype)
        volume_embedding = model.emb_vol(volume[:, :, None]).transpose(1, 2)
        preprocessed = (
            model.pre(content) * mask
            + model.emb_uv(uv.long()).transpose(1, 2)
            + volume_embedding
        )
        log_f0 = 2595.0 * torch.log10(1.0 + f0.unsqueeze(1) / 700.0) / 500.0
        normalized_log_f0 = utils.normalize_f0(log_f0, mask, uv, random_scale=False)
        predicted_log_f0 = model.f0_decoder(
            preprocessed,
            normalized_log_f0,
            mask,
            spk_emb=speaker,
        )
        predicted_f0 = (
            700.0 * (torch.pow(10, predicted_log_f0 * 500.0 / 2595.0) - 1.0)
        ).squeeze(1)
        predicted_f0_coarse = utils.f0_to_coarse(predicted_f0)
        hidden = preprocessed + model.enc_p.f0_emb(f0_coarse).transpose(1, 2)
        hidden = model.enc_p.enc_(hidden * mask, mask)
        stats = model.enc_p.proj(hidden) * mask
        mean, log_scale = torch.split(stats, model.enc_p.out_channels, dim=1)
        sampled = (mean + encoder_noise * torch.exp(log_scale) * 0.4) * mask
        flowed = model.flow(sampled, mask, g=speaker, reverse=True)
        with RandomCapture() as random_capture:
            decoded = model.dec(flowed * mask, g=speaker, f0=f0)

    assert len(random_capture.uniform) == 1, len(random_capture.uniform)
    assert len(random_capture.normal) == 2, len(random_capture.normal)
    initial_phase = random_capture.uniform[0]
    initial_phase[:, 0] = 0

    stages["speaker"] = channels_last(speaker)
    stages["mask"] = channels_last(mask)
    stages["volume_embedding"] = channels_last(volume_embedding)
    stages["preprocessed"] = channels_last(preprocessed)
    stages["f0_decoder.log_f0"] = channels_last(log_f0)
    stages["f0_decoder.normalized_log_f0"] = channels_last(normalized_log_f0)
    stages["f0_decoder.predicted_log_f0"] = channels_last(predicted_log_f0)
    stages["f0_decoder.f0"] = predicted_f0.contiguous()
    stages["f0_decoder.f0_coarse"] = predicted_f0_coarse.contiguous()
    stages["mean"] = channels_last(mean)
    stages["log_scale"] = channels_last(log_scale)
    stages["sampled"] = channels_last(sampled)
    stages["flowed"] = channels_last(flowed)
    stages["decoder.initial_phase"] = initial_phase.contiguous()
    stages["decoder.sine_noise"] = random_capture.normal[0].contiguous()
    stages["decoder.source_noise"] = random_capture.normal[1].contiguous()
    stages["decoder.output"] = channels_last(decoded)
    stages["cuda.preprocessed"] = channels_last(reference["gan"]["x_pre"])
    stages["cuda.mean"] = channels_last(reference["gan"]["m_p"])
    stages["cuda.log_scale"] = channels_last(reference["gan"]["logs_p"])
    stages["cuda.sampled"] = channels_last(reference["gan"]["z_p"])
    stages["cuda.flowed"] = channels_last(reference["gan"]["z"])
    for handle in handles:
        handle.remove()
    save_file(stages, args.output)


if __name__ == "__main__":
    main()
