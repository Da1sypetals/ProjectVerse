import argparse
import json
from pathlib import Path

import librosa
import torch
from safetensors.torch import save_file


def fuse_weight_norm(state):
    fused = {}
    consumed = set()
    for key, value in state.items():
        if not key.endswith(".weight_v"):
            continue
        key_g = f"{key[:-1]}g"
        assert key_g in state, f"missing weight_norm scale for {key}"
        weight_v = value.float()
        weight_g = state[key_g].float()
        weight_dimensions = [index for index, size in enumerate(weight_g.shape) if size != 1]
        assert len(weight_dimensions) <= 1, (key_g, weight_g.shape)
        dimension = weight_dimensions[0] if weight_dimensions else -1
        weight = torch._weight_norm(weight_v, weight_g, dimension)
        fused[key[:-2]] = weight.to(value.dtype)
        consumed.add(key)
        consumed.add(key_g)

    for key, value in state.items():
        if key not in consumed:
            fused[key] = value
    return fused


def convert_layout(state, transpose_convs):
    converted = {}
    for key, value in state.items():
        tensor = value.detach().cpu()
        if key.endswith(".weight") and tensor.ndim == 3:
            if any(marker in key for marker in transpose_convs):
                tensor = tensor.permute(1, 2, 0)
            else:
                tensor = tensor.permute(0, 2, 1)
        elif key.endswith(".weight") and tensor.ndim == 4:
            tensor = tensor.permute(0, 2, 3, 1)
        converted[key] = tensor.contiguous()
    return converted


def save_checkpoint(source, destination, state_key, transpose_convs, cast_float32=False):
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    assert state_key in checkpoint, f"{source} has no {state_key!r} state"
    state = checkpoint[state_key]
    assert isinstance(state, dict) and state, f"{source} contains an empty state"
    if cast_float32:
        state = {
            name: tensor.float() if tensor.is_floating_point() else tensor
            for name, tensor in state.items()
        }
    state = fuse_weight_norm(state)
    state = convert_layout(state, transpose_convs)
    save_file(state, destination)
    return {
        "source": str(source.resolve()),
        "destination": str(destination.resolve()),
        "tensor_count": len(state),
        "parameters": sum(tensor.numel() for tensor in state.values()),
    }


def save_vocoder_checkpoint(source, config_source, destination):
    config = json.loads(config_source.read_text(encoding="utf-8"))
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    assert "generator" in checkpoint, f"{source} has no 'generator' state"
    state = checkpoint["generator"]
    assert isinstance(state, dict) and state, f"{source} contains an empty generator state"
    state = fuse_weight_norm(state)
    state = convert_layout(state, ("ups.",))
    state["_buffers.mel_basis"] = torch.from_numpy(
        librosa.filters.mel(
            sr=config["sampling_rate"],
            n_fft=config["n_fft"],
            n_mels=config["num_mels"],
            fmin=config["fmin"],
            fmax=config["fmax"],
        )
    ).transpose(0, 1).contiguous()
    state["_buffers.hann_window"] = torch.hann_window(config["win_size"])
    save_file(state, destination)
    return {
        "source": str(source.resolve()),
        "config": str(config_source.resolve()),
        "destination": str(destination.resolve()),
        "tensor_count": len(state),
        "parameters": sum(tensor.numel() for tensor in state.values()),
    }


def flatten_reference(value, prefix, tensors, scalars):
    if isinstance(value, torch.Tensor):
        tensors[prefix] = value.detach().cpu().contiguous()
    elif isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            flatten_reference(child, name, tensors, scalars)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            name = f"{prefix}.{index}"
            flatten_reference(child, name, tensors, scalars)
    else:
        scalars[prefix] = value


def save_reference(source, tensor_destination, metadata_destination):
    reference = torch.load(source, map_location="cpu", weights_only=False)
    tensors = {}
    scalars = {}
    flatten_reference(reference, "", tensors, scalars)
    assert tensors, f"{source} contains no tensors"
    save_file(tensors, tensor_destination)
    metadata_destination.write_text(
        json.dumps(scalars, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return {
        "source": str(source.resolve()),
        "destination": str(tensor_destination.resolve()),
        "tensor_count": len(tensors),
        "values": sum(tensor.numel() for tensor in tensors.values()),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gan", type=Path, required=True)
    parser.add_argument("--diffusion", type=Path, required=True)
    parser.add_argument("--contentvec", type=Path, required=True)
    parser.add_argument("--vocoder", type=Path, required=True)
    parser.add_argument("--vocoder-config", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    assert args.gan.is_file(), args.gan
    assert args.diffusion.is_file(), args.diffusion
    assert args.contentvec.is_file(), args.contentvec
    assert args.vocoder.is_file(), args.vocoder
    assert args.vocoder_config.is_file(), args.vocoder_config
    assert args.reference.is_file(), args.reference
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "format": "sovits-svc-mlx-v1",
        "layout": {
            "activation": "channels_last",
            "conv1d_weight": "out_kernel_in",
            "conv_transpose1d_weight": "out_kernel_in",
            "linear_weight": "out_in",
        },
        "gan": save_checkpoint(
            args.gan,
            args.output_dir / "gan.safetensors",
            "model",
            ("dec.ups.",),
        ),
        "diffusion": save_checkpoint(
            args.diffusion,
            args.output_dir / "diffusion.safetensors",
            "model",
            (),
        ),
        "contentvec": save_checkpoint(
            args.contentvec,
            args.output_dir / "contentvec.safetensors",
            "model",
            (),
            cast_float32=True,
        ),
        "vocoder": save_vocoder_checkpoint(
            args.vocoder,
            args.vocoder_config,
            args.output_dir / "vocoder.safetensors",
        ),
        "reference": save_reference(
            args.reference,
            args.output_dir / "reference.safetensors",
            args.output_dir / "reference.json",
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
