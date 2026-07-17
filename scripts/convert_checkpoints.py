import argparse
import json
from pathlib import Path

import librosa
import torch
from safetensors.torch import load_file, save_file


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
    destination.parent.mkdir(parents=True, exist_ok=True)
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
    destination.parent.mkdir(parents=True, exist_ok=True)
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
    tensor_destination.parent.mkdir(parents=True, exist_ok=True)
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


def save_flow_matching_checkpoint(source, destination):
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    assert "model" in checkpoint, f"{source} has no 'model' state"
    source_state = checkpoint["model"]
    assert isinstance(source_state, dict) and source_state, f"{source} contains an empty model state"
    state = {}
    for name, tensor in source_state.items():
        if name.startswith("unit2mel."):
            state[name.removeprefix("unit2mel.")] = tensor
        elif name in {"a_k", "b_k", "t_k"}:
            state[name] = tensor
        else:
            raise AssertionError(f"unexpected Flow Matching tensor: {name}")
    state["_flow.alphas_cumprod_ascending"] = source_state[
        "unit2mel.decoder.alphas_cumprod"
    ].flip(0)
    state["_flow.timesteps"] = torch.tensor(checkpoint["config"]["timesteps"], dtype=torch.int32)
    state["_flow.t_eps"] = torch.tensor(checkpoint["config"]["t_eps"], dtype=torch.float32)
    state["_flow.default_ode_steps"] = torch.tensor(
        checkpoint["config"]["ode_steps"],
        dtype=torch.int32,
    )
    state = fuse_weight_norm(state)
    state = convert_layout(state, ())
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, destination)
    return {
        "source": str(source.resolve()),
        "destination": str(destination.resolve()),
        "tensor_count": len(state),
        "parameters": sum(tensor.numel() for tensor in state.values()),
        "global_step": checkpoint["global_step"],
        "epoch": checkpoint["epoch"],
        "config": checkpoint["config"],
    }


def save_existing_safetensors(source, destination):
    state = load_file(source)
    assert state, f"{source} contains no tensors"
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, destination)
    return {
        "source": str(source.resolve()),
        "destination": str(destination.resolve()),
        "tensor_count": len(state),
        "parameters": sum(tensor.numel() for tensor in state.values()),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gan", nargs=2, action="append", metavar=("NAME", "PATH"), required=True)
    parser.add_argument("--shallow-diffusion", type=Path, required=True)
    parser.add_argument("--flow-matching", type=Path, required=True)
    parser.add_argument("--contentvec", type=Path, required=True)
    parser.add_argument("--fcpe", type=Path, required=True)
    parser.add_argument("--vocoder", type=Path, required=True)
    parser.add_argument("--vocoder-config", type=Path, required=True)
    parser.add_argument(
        "--reference",
        nargs=2,
        action="append",
        metavar=("NAME", "PATH"),
        default=[],
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    for name, path in args.gan:
        path = Path(path)
        assert name, "GAN name cannot be empty"
        assert path.is_file(), path
    assert args.shallow_diffusion.is_file(), args.shallow_diffusion
    assert args.flow_matching.is_file(), args.flow_matching
    assert args.contentvec.is_file(), args.contentvec
    assert args.fcpe.is_file(), args.fcpe
    assert args.vocoder.is_file(), args.vocoder
    assert args.vocoder_config.is_file(), args.vocoder_config
    for name, path in args.reference:
        path = Path(path)
        assert name, "reference name cannot be empty"
        assert path.is_file(), path
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name, path in args.gan:
        save_checkpoint(
            Path(path),
            args.output_dir / "gan" / name / "model.safetensors",
            "model",
            ("dec.ups.",),
        )
    save_checkpoint(
        args.shallow_diffusion,
        args.output_dir / "refine" / "shallow_diffusion" / "guan" / "model.safetensors",
        "model",
        (),
    )
    save_flow_matching_checkpoint(
        args.flow_matching,
        args.output_dir / "refine" / "flow_matching" / "opencpop" / "model.safetensors",
    )
    save_checkpoint(
        args.contentvec,
        args.output_dir / "encoder" / "contentvec" / "model.safetensors",
        "model",
        (),
        cast_float32=True,
    )
    save_existing_safetensors(
        args.fcpe,
        args.output_dir / "pitch" / "fcpe" / "model.safetensors",
    )
    save_vocoder_checkpoint(
        args.vocoder,
        args.vocoder_config,
        args.output_dir / "vocoder" / "nsf_hifigan" / "model.safetensors",
    )
    for name, path in args.reference:
        save_reference(
            Path(path),
            args.output_dir / "references" / name / "tensors.safetensors",
            args.output_dir / "references" / name / "metadata.json",
        )


if __name__ == "__main__":
    main()
