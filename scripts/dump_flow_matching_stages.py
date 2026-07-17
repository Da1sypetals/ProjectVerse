import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def channels_last(tensor):
    if tensor.ndim == 3:
        return tensor.permute(0, 2, 1).contiguous()
    return tensor.contiguous()


def main():
    args = parse_args()
    assert args.source_root.is_dir(), args.source_root
    assert args.checkpoint.is_file(), args.checkpoint
    assert args.reference.is_file(), args.reference
    sys.path.insert(0, str(args.source_root.resolve()))
    sys.path.insert(0, str((args.source_root / "so-vits-svc").resolve()))

    from shallow_fm.model import ShallowFM

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = ShallowFM(
        input_channel=config["encoder_out_channels"],
        n_spk=config["n_spk"],
        use_pitch_aug=config["use_pitch_aug"],
        out_dims=128,
        n_layers=config["n_layers"],
        n_chans=config["n_chans"],
        n_hidden=config["n_hidden"],
        timesteps=config["timesteps"],
        k_step_max=config["k_step_max"],
        shallow_k=config["shallow_k"],
        p_start=config["p_start"],
        t_eps=config["t_eps"],
    ).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    reference = torch.load(args.reference, map_location="cpu", weights_only=False)
    spec = reference["fm"]["ode_x_vp"][0]
    diffusion_step = reference["fm"]["ode_k"][0]
    condition = reference["fm"]["cond"]
    wavenet = model.unit2mel.decoder.denoise_fn
    stages = {}
    handles = []

    def capture(name):
        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                stages[f"{name}.residual"] = channels_last(output[0].detach())
                stages[f"{name}.skip"] = channels_last(output[1].detach())
            else:
                stages[name] = channels_last(output.detach())

        return hook

    handles.append(wavenet.input_projection.register_forward_hook(capture("input_projection")))
    handles.append(wavenet.diffusion_embedding.register_forward_hook(capture("diffusion_embedding")))
    handles.append(wavenet.mlp[0].register_forward_hook(capture("mlp.0")))
    handles.append(wavenet.mlp[1].register_forward_hook(capture("mlp.1")))
    handles.append(wavenet.mlp[2].register_forward_hook(capture("mlp.2")))
    for index, layer in enumerate(wavenet.residual_layers):
        handles.append(layer.register_forward_hook(capture(f"residual_layers.{index}")))
    handles.append(wavenet.skip_projection.register_forward_hook(capture("skip_projection")))
    handles.append(wavenet.output_projection.register_forward_hook(capture("output_projection")))

    with torch.no_grad():
        stages["output"] = wavenet(spec, diffusion_step, condition).contiguous()
    for handle in handles:
        handle.remove()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(stages, args.output)


if __name__ == "__main__":
    main()
