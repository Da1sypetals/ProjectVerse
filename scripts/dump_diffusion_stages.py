import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
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
    assert args.source_dir.is_dir(), args.source_dir
    assert args.checkpoint.is_file(), args.checkpoint
    assert args.reference.is_file(), args.reference
    sys.path.insert(0, str(args.source_dir.resolve()))

    from diffusion.dpm_solver_pytorch import DPM_Solver, NoiseScheduleVP, model_wrapper
    from diffusion.wavenet import WaveNet

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model"]
    prefix = "decoder.denoise_fn."
    state = {key[len(prefix):]: value for key, value in checkpoint.items() if key.startswith(prefix)}
    model = WaveNet(in_dims=128, n_layers=20, n_chans=512, n_hidden=256).eval()
    model.load_state_dict(state, strict=True)

    reference = torch.load(args.reference, map_location="cpu", weights_only=False)
    spec = reference["diffusion"]["x_noisy"]
    timestep = torch.tensor([reference["diffusion"]["t_probe"]], dtype=torch.long)
    cond = reference["diffusion"]["cond"].transpose(1, 2)
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

    handles.append(model.input_projection.register_forward_hook(capture("input_projection")))
    handles.append(model.diffusion_embedding.register_forward_hook(capture("diffusion_embedding")))
    handles.append(model.mlp[0].register_forward_hook(capture("mlp.0")))
    handles.append(model.mlp[1].register_forward_hook(capture("mlp.1")))
    handles.append(model.mlp[2].register_forward_hook(capture("mlp.2")))
    for index, layer in enumerate(model.residual_layers):
        handles.append(layer.register_forward_hook(capture(f"residual_layers.{index}")))
    handles.append(model.skip_projection.register_forward_hook(capture("skip_projection")))
    handles.append(model.output_projection.register_forward_hook(capture("output_projection")))

    with torch.no_grad():
        output = model(spec, timestep, cond)
        noise_schedule = NoiseScheduleVP(schedule="discrete", betas=checkpoint["decoder.betas"][:100])
        model_fn = model_wrapper(
            model,
            noise_schedule,
            model_type="noise",
            model_kwargs={"cond": cond},
        )
        solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")
        dpm_output, dpm_intermediates = solver.sample(
            spec,
            steps=10,
            order=2,
            skip_type="time_uniform",
            method="multistep",
            return_intermediate=True,
        )
    stages["output"] = output.contiguous()
    stages["cuda_reference_output"] = reference["diffusion"]["noise_pred_probe"].contiguous()
    for index, intermediate in enumerate(dpm_intermediates):
        stages[f"dpm.intermediate.{index}"] = intermediate.contiguous()
    stages["dpm.output"] = dpm_output.clone().contiguous()
    stages["dpm.mel"] = (
        (dpm_output.squeeze(1).transpose(1, 2) + 1) / 2 * 14 - 12
    ).contiguous()
    stages["dpm.cuda_reference_mel"] = reference["diffusion"]["mel_refined"].contiguous()
    for handle in handles:
        handle.remove()
    save_file(stages, args.output)


if __name__ == "__main__":
    main()
