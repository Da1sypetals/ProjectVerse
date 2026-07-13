import argparse
from pathlib import Path

import torch
from fairseq import checkpoint_utils
from safetensors.torch import load_file, save_file
from torch.nn import functional as F


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    assert args.checkpoint.is_file(), args.checkpoint
    assert args.reference.is_file(), args.reference
    models, _saved_cfg, _task = checkpoint_utils.load_model_ensemble_and_task(
        [str(args.checkpoint.resolve())]
    )
    model = models[0].eval()
    reference = load_file(args.reference)
    wav = reference["features.wav16k"].unsqueeze(0)
    padding_mask = torch.zeros_like(wav, dtype=torch.bool)
    stages = {"input": wav.contiguous()}
    handles = []

    def capture_channel_first(name):
        def hook(_module, _inputs, output):
            stages[name] = output.detach().transpose(1, 2).contiguous()

        return hook

    def capture_channel_last(name):
        def hook(_module, _inputs, output):
            stages[name] = output.detach().contiguous()

        return hook

    def capture_time_first(name):
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            stages[name] = value.detach().transpose(0, 1).contiguous()

        return hook

    for index, block in enumerate(model.feature_extractor.conv_layers):
        handles.append(
            block[0].register_forward_hook(capture_channel_first(f"feature.conv.{index}"))
        )
        handles.append(
            block.register_forward_hook(capture_channel_first(f"feature.block.{index}"))
        )
    handles.append(model.layer_norm.register_forward_hook(capture_channel_last("feature.norm")))
    handles.append(
        model.post_extract_proj.register_forward_hook(capture_channel_last("feature.projection"))
    )
    handles.append(
        model.encoder.pos_conv[0].register_forward_hook(
            capture_channel_first("encoder.position_conv")
        )
    )
    handles.append(
        model.encoder.pos_conv.register_forward_hook(capture_channel_first("encoder.position"))
    )
    handles.append(
        model.encoder.layer_norm.register_forward_hook(
            capture_channel_last("encoder.input_norm")
        )
    )
    for index, layer in enumerate(model.encoder.layers):
        handles.append(
            layer.register_forward_hook(capture_time_first(f"encoder.layer.{index}"))
        )

    with torch.no_grad():
        output = model.extract_features(
            source=wav,
            padding_mask=padding_mask,
            output_layer=12,
        )[0]
        expanded = F.interpolate(output.transpose(1, 2), size=258, mode="nearest")

    stages["output"] = output.detach().clone().contiguous()
    stages["expanded"] = expanded.transpose(1, 2).contiguous()
    stages["reference_expanded"] = reference["features.c"].transpose(1, 2).contiguous()
    for handle in handles:
        handle.remove()
    save_file(stages, args.output)


if __name__ == "__main__":
    main()
