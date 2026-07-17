# sovits-svc-mlx

Rust and MLX inference for so-vits-svc 4.1 GAN, shallow-diffusion, and Flow Matching checkpoints.

Converted checkpoints are organized under `../ckpt/mlx`:

- `gan/e83_s2400/model.safetensors`
- `gan/opencpop/model.safetensors`
- `refine/shallow_diffusion/guan/model.safetensors`
- `refine/flow_matching/opencpop/model.safetensors`
- `encoder/contentvec/model.safetensors`
- `pitch/fcpe/model.safetensors`
- `vocoder/nsf_hifigan/model.safetensors`

Every checkpoint is supplied through its own CLI argument. Runtime loading does not use a manifest.

## Run

```sh
cargo run --release --example infer -- --help
```

The CLI exposes every single-segment and sliced-inference parameter. It accepts 44.1 or 48 kHz audio in every format supported by Babycat and writes `<input-stem>-converted.wav` beside the input file. `--input-gain` accepts integer dB values from `-12` to `12`, defaults to `0`, and is applied before slicing, resampling, and feature extraction. Silence slicing is enabled by default to match the original Python entry point. Use `--shallow-diffusion`, `--flow-matching`, or neither to choose the refiner.

## Flow Matching

```sh
cargo run --release --example flow_matching -- infer input.wav
cargo run --release --example flow_matching -- verify-reference
```

The verification command injects the captured PyTorch entry noise, checks the first WaveNet pass, and aligns the full 50-step endpoint ODE against the reference tensors.

## Web

```sh
cargo run --release --example web -- \
  --bind 127.0.0.1:3000
```

Open `http://127.0.0.1:3000`. The frontend lets the user select no refiner, shallow diffusion, or Flow Matching after GAN inference. It provides Babycat audio drag-and-drop, every inference and slicing parameter, output preview, and WAV download. It accepts 44.1 or 48 kHz input, resamples 48 kHz for the model, and derives `<input-stem>-converted.wav`. Uploaded audio and generated output stay in memory.
