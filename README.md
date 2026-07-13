# sovits-svc-mlx

Rust and MLX inference for the supplied so-vits-svc 4.1 GAN and shallow-diffusion checkpoints

## Run

```sh
cargo run --release --example infer -- --help
```

The CLI exposes every single-segment and sliced-inference parameter. It accepts 44.1 or 48 kHz audio in every format supported by Babycat and writes `<input-stem>-converted.wav` beside the input file. Silence slicing is enabled by default to match the original Python inference entry point; pass `--no-slicing` to disable it.

## Web

```sh
cargo run --release --example web -- \
  --artifact-dir artifacts \
  --fcpe-checkpoint ../ckpt/fcpe.safetensors \
  --bind 127.0.0.1:3000
```

Open `http://127.0.0.1:3000`. The self-contained frontend provides Babycat audio drag-and-drop, every inference and slicing parameter, output preview, and WAV download. It accepts 44.1 or 48 kHz input, resamples 48 kHz for the model, and always derives `<input-stem>-converted.wav`; the output name cannot be overridden. Uploaded audio and generated output stay in memory.
