# sovits-svc-mlx

Rust and MLX inference for the supplied so-vits-svc 4.1 GAN and shallow-diffusion checkpoints. The runtime path contains no Python dependency.

The implementation includes ContentVec768 layer 12, FCPE integration through the `fcpe-mlx` submodule, the complete VITS encoder/flow/F0 decoder/NSF generator, Unit2Mel, the 20-layer diffusion WaveNet, DPM-Solver++ order 2, standalone NSF-HiFiGAN, torchaudio-compatible sinc resampling, mel/STFT, volume extraction, silence slicing, chunk padding, overlap crossfade, and RMS-envelope adjustment.

## Checkout

```shell
git submodule update --init --recursive
```

## Checkpoint conversion

The conversion writes convolution kernels directly in MLX channels-last layout. Runtime weight transposition is not used.

```shell
python scripts/convert_checkpoints.py \
  --gan ../ckpt/e83_s2400/G_e83_s2400.pth \
  --diffusion ../ckpt/guan_diffusion/model_1000.pt \
  --contentvec ../ckpt/checkpoint_best_legacy_500.pt \
  --vocoder ../ckpt/nsf_hifigan/model \
  --vocoder-config ../ckpt/nsf_hifigan/config.json \
  --reference ../port/sovits_infer_ref.pt \
  --output-dir artifacts
```

This produces `gan.safetensors`, `diffusion.safetensors`, `contentvec.safetensors`, `vocoder.safetensors`, and the flattened reference tensors.

## Inference

Single-segment shallow-diffusion inference:

```shell
cargo run --release --example infer -- \
  artifacts ../ckpt/fcpe.safetensors input.wav output.wav 0
```

Silence slicing with the Python defaults:

```shell
cargo run --release --example infer_sliced -- \
  artifacts ../ckpt/fcpe.safetensors input.wav output.wav 0
```

The sliced example also accepts optional clip duration, crossfade duration, and crossfade ratio after the pitch-shift argument.

## Numerical alignment

Generate genuine PyTorch CPU stage references before running the checks:

```shell
python scripts/dump_audio_stages.py --reference artifacts/reference.safetensors --output artifacts/audio_stages.safetensors
python scripts/dump_contentvec_stages.py --checkpoint ../ckpt/checkpoint_best_legacy_500.pt --reference artifacts/reference.safetensors --output artifacts/contentvec_stages.safetensors
python scripts/dump_gan_stages.py --source-dir ../so-vits-svc --checkpoint ../ckpt/e83_s2400/G_e83_s2400.pth --config ../ckpt/e83_s2400/config.json --reference ../port/sovits_infer_ref.pt --output artifacts/gan_stages.safetensors
python scripts/dump_diffusion_stages.py --source-dir ../so-vits-svc --checkpoint ../ckpt/guan_diffusion/model_1000.pt --reference artifacts/reference.safetensors --output artifacts/diffusion_stages.safetensors
python scripts/dump_vocoder_stages.py --source-dir ../so-vits-svc --checkpoint ../ckpt/nsf_hifigan/model --reference artifacts/reference.safetensors --output artifacts/vocoder_reference.safetensors
```

Run the strict checks:

```shell
cargo run --release --example align_audio -- artifacts
cargo run --release --example align_slicer -- artifacts
cargo run --release --example align_contentvec -- artifacts
cargo run --release --example align_gan_core -- artifacts
cargo run --release --example align_diffusion -- artifacts
cargo run --release --example align_vocoder -- artifacts
```

FCPE inference uses the `fcpe-mlx` submodule and the supplied `../ckpt/fcpe.safetensors` directly. Its implementation and checkpoint are treated as authoritative and are intentionally excluded from the legacy so-vits-svc FCPE alignment checks.
