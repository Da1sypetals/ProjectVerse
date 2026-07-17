# Project Verse

Project verse converts your voice to any other singer's.

Based on [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc), with optional new refiner based on flow-matching.

## Run

```sh
cargo run --release --example web -- --help
```

## Algorithm

This is an inference-only repo.

The main algorithm is a GAN generator, with an optional refiner based on shallow diffusion, both from so-vits-svc project.

There is also a new alternative refiner based on flow matching, which slightly mitigates the problem of unclear articulation introduced by shallow diffusion. Its base model is converted from the shallow diffusion's by matching signao-to-noise ratio, and is continue-trained for even better quality.