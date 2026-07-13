use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, ensure};
use mlx_rs::Array;
use mlx_rs::nn::{Conv1d, ConvTranspose1d, Embedding, GroupNorm, LayerNorm, Linear};

#[derive(Debug)]
pub struct Weights {
    tensors: HashMap<String, Array>,
}

impl Weights {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let tensors = Array::load_safetensors(path)
            .with_context(|| format!("failed to load {}", path.display()))?;
        ensure!(
            !tensors.is_empty(),
            "{} contains no tensors",
            path.display()
        );
        Ok(Self { tensors })
    }

    pub fn take(&mut self, name: &str) -> Result<Array> {
        self.tensors
            .remove(name)
            .with_context(|| format!("missing tensor {name}"))
    }

    pub fn linear(
        &mut self,
        name: &str,
        input_dims: i32,
        output_dims: i32,
        bias: bool,
    ) -> Result<Linear> {
        let mut layer = Linear::new(input_dims, output_dims)?;
        layer.weight.value = self.take(&format!("{name}.weight"))?;
        layer.bias.value = if bias {
            Some(self.take(&format!("{name}.bias"))?)
        } else {
            None
        };
        ensure!(
            layer.weight.shape() == [output_dims, input_dims],
            "invalid {name}.weight shape {:?}",
            layer.weight.shape()
        );
        Ok(layer)
    }

    pub fn conv1d(
        &mut self,
        name: &str,
        input_channels: i32,
        output_channels: i32,
        kernel_size: i32,
        padding: i32,
        dilation: i32,
        stride: i32,
        groups: i32,
    ) -> Result<Conv1d> {
        let mut layer = Conv1d::new(input_channels, output_channels, kernel_size)?;
        layer.weight.value = self.take(&format!("{name}.weight"))?;
        layer.bias.value = Some(self.take(&format!("{name}.bias"))?);
        layer.padding = padding;
        layer.dilation = dilation;
        layer.stride = stride;
        layer.groups = groups;
        ensure!(
            layer.weight.shape() == [output_channels, kernel_size, input_channels / groups],
            "invalid {name}.weight shape {:?}",
            layer.weight.shape()
        );
        Ok(layer)
    }

    pub fn conv1d_without_bias(
        &mut self,
        name: &str,
        input_channels: i32,
        output_channels: i32,
        kernel_size: i32,
        padding: i32,
        dilation: i32,
        stride: i32,
        groups: i32,
    ) -> Result<Conv1d> {
        let mut layer = Conv1d::new(input_channels, output_channels, kernel_size)?;
        layer.weight.value = self.take(&format!("{name}.weight"))?;
        layer.bias.value = None;
        layer.padding = padding;
        layer.dilation = dilation;
        layer.stride = stride;
        layer.groups = groups;
        ensure!(
            layer.weight.shape() == [output_channels, kernel_size, input_channels / groups],
            "invalid {name}.weight shape {:?}",
            layer.weight.shape()
        );
        Ok(layer)
    }

    pub fn conv_transpose1d(
        &mut self,
        name: &str,
        input_channels: i32,
        output_channels: i32,
        kernel_size: i32,
        padding: i32,
        stride: i32,
    ) -> Result<ConvTranspose1d> {
        let mut layer = ConvTranspose1d::new(input_channels, output_channels, kernel_size)?;
        layer.weight.value = self.take(&format!("{name}.weight"))?;
        layer.bias.value = Some(self.take(&format!("{name}.bias"))?);
        layer.padding = padding;
        layer.stride = stride;
        ensure!(
            layer.weight.shape() == [output_channels, kernel_size, input_channels],
            "invalid {name}.weight shape {:?}",
            layer.weight.shape()
        );
        Ok(layer)
    }

    pub fn embedding(&mut self, name: &str, count: i32, dimensions: i32) -> Result<Embedding> {
        let mut layer = Embedding::new(count, dimensions)?;
        layer.weight.value = self.take(&format!("{name}.weight"))?;
        ensure!(
            layer.weight.shape() == [count, dimensions],
            "invalid {name}.weight shape {:?}",
            layer.weight.shape()
        );
        Ok(layer)
    }

    pub fn layer_norm(&mut self, name: &str, dimensions: i32) -> Result<LayerNorm> {
        let mut layer = LayerNorm::new(dimensions)?;
        layer.weight.value = Some(self.take(&format!("{name}.gamma"))?);
        layer.bias.value = Some(self.take(&format!("{name}.beta"))?);
        Ok(layer)
    }

    pub fn standard_layer_norm(&mut self, name: &str, dimensions: i32) -> Result<LayerNorm> {
        let mut layer = LayerNorm::new(dimensions)?;
        layer.weight.value = Some(self.take(&format!("{name}.weight"))?);
        layer.bias.value = Some(self.take(&format!("{name}.bias"))?);
        Ok(layer)
    }

    pub fn group_norm(
        &mut self,
        name: &str,
        group_count: i32,
        dimensions: i32,
    ) -> Result<GroupNorm> {
        let mut layer = GroupNorm::new(group_count, dimensions)?;
        layer.pytorch_compatible = true;
        layer.weight.value = Some(self.take(&format!("{name}.weight"))?);
        layer.bias.value = Some(self.take(&format!("{name}.bias"))?);
        Ok(layer)
    }

    pub fn discard_prefix(&mut self, prefix: &str) -> usize {
        let before = self.tensors.len();
        self.tensors.retain(|name, _| !name.starts_with(prefix));
        before - self.tensors.len()
    }

    pub fn finish(self) -> Result<()> {
        ensure!(
            self.tensors.is_empty(),
            "unused tensors: {}",
            self.tensors.keys().cloned().collect::<Vec<_>>().join(", ")
        );
        Ok(())
    }
}
