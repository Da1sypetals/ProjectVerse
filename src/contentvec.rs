use std::path::Path;

use anyhow::{Result, ensure};
use mlx_rs::Array;
use mlx_rs::builder::Builder;
use mlx_rs::module::Module;
use mlx_rs::nn::{
    self, Conv1d, GroupNorm, LayerNorm, Linear, MultiHeadAttention, MultiHeadAttentionBuilder,
};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::{floor, pad, r#where};
use mlx_rs::quantization::MaybeQuantized;

use crate::weights::Weights;

const FEATURE_CHANNELS: i32 = 512;
const EMBEDDING_DIMENSIONS: i32 = 768;
const FEED_FORWARD_DIMENSIONS: i32 = 3072;
const ATTENTION_HEADS: i32 = 12;

#[derive(Debug)]
struct TransformerLayer {
    attention: MultiHeadAttention,
    attention_norm: LayerNorm,
    feed_forward_1: Linear,
    feed_forward_2: Linear,
    final_norm: LayerNorm,
}

impl TransformerLayer {
    fn load(weights: &mut Weights, index: usize) -> Result<Self> {
        let prefix = format!("encoder.layers.{index}");
        let attention_prefix = format!("{prefix}.self_attn");
        let mut attention = MultiHeadAttentionBuilder::new(EMBEDDING_DIMENSIONS, ATTENTION_HEADS)
            .bias(true)
            .build()?;
        attention.query_proj = MaybeQuantized::new(weights.linear(
            &format!("{attention_prefix}.q_proj"),
            EMBEDDING_DIMENSIONS,
            EMBEDDING_DIMENSIONS,
            true,
        )?);
        attention.key_proj = MaybeQuantized::new(weights.linear(
            &format!("{attention_prefix}.k_proj"),
            EMBEDDING_DIMENSIONS,
            EMBEDDING_DIMENSIONS,
            true,
        )?);
        attention.value_proj = MaybeQuantized::new(weights.linear(
            &format!("{attention_prefix}.v_proj"),
            EMBEDDING_DIMENSIONS,
            EMBEDDING_DIMENSIONS,
            true,
        )?);
        attention.output_proj = MaybeQuantized::new(weights.linear(
            &format!("{attention_prefix}.out_proj"),
            EMBEDDING_DIMENSIONS,
            EMBEDDING_DIMENSIONS,
            true,
        )?);
        Ok(Self {
            attention,
            attention_norm: weights.standard_layer_norm(
                &format!("{prefix}.self_attn_layer_norm"),
                EMBEDDING_DIMENSIONS,
            )?,
            feed_forward_1: weights.linear(
                &format!("{prefix}.fc1"),
                EMBEDDING_DIMENSIONS,
                FEED_FORWARD_DIMENSIONS,
                true,
            )?,
            feed_forward_2: weights.linear(
                &format!("{prefix}.fc2"),
                FEED_FORWARD_DIMENSIONS,
                EMBEDDING_DIMENSIONS,
                true,
            )?,
            final_norm: weights
                .standard_layer_norm(&format!("{prefix}.final_layer_norm"), EMBEDDING_DIMENSIONS)?,
        })
    }

    fn forward(&mut self, input: &Array, attention_mask: Option<&Array>) -> Result<Array> {
        let attention = self
            .attention
            .forward((input, input, input, attention_mask))?;
        let hidden = self.attention_norm.forward(&(input + attention))?;
        let feed_forward = self.feed_forward_1.forward(&hidden)?;
        let feed_forward = nn::gelu(feed_forward)?;
        let feed_forward = self.feed_forward_2.forward(&feed_forward)?;
        Ok(self.final_norm.forward(&(hidden + feed_forward))?)
    }
}

#[derive(Debug, Default)]
pub struct ContentVecTrace {
    pub feature_convolutions: Vec<Array>,
    pub feature_blocks: Vec<Array>,
    pub feature_norm: Option<Array>,
    pub feature_projection: Option<Array>,
    pub position_convolution: Option<Array>,
    pub position: Option<Array>,
    pub encoder_input_norm: Option<Array>,
    pub encoder_layers: Vec<Array>,
}

#[derive(Debug)]
pub struct ContentVec {
    feature_convolutions: Vec<Conv1d>,
    first_feature_norm: GroupNorm,
    feature_norm: LayerNorm,
    feature_projection: Linear,
    position_convolution: Conv1d,
    encoder_input_norm: LayerNorm,
    encoder_layers: Vec<TransformerLayer>,
}

impl ContentVec {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let mut weights = Weights::load(path)?;
        let kernels = [10, 3, 3, 3, 3, 2, 2];
        let strides = [5, 2, 2, 2, 2, 2, 2];
        let feature_convolutions = kernels
            .into_iter()
            .zip(strides)
            .enumerate()
            .map(|(index, (kernel, stride))| {
                weights.conv1d_without_bias(
                    &format!("feature_extractor.conv_layers.{index}.0"),
                    if index == 0 { 1 } else { FEATURE_CHANNELS },
                    FEATURE_CHANNELS,
                    kernel,
                    0,
                    1,
                    stride,
                    1,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let model = Self {
            feature_convolutions,
            first_feature_norm: weights.group_norm(
                "feature_extractor.conv_layers.0.2",
                FEATURE_CHANNELS,
                FEATURE_CHANNELS,
            )?,
            feature_norm: weights.standard_layer_norm("layer_norm", FEATURE_CHANNELS)?,
            feature_projection: weights.linear(
                "post_extract_proj",
                FEATURE_CHANNELS,
                EMBEDDING_DIMENSIONS,
                true,
            )?,
            position_convolution: weights.conv1d(
                "encoder.pos_conv.0",
                EMBEDDING_DIMENSIONS,
                EMBEDDING_DIMENSIONS,
                128,
                64,
                1,
                1,
                16,
            )?,
            encoder_input_norm: weights
                .standard_layer_norm("encoder.layer_norm", EMBEDDING_DIMENSIONS)?,
            encoder_layers: (0..12)
                .map(|index| TransformerLayer::load(&mut weights, index))
                .collect::<Result<Vec<_>>>()?,
        };
        ensure!(weights.discard_prefix("mask_emb") == 1, "mask_emb missing");
        ensure!(
            weights.discard_prefix("label_embs_concat") == 1,
            "label_embs_concat missing"
        );
        ensure!(
            weights.discard_prefix("final_proj.") == 2,
            "final_proj tensors missing"
        );
        weights.finish()?;
        Ok(model)
    }

    fn encode_inner(
        &mut self,
        wav: &Array,
        mut trace: Option<&mut ContentVecTrace>,
    ) -> Result<Array> {
        ensure!(wav.ndim() == 2, "wav must have shape [batch, samples]");
        let mut hidden = wav.expand_dims(-1)?;
        for (index, convolution) in self.feature_convolutions.iter_mut().enumerate() {
            hidden = convolution.forward(&hidden)?;
            if let Some(trace) = trace.as_deref_mut() {
                trace.feature_convolutions.push(hidden.clone());
            }
            if index == 0 {
                hidden = self.first_feature_norm.forward(&hidden)?;
            }
            hidden = nn::gelu(hidden)?;
            if let Some(trace) = trace.as_deref_mut() {
                trace.feature_blocks.push(hidden.clone());
            }
        }
        hidden = self.feature_norm.forward(&hidden)?;
        if let Some(trace) = trace.as_deref_mut() {
            trace.feature_norm = Some(hidden.clone());
        }
        hidden = self.feature_projection.forward(&hidden)?;
        if let Some(trace) = trace.as_deref_mut() {
            trace.feature_projection = Some(hidden.clone());
        }

        let mut position = self.position_convolution.forward(&hidden)?;
        if let Some(trace) = trace.as_deref_mut() {
            trace.position_convolution = Some(position.clone());
        }
        position = position.index((.., ..-1, ..));
        position = nn::gelu(position)?;
        if let Some(trace) = trace.as_deref_mut() {
            trace.position = Some(position.clone());
        }
        hidden = self.encoder_input_norm.forward(&(hidden + position))?;
        if let Some(trace) = trace.as_deref_mut() {
            trace.encoder_input_norm = Some(hidden.clone());
        }

        let unpadded_length = hidden.shape()[1];
        let pad_length = (2 - unpadded_length % 2) % 2;
        let attention_mask = if pad_length > 0 {
            hidden = pad(&hidden, &[(0, 0), (0, pad_length), (0, 0)], None, None)?;
            let positions = Array::arange::<_, i32>(None, hidden.shape()[1], None)?;
            Some(
                r#where(
                    &positions.ge(Array::from_int(unpadded_length))?,
                    &Array::from_f32(f32::NEG_INFINITY),
                    &Array::from_f32(0.0),
                )?
                .reshape(&[1, 1, 1, -1])?,
            )
        } else {
            None
        };
        for layer in &mut self.encoder_layers {
            hidden = layer.forward(&hidden, attention_mask.as_ref())?;
            if let Some(trace) = trace.as_deref_mut() {
                trace.encoder_layers.push(hidden.clone());
            }
        }
        Ok(if pad_length > 0 {
            hidden.index((.., ..unpadded_length, ..))
        } else {
            hidden
        })
    }

    pub fn encode(&mut self, wav: &Array) -> Result<Array> {
        self.encode_inner(wav, None)
    }

    pub fn encode_traced(&mut self, wav: &Array) -> Result<(Array, ContentVecTrace)> {
        let mut trace = ContentVecTrace::default();
        let output = self.encode_inner(wav, Some(&mut trace))?;
        Ok((output, trace))
    }

    pub fn expand_nearest(features: &Array, target_length: i32) -> Result<Array> {
        ensure!(
            features.ndim() == 3,
            "features must have shape [batch, frames, channels]"
        );
        ensure!(target_length > 0, "target length must be positive");
        let source_length = features.shape()[1];
        let indices = floor(
            &(Array::arange::<_, f32>(None, target_length, None)? * source_length as f32
                / target_length as f32),
        )?
        .as_type::<i32>()?;
        Ok(features.take_axis(indices, 1)?)
    }
}
