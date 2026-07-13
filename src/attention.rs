use anyhow::Result;
use mlx_rs::Array;
use mlx_rs::module::Module;
use mlx_rs::nn::{self, Conv1d, LayerNorm};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::{matmul, pad, softmax_axis, tril, r#where};

use crate::weights::Weights;

const CHANNELS: i32 = 192;
const FILTER_CHANNELS: i32 = 768;
const HEADS: i32 = 2;
const HEAD_CHANNELS: i32 = CHANNELS / HEADS;
const WINDOW_SIZE: i32 = 4;

#[derive(Debug)]
struct MultiHeadAttention {
    conv_q: Conv1d,
    conv_k: Conv1d,
    conv_v: Conv1d,
    conv_o: Conv1d,
    emb_rel_k: Option<Array>,
    emb_rel_v: Option<Array>,
}

impl MultiHeadAttention {
    fn load_with_prefix(weights: &mut Weights, prefix: &str, relative: bool) -> Result<Self> {
        Ok(Self {
            conv_q: weights.conv1d(
                &format!("{prefix}.conv_q"),
                CHANNELS,
                CHANNELS,
                1,
                0,
                1,
                1,
                1,
            )?,
            conv_k: weights.conv1d(
                &format!("{prefix}.conv_k"),
                CHANNELS,
                CHANNELS,
                1,
                0,
                1,
                1,
                1,
            )?,
            conv_v: weights.conv1d(
                &format!("{prefix}.conv_v"),
                CHANNELS,
                CHANNELS,
                1,
                0,
                1,
                1,
                1,
            )?,
            conv_o: weights.conv1d(
                &format!("{prefix}.conv_o"),
                CHANNELS,
                CHANNELS,
                1,
                0,
                1,
                1,
                1,
            )?,
            emb_rel_k: relative
                .then(|| weights.take(&format!("{prefix}.emb_rel_k")))
                .transpose()?,
            emb_rel_v: relative
                .then(|| weights.take(&format!("{prefix}.emb_rel_v")))
                .transpose()?,
        })
    }

    fn load_relative(weights: &mut Weights, index: usize) -> Result<Self> {
        Self::load_with_prefix(weights, &format!("enc_p.enc_.attn_layers.{index}"), true)
    }

    fn load_plain(weights: &mut Weights, prefix: &str) -> Result<Self> {
        Self::load_with_prefix(weights, prefix, false)
    }

    fn relative_embeddings(embeddings: &Array, length: i32) -> Result<Array> {
        let pad_length = (length - (WINDOW_SIZE + 1)).max(0);
        let slice_start = ((WINDOW_SIZE + 1) - length).max(0);
        let slice_end = slice_start + 2 * length - 1;
        let padded = if pad_length > 0 {
            pad(
                embeddings,
                &[(0, 0), (pad_length, pad_length), (0, 0)],
                None,
                None,
            )?
        } else {
            embeddings.clone()
        };
        Ok(padded.index((.., slice_start..slice_end, ..)))
    }

    fn relative_to_absolute(x: &Array, length: i32) -> Result<Array> {
        let batch = x.shape()[0];
        let heads = x.shape()[1];
        let x = pad(x, &[(0, 0), (0, 0), (0, 0), (0, 1)], None, None)?;
        let x = x.reshape(&[batch, heads, length * 2 * length])?;
        let x = pad(&x, &[(0, 0), (0, 0), (0, length - 1)], None, None)?;
        let x = x.reshape(&[batch, heads, length + 1, 2 * length - 1])?;
        Ok(x.index((.., .., 0..length, (length - 1)..)))
    }

    fn absolute_to_relative(x: &Array, length: i32) -> Result<Array> {
        let batch = x.shape()[0];
        let heads = x.shape()[1];
        let x = pad(x, &[(0, 0), (0, 0), (0, 0), (0, length - 1)], None, None)?;
        let x = x.reshape(&[batch, heads, length * length + length * (length - 1)])?;
        let x = pad(&x, &[(0, 0), (0, 0), (length, 0)], None, None)?;
        let x = x.reshape(&[batch, heads, length, 2 * length])?;
        Ok(x.index((.., .., .., 1..)))
    }

    fn forward(&mut self, input: &Array, mask: &Array) -> Result<Array> {
        let batch = input.shape()[0];
        let length = input.shape()[1];
        let query = self
            .conv_q
            .forward(input)?
            .reshape(&[batch, length, HEADS, HEAD_CHANNELS])?
            .swap_axes(1, 2)?;
        let key = self
            .conv_k
            .forward(input)?
            .reshape(&[batch, length, HEADS, HEAD_CHANNELS])?
            .swap_axes(1, 2)?;
        let value = self
            .conv_v
            .forward(input)?
            .reshape(&[batch, length, HEADS, HEAD_CHANNELS])?
            .swap_axes(1, 2)?;

        let scaled_query = &query / (HEAD_CHANNELS as f32).sqrt();
        let mut scores = matmul(&scaled_query, key.swap_axes(-2, -1)?)?;
        if let Some(relative_k) = &self.emb_rel_k {
            let relative_k = Self::relative_embeddings(relative_k, length)?;
            let relative_logits =
                matmul(&scaled_query, relative_k.expand_dims(0)?.swap_axes(-2, -1)?)?;
            scores = scores + Self::relative_to_absolute(&relative_logits, length)?;
        }
        scores = r#where(
            &mask.eq(Array::from_int(0))?,
            &Array::from_f32(-1.0e4),
            &scores,
        )?;
        let attention = softmax_axis(&scores, -1, None)?;

        let mut output = matmul(&attention, &value)?;
        if let Some(relative_v) = &self.emb_rel_v {
            let relative_attention = Self::absolute_to_relative(&attention, length)?;
            let relative_v = Self::relative_embeddings(relative_v, length)?;
            output = output + matmul(&relative_attention, relative_v.expand_dims(0)?)?;
        }
        let output = output
            .swap_axes(1, 2)?
            .reshape(&[batch, length, CHANNELS])?;
        Ok(self.conv_o.forward(&output)?)
    }
}

#[derive(Debug)]
struct FeedForward {
    conv_1: Conv1d,
    conv_2: Conv1d,
    causal: bool,
}

impl FeedForward {
    fn load_with_prefix(weights: &mut Weights, prefix: &str, causal: bool) -> Result<Self> {
        Ok(Self {
            conv_1: weights.conv1d(
                &format!("{prefix}.conv_1"),
                CHANNELS,
                FILTER_CHANNELS,
                3,
                if causal { 0 } else { 1 },
                1,
                1,
                1,
            )?,
            conv_2: weights.conv1d(
                &format!("{prefix}.conv_2"),
                FILTER_CHANNELS,
                CHANNELS,
                3,
                if causal { 0 } else { 1 },
                1,
                1,
                1,
            )?,
            causal,
        })
    }

    fn load(weights: &mut Weights, index: usize) -> Result<Self> {
        Self::load_with_prefix(weights, &format!("enc_p.enc_.ffn_layers.{index}"), false)
    }

    fn forward(&mut self, input: &Array, mask: &Array) -> Result<Array> {
        let masked = input * mask;
        let masked = if self.causal {
            pad(&masked, &[(0, 0), (2, 0), (0, 0)], None, None)?
        } else {
            masked
        };
        let hidden = self.conv_1.forward(&masked)?;
        let hidden = nn::relu(hidden)?;
        let hidden = hidden * mask;
        let hidden = if self.causal {
            pad(&hidden, &[(0, 0), (2, 0), (0, 0)], None, None)?
        } else {
            hidden
        };
        Ok(self.conv_2.forward(&hidden)? * mask)
    }
}

#[derive(Debug)]
struct EncoderLayer {
    attention: MultiHeadAttention,
    attention_norm: LayerNorm,
    feed_forward: FeedForward,
    feed_forward_norm: LayerNorm,
}

impl EncoderLayer {
    fn load(weights: &mut Weights, index: usize) -> Result<Self> {
        Ok(Self {
            attention: MultiHeadAttention::load_relative(weights, index)?,
            attention_norm: weights
                .layer_norm(&format!("enc_p.enc_.norm_layers_1.{index}"), CHANNELS)?,
            feed_forward: FeedForward::load(weights, index)?,
            feed_forward_norm: weights
                .layer_norm(&format!("enc_p.enc_.norm_layers_2.{index}"), CHANNELS)?,
        })
    }
}

#[derive(Debug)]
pub(crate) struct Encoder {
    layers: Vec<EncoderLayer>,
}

impl Encoder {
    pub(crate) fn load(weights: &mut Weights) -> Result<Self> {
        Ok(Self {
            layers: (0..6)
                .map(|index| EncoderLayer::load(weights, index))
                .collect::<Result<Vec<_>>>()?,
        })
    }

    pub(crate) fn forward(&mut self, input: &Array, mask: &Array) -> Result<Array> {
        let channel_first_mask = mask.swap_axes(1, 2)?;
        let attention_mask =
            channel_first_mask.expand_dims(2)? * channel_first_mask.expand_dims(-1)?;
        let mut hidden = input * mask;
        for layer in &mut self.layers {
            let attention = layer.attention.forward(&hidden, &attention_mask)?;
            hidden = layer.attention_norm.forward(&(hidden + attention))?;
            let feed_forward = layer.feed_forward.forward(&hidden, mask)?;
            hidden = layer.feed_forward_norm.forward(&(hidden + feed_forward))?;
        }
        Ok(hidden * mask)
    }
}

#[derive(Debug)]
struct CausalLayer {
    attention: MultiHeadAttention,
    attention_norm: LayerNorm,
    feed_forward: FeedForward,
    feed_forward_norm: LayerNorm,
}

impl CausalLayer {
    fn load(weights: &mut Weights, index: usize) -> Result<Self> {
        Ok(Self {
            attention: MultiHeadAttention::load_plain(
                weights,
                &format!("f0_decoder.decoder.self_attn_layers.{index}"),
            )?,
            attention_norm: weights.layer_norm(
                &format!("f0_decoder.decoder.norm_layers_0.{index}"),
                CHANNELS,
            )?,
            feed_forward: FeedForward::load_with_prefix(
                weights,
                &format!("f0_decoder.decoder.ffn_layers.{index}"),
                true,
            )?,
            feed_forward_norm: weights.layer_norm(
                &format!("f0_decoder.decoder.norm_layers_1.{index}"),
                CHANNELS,
            )?,
        })
    }
}

#[derive(Debug)]
pub(crate) struct CausalFft {
    layers: Vec<CausalLayer>,
}

impl CausalFft {
    pub(crate) fn load(weights: &mut Weights) -> Result<Self> {
        Ok(Self {
            layers: (0..6)
                .map(|index| CausalLayer::load(weights, index))
                .collect::<Result<Vec<_>>>()?,
        })
    }

    pub(crate) fn forward(&mut self, input: &Array, mask: &Array) -> Result<Array> {
        let length = input.shape()[1];
        let attention_mask = tril(Array::ones::<f32>(&[length, length])?, None)?
            .expand_dims(0)?
            .expand_dims(0)?;
        let mut hidden = input * mask;
        for layer in &mut self.layers {
            let attention = layer.attention.forward(&hidden, &attention_mask)?;
            hidden = layer.attention_norm.forward(&(hidden + attention))?;
            let feed_forward = layer.feed_forward.forward(&hidden, mask)?;
            hidden = layer.feed_forward_norm.forward(&(hidden + feed_forward))?;
        }
        Ok(hidden * mask)
    }
}
