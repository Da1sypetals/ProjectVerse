use anyhow::Result;
use mlx_rs::Array;
use mlx_rs::module::Module;
use mlx_rs::nn::{self, Conv1d};
use mlx_rs::ops::indexing::{IndexOp, IntoStrideBy};
use mlx_rs::ops::{concatenate_axis, tanh, zeros_like};

use crate::weights::Weights;

const CHANNELS: i32 = 192;
const HALF_CHANNELS: i32 = CHANNELS / 2;
const LAYERS: usize = 4;

#[derive(Debug)]
struct WaveNet {
    cond_layer: Conv1d,
    in_layers: Vec<Conv1d>,
    res_skip_layers: Vec<Conv1d>,
}

impl WaveNet {
    fn load(weights: &mut Weights, flow_index: usize) -> Result<Self> {
        let prefix = format!("flow.flows.{flow_index}.enc");
        Ok(Self {
            cond_layer: weights.conv1d(
                &format!("{prefix}.cond_layer"),
                768,
                CHANNELS * 2 * LAYERS as i32,
                1,
                0,
                1,
                1,
                1,
            )?,
            in_layers: (0..LAYERS)
                .map(|index| {
                    weights.conv1d(
                        &format!("{prefix}.in_layers.{index}"),
                        CHANNELS,
                        CHANNELS * 2,
                        5,
                        2,
                        1,
                        1,
                        1,
                    )
                })
                .collect::<Result<Vec<_>>>()?,
            res_skip_layers: (0..LAYERS)
                .map(|index| {
                    let output_channels = if index + 1 < LAYERS {
                        CHANNELS * 2
                    } else {
                        CHANNELS
                    };
                    weights.conv1d(
                        &format!("{prefix}.res_skip_layers.{index}"),
                        CHANNELS,
                        output_channels,
                        1,
                        0,
                        1,
                        1,
                        1,
                    )
                })
                .collect::<Result<Vec<_>>>()?,
        })
    }

    fn forward(&mut self, input: &Array, mask: &Array, speaker: &Array) -> Result<Array> {
        let condition = self.cond_layer.forward(speaker)?;
        let mut hidden = input.clone();
        let mut output = zeros_like(input)?;
        for index in 0..LAYERS {
            let input_activation = self.in_layers[index].forward(&hidden)?;
            let offset = index as i32 * CHANNELS * 2;
            let condition_activation = condition.index((.., .., offset..(offset + CHANNELS * 2)));
            let activations = (input_activation + condition_activation).split(2, -1)?;
            let activations = tanh(&activations[0])? * nn::sigmoid(&activations[1])?;
            let residual_skip = self.res_skip_layers[index].forward(&activations)?;
            if index + 1 < LAYERS {
                let parts = residual_skip.split(2, -1)?;
                hidden = (hidden + &parts[0]) * mask;
                output = output + &parts[1];
            } else {
                output = output + residual_skip;
            }
        }
        Ok(output * mask)
    }
}

#[derive(Debug)]
struct ResidualCoupling {
    pre: Conv1d,
    encoder: WaveNet,
    post: Conv1d,
}

impl ResidualCoupling {
    fn load(weights: &mut Weights, flow_index: usize) -> Result<Self> {
        let prefix = format!("flow.flows.{flow_index}");
        Ok(Self {
            pre: weights.conv1d(
                &format!("{prefix}.pre"),
                HALF_CHANNELS,
                CHANNELS,
                1,
                0,
                1,
                1,
                1,
            )?,
            encoder: WaveNet::load(weights, flow_index)?,
            post: weights.conv1d(
                &format!("{prefix}.post"),
                CHANNELS,
                HALF_CHANNELS,
                1,
                0,
                1,
                1,
                1,
            )?,
        })
    }

    fn reverse(&mut self, input: &Array, mask: &Array, speaker: &Array) -> Result<Array> {
        let parts = input.split(2, -1)?;
        let hidden = self.pre.forward(&parts[0])? * mask;
        let hidden = self.encoder.forward(&hidden, mask, speaker)?;
        let mean = self.post.forward(&hidden)? * mask;
        let transformed = (&parts[1] - mean) * mask;
        Ok(concatenate_axis(&[&parts[0], &transformed], -1)?)
    }
}

#[derive(Debug)]
pub(crate) struct ResidualCouplingBlock {
    couplings: Vec<ResidualCoupling>,
}

impl ResidualCouplingBlock {
    pub(crate) fn load(weights: &mut Weights) -> Result<Self> {
        Ok(Self {
            couplings: [0, 2, 4, 6]
                .into_iter()
                .map(|index| ResidualCoupling::load(weights, index))
                .collect::<Result<Vec<_>>>()?,
        })
    }

    pub(crate) fn reverse(
        &mut self,
        input: &Array,
        mask: &Array,
        speaker: &Array,
    ) -> Result<Array> {
        let mut hidden = input.clone();
        for coupling in self.couplings.iter_mut().rev() {
            hidden = hidden.index((.., .., (..).stride_by(-1)));
            hidden = coupling.reverse(&hidden, mask, speaker)?;
        }
        Ok(hidden)
    }
}
