use ndarray::{Array1, ArrayView1};

use crate::neural_network::Network;

use super::Predict;

impl Predict for Network {
    fn predict(&self, input: ArrayView1<f32>, hot_outputs: usize) -> Array1<f32> {
        let mut activations = input.to_owned();

        for layer_index in 0..self.layers.len() {
            let result = self.layers[layer_index].forward(activations.view());
            activations = result.output;
        }

        activations
    }
}
