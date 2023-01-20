use ndarray::{Array1, ArrayView1};
use rand::RngCore;

use crate::neural_network::layers::{xavier, Layer};

pub fn tanh(input_size: usize, output_size: usize, rng: &mut Box<dyn RngCore>) -> Layer {
    let (weights, biases) = xavier(input_size, output_size, rng);

    Layer {
        activation_function: activation_function,
        activation_function_derivative: activation_function_derivative,
        weights,
        biases,
    }
}

fn activation_function(x: ArrayView1<f32>) -> Array1<f32> {
    x.mapv(|x| x.tanh())
}

fn activation_function_derivative(x: ArrayView1<f32>) -> Array1<f32> {
    x.mapv(|x| 1.0 - x.tanh().powi(2))
}
