use crate::neural_network::layers::{xavier, Layer};
use ndarray::{Array1, ArrayView1};
use rand::RngCore;

pub fn swish(input_size: usize, output_size: usize, rng: &mut Box<dyn RngCore>) -> Layer {
    let (weights, biases) = xavier(input_size, output_size, rng);

    Layer {
        activation_function: activation_function,
        activation_function_derivative: activation_function_derivative,
        weights,
        biases,
    }
}

fn activation_function(array: ArrayView1<f32>) -> Array1<f32> {
    array.mapv(|x| x / (1.0 + (-x).exp()))
}

fn activation_function_derivative(array: ArrayView1<f32>) -> Array1<f32> {
    array.mapv(|x| ((-x).exp() * (x + 1.0) + 1.0) / (1.0 + (-x).exp()).powi(2))
}
