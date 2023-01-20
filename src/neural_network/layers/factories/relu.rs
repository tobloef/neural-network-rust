use crate::neural_network::layers::{weight_initialization::he, Layer};
use ndarray::{Array1, ArrayView1};
use rand::RngCore;

pub fn relu(input_size: usize, output_size: usize, rng: &mut Box<dyn RngCore>) -> Layer {
    let (weights, biases) = he(input_size, output_size, rng);

    Layer {
        activation_function: activation_function,
        activation_function_derivative: activation_function_derivative,
        weights,
        biases,
    }
}

fn activation_function(array: ArrayView1<f32>) -> Array1<f32> {
    array.mapv(|x| x.max(0.0))
}

fn activation_function_derivative(array: ArrayView1<f32>) -> Array1<f32> {
    array.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })
}
