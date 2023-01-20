use crate::neural_network::layers::{he, Layer};
use ndarray::{Array1, ArrayView1};
use rand::RngCore;

const LEAKY_RELU_SLOPE: f32 = 0.1;

pub fn leaky_relu(input_size: usize, output_size: usize, rng: &mut Box<dyn RngCore>) -> Layer {
    let (weights, biases) = he(input_size, output_size, rng);

    Layer {
        activation_function: activation_function,
        activation_function_derivative: activation_function_derivative,
        weights,
        biases,
    }
}

fn activation_function(array: ArrayView1<f32>) -> Array1<f32> {
    array.mapv(|x| x.max(x * LEAKY_RELU_SLOPE))
}

fn activation_function_derivative(array: ArrayView1<f32>) -> Array1<f32> {
    array.mapv(|x| if x > 0.0 { 1.0 } else { LEAKY_RELU_SLOPE })
}
