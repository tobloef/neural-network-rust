use std::sync::Mutex;

use crate::neural_network::layers::{xavier, Layer};
use lazy_static::lazy_static;
use ndarray::{Array1, ArrayView1};
use rand::RngCore;

lazy_static! {
    static ref GLOBAL: Mutex<f32> = Mutex::new(f32::MAX);
}

pub fn softmax(input_size: usize, output_size: usize, rng: &mut Box<dyn RngCore>) -> Layer {
    let (weights, biases) = xavier(input_size, output_size, rng);

    Layer {
        activation_function: activation_function,
        activation_function_derivative: activation_function_derivative,
        weights,
        biases,
    }
}

fn activation_function(x: ArrayView1<f32>) -> Array1<f32> {
    let max = x.fold(f32::MIN, |acc, &x| acc.max(x));
    let exp = x.mapv(|x| (x - max).exp());
    let sum = exp.sum();

    let result = exp / sum;

    result
}

fn activation_function_derivative(x: ArrayView1<f32>) -> Array1<f32> {
    let softmax = activation_function(x);

    let result = softmax.mapv(|x| x * (1.0 - x));

    result
}
