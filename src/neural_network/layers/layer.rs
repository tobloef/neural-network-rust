use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::utils::{to_2d, to_2d_transposed};

pub struct ForwardResult {
    pub output: Array1<f32>,
}

pub struct BackwardResult {
    pub input_effect_error: Array1<f32>,
    pub weight_effect_error: Array2<f32>,
    pub bias_effect_error: Array1<f32>,
}

struct DenseBackwardResult {
    input_effect_error: Array1<f32>,
    weight_effect_error: Array2<f32>,
    bias_effect_error: Array1<f32>,
}

struct ActivationBackwardResult {
    input_effect_error: Array1<f32>,
}

#[derive(Clone)]
pub struct Layer {
    pub weights: Array2<f32>,
    pub biases: Array1<f32>,
    pub activation_function: fn(ArrayView1<f32>) -> Array1<f32>,
    pub activation_function_derivative: fn(ArrayView1<f32>) -> Array1<f32>,
}

impl Layer {
    pub fn forward(&self, input: ArrayView1<f32>) -> ForwardResult {
        let dense_output = self.dense_forward(input);
        let activation_output = self.activation_forward(dense_output.view());

        ForwardResult {
            output: activation_output,
        }
    }

    fn dense_forward(&self, input: ArrayView1<f32>) -> Array1<f32> {
        &self.weights.dot(&input) + &self.biases
    }

    fn activation_forward(&self, dense_output: ArrayView1<f32>) -> Array1<f32> {
        (self.activation_function)(dense_output)
    }

    pub fn backward(
        &self,
        input: ArrayView1<f32>,
        output_effect_error: ArrayView1<f32>,
    ) -> BackwardResult {
        let dense_output = self.dense_forward(input);

        let ActivationBackwardResult {
            input_effect_error: activation_effect_error,
        } = self.activation_backward(dense_output.view(), output_effect_error);

        let dense_backward_result = self.dense_backward(input, activation_effect_error.view());

        BackwardResult {
            input_effect_error: dense_backward_result.input_effect_error,
            weight_effect_error: dense_backward_result.weight_effect_error,
            bias_effect_error: dense_backward_result.bias_effect_error,
        }
    }

    fn dense_backward(
        &self,
        input: ArrayView1<f32>,
        activation_effect_error: ArrayView1<f32>,
    ) -> DenseBackwardResult {
        let output_effect_error_2d_transposed = to_2d_transposed(activation_effect_error);
        let input_2d = to_2d(input);
        let weight_effect_error = output_effect_error_2d_transposed.dot(&input_2d);

        let weights_transposed = self.weights.t();
        let input_effect_error: Array1<f32> = weights_transposed.dot(&activation_effect_error);

        let bias_effect_error = activation_effect_error.to_owned();

        DenseBackwardResult {
            input_effect_error,
            weight_effect_error,
            bias_effect_error,
        }
    }

    fn activation_backward(
        &self,
        dense_output: ArrayView1<f32>,
        output_effect_error: ArrayView1<f32>,
    ) -> ActivationBackwardResult {
        let input_effect_output = (self.activation_function_derivative)(dense_output);
        let input_effect_error = &output_effect_error * &input_effect_output;

        ActivationBackwardResult { input_effect_error }
    }

    pub fn learn(
        &mut self,
        delta_weights: ArrayView2<f32>,
        delta_bias: ArrayView1<f32>,
        learning_rate: f32,
    ) {
        self.weights -= &(&delta_weights * learning_rate);
        self.biases -= &(&delta_bias * learning_rate);
    }
}
