use super::{Train, TrainingConfig, TrainingResult};
use crate::neural_network::{layers::Layer, Network, Test, TestingResult};
use ndarray::{Array1, Array2, ArrayView1};
use rand::seq::SliceRandom;
use rayon::prelude::*;
use std::time::Duration;

impl Train for Network {
    fn train(&mut self, config: &TrainingConfig) -> TrainingResult {
        config.logger.log_network_start(config, self);

        let start_time = std::time::Instant::now();

        let mut epoch_results: Vec<EpochResult> = Vec::new();
        let mut last_epoch_result: Option<EpochResult> = None;

        let mut best_network: Option<Network> = None;
        let mut best_accuracy: Option<f32> = None;

        let training_sets = get_training_sets(&config);

        let max_epochs = config.max_epochs.unwrap_or(usize::MAX);

        for epoch in 0..max_epochs {
            let epoch_result: EpochResult = train_epoch(self, &config, epoch, &training_sets);

            epoch_results.push(epoch_result.clone());

            let should_stop_early = check_early_stop(&epoch_result, config);

            last_epoch_result = Some(epoch_result);

            let accuracy = last_epoch_result.unwrap().accuracy;
            if best_accuracy.is_none() || accuracy > best_accuracy {
                best_network = Some(self.clone());
                best_accuracy = accuracy;
            }

            if should_stop_early {
                break;
            }
        }

        let final_epoch_result = last_epoch_result.unwrap();
        let duration = start_time.elapsed();

        let result = TrainingResult {
            error: final_epoch_result.error,
            accuracy: final_epoch_result.accuracy,
            epochs: final_epoch_result.epoch + 1,
            duration,
            best_accuracy,
            best_network,
        };

        config.logger.log_network_end(&result);

        result
    }
}

pub type TrainingSet<'a> = (ArrayView1<'a, f32>, ArrayView1<'a, f32>);

fn get_training_sets<'a>(config: &'a TrainingConfig) -> Vec<TrainingSet<'a>> {
    config
        .training_data
        .inputs
        .outer_iter()
        .zip(config.training_data.expected_outputs.outer_iter())
        .collect::<Vec<TrainingSet<'a>>>()
}

fn check_early_stop(epoch_result: &EpochResult, config: &TrainingConfig) -> bool {
    let EpochResult {
        error, accuracy, ..
    } = *epoch_result;

    let use_error_goal = config.error_goal.is_some();
    let error_goal_met = Some(error) <= config.error_goal;

    let use_accuracy_goal = config.accuracy_goal.is_some() && accuracy.is_some();
    let accuracy_goal_met = accuracy >= config.accuracy_goal;

    let used_goals_met = match (use_error_goal, use_accuracy_goal) {
        (true, true) => error_goal_met && accuracy_goal_met,
        (true, false) => error_goal_met,
        (false, true) => accuracy_goal_met,
        (false, false) => false,
    };

    let use_min_epochs = config.min_epochs.is_some();
    let min_epochs_met = Some(epoch_result.epoch) >= config.min_epochs;

    (!use_min_epochs || min_epochs_met) && used_goals_met
}

#[derive(Clone, Copy)]
pub struct EpochResult {
    pub error: f32,
    pub accuracy: Option<f32>,
    pub duration: Duration,
    pub epoch: usize,
}

fn train_epoch(
    network: &mut Network,
    config: &TrainingConfig,
    epoch: usize,
    training_sets: &Vec<TrainingSet<'_>>,
) -> EpochResult {
    config.logger.log_epoch_start(config, epoch);

    let start_time = std::time::Instant::now();
    let mut total_error = None;

    let mut shuffled_training_sets = training_sets.clone();
    shuffled_training_sets.shuffle(&mut network.rng);

    let batch_size = config.batch_size.unwrap_or(training_sets.len());
    let batches = shuffled_training_sets.chunks(batch_size);
    let batches_count = batches.len();

    for (batch_index, batch) in batches.enumerate() {
        let batch_result = train_batch(network, config, epoch, batch_index, batch);

        total_error = Some(total_error.unwrap_or(0.0) + batch_result.error);
    }

    let test_result = try_test(network, config);

    let error = total_error.unwrap() / batches_count as f32;
    let accuracy = test_result.map(|result| result.accuracy);
    let duration = start_time.elapsed();

    let result = EpochResult {
        error,
        accuracy,
        duration,
        epoch,
    };

    config.logger.log_epoch_end(&result);

    result
}

fn try_test(network: &Network, config: &TrainingConfig) -> Option<TestingResult> {
    if let Some(testing_config) = &config.testing_config {
        let testing_network = network.clone();
        let result = testing_network.test(testing_config);

        Some(result)
    } else {
        None
    }
}

pub struct BatchResult {
    pub error: f32,
    pub duration: Duration,
}

// TODO: Refactor this and the mapper/reducer thing
fn train_batch(
    network: &mut Network,
    config: &TrainingConfig,
    epoch: usize,
    batch_index: usize,
    batch: &[TrainingSet<'_>],
) -> BatchResult {
    config
        .logger
        .log_batch_start(config, batch_index, batch.len());

    let start_time = std::time::Instant::now();

    let layers = &network.layers;
    let error_function = network.error_function;
    let error_function_derivative = network.error_function_derivative;

    let do_parallel = config.parallel && batch.len() > 1;

    let (total_error, total_layer_delta_weights, total_layer_delta_biases) = if do_parallel {
        let identity = (
            0.0,
            layers
                .iter()
                .map(|l| Array2::zeros(l.weights.dim()))
                .collect(),
            layers
                .iter()
                .map(|l| Array1::zeros(l.biases.dim()))
                .collect(),
        );

        batch
            .iter()
            .enumerate()
            .par_bridge()
            .map(|(i, td)| mapper(layers, error_function, error_function_derivative, i, td))
            .reduce(|| identity.to_owned(), reducer)
    } else {
        batch
            .iter()
            .enumerate()
            .map(|(i, td)| mapper(layers, error_function, error_function_derivative, i, td))
            .reduce(reducer)
            .unwrap()
    };

    let learning_rate = config
        .learning_rates
        .iter()
        .rev()
        .find(|(e, r)| epoch >= *e)
        .expect(format!("no suitable learning rate found for epoch {}", epoch).as_str())
        .1;

    apply_learning(
        network,
        learning_rate,
        batch.len(),
        total_layer_delta_weights,
        total_layer_delta_biases,
    );

    let error = total_error / batch.len() as f32;
    let duration = start_time.elapsed();

    let result = BatchResult { error, duration };

    config.logger.log_batch_end(&result);

    result
}

type ReducerItem = (f32, Vec<Array2<f32>>, Vec<Array1<f32>>);

// TODO: Needs a better name
fn mapper(
    layers: &Vec<Layer>,
    error_function: ErrorFunction,
    error_function_derivative: ErrorFunctionDerivative,
    index: usize,
    (input, expected_output): &TrainingSet<'_>,
) -> ReducerItem {
    let forward_result = propagate_forward(layers, *input);

    let predicted_output = forward_result.output.view();

    let error = get_error(error_function, predicted_output, *expected_output);

    let backward_result = propagate_backward(
        layers,
        error_function_derivative,
        predicted_output,
        *expected_output,
        forward_result.layer_inputs,
    );

    (
        error,
        backward_result.layer_delta_weights,
        backward_result.layer_delta_biases,
    )
}

// TODO: Needs a better name
fn reducer(a: ReducerItem, b: ReducerItem) -> ReducerItem {
    let (a_error, a_layer_delta_weights, a_layer_delta_biases) = a;
    let (b_error, b_layer_delta_weights, b_layer_delta_biases) = b;

    let sum_error = a_error + b_error;
    let sum_layer_delta_weights = a_layer_delta_weights
        .iter()
        .zip(b_layer_delta_weights.iter())
        .map(|(a, b)| a + b)
        .collect();
    let sum_layer_delta_biases = a_layer_delta_biases
        .iter()
        .zip(b_layer_delta_biases.iter())
        .map(|(a, b)| a + b)
        .collect();

    (sum_error, sum_layer_delta_weights, sum_layer_delta_biases)
}

struct PropagateForwardResult {
    output: Array1<f32>,
    layer_inputs: Vec<Array1<f32>>,
}

fn propagate_forward(layers: &Vec<Layer>, input: ArrayView1<f32>) -> PropagateForwardResult {
    let mut layer_inputs: Vec<Array1<f32>> = Vec::new();

    let mut activations: Array1<f32> = input.to_owned();

    for layer_index in 0..layers.len() {
        layer_inputs.push(activations.to_owned());

        let result = layers[layer_index].forward(activations.view());

        activations = result.output;
    }

    let output: Array1<f32> = activations.clone().to_owned();

    PropagateForwardResult {
        output,
        layer_inputs,
    }
}

struct PropagateBackwardResult {
    layer_delta_weights: Vec<Array2<f32>>,
    layer_delta_biases: Vec<Array1<f32>>,
}

type ErrorFunction = fn(ArrayView1<f32>, ArrayView1<f32>) -> f32;
type ErrorFunctionDerivative = fn(ArrayView1<f32>, ArrayView1<f32>) -> Array1<f32>;

fn propagate_backward<'a>(
    layers: &Vec<Layer>,
    error_function_derivative: ErrorFunctionDerivative,
    predicted_output: ArrayView1<f32>,
    expected_output: ArrayView1<f32>,
    layer_inputs: Vec<Array1<f32>>,
) -> PropagateBackwardResult {
    let mut layer_delta_weights_reverse: Vec<Array2<f32>> = Vec::new();
    let mut layer_delta_biases_reverse: Vec<Array1<f32>> = Vec::new();

    let mut output_effect_error = (error_function_derivative)(predicted_output, expected_output);

    for layer_index in (0..layers.len()).rev() {
        let input = &layer_inputs[layer_index];
        let result = layers[layer_index].backward(input.view(), output_effect_error.view());

        layer_delta_weights_reverse.push(result.weight_effect_error);
        layer_delta_biases_reverse.push(result.bias_effect_error);

        output_effect_error = result.input_effect_error;
    }

    let layer_delta_weights: Vec<Array2<f32>> =
        layer_delta_weights_reverse.into_iter().rev().collect();
    let layer_delta_biases: Vec<Array1<f32>> =
        layer_delta_biases_reverse.into_iter().rev().collect();

    PropagateBackwardResult {
        layer_delta_weights,
        layer_delta_biases,
    }
}

fn apply_learning(
    network: &mut Network,
    learning_rate: f32,
    batch_size: usize,
    delta_weights: Vec<Array2<f32>>,
    delta_biases: Vec<Array1<f32>>,
) {
    for layer_index in 0..network.layers.len() {
        let delta_weights = delta_weights[layer_index].view();
        let delta_bias = delta_biases[layer_index].view();

        network.layers[layer_index].learn(
            delta_weights,
            delta_bias,
            learning_rate / batch_size as f32,
        )
    }
}

fn get_error(
    error_function: ErrorFunction,
    predicted_output: ArrayView1<f32>,
    expected_output: ArrayView1<f32>,
) -> f32 {
    (error_function)(predicted_output, expected_output)
}
