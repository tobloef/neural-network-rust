use crate::{
    neural_network::{Network, Predict, Test, TestingConfig, TestingResult},
    utils::{arrayview1_eqauls, indexes_to_n_hot, n_hot_to_indexes, output_to_n_hot},
};
use ndarray::Array1;
use std::collections::HashMap;

pub struct EnsembleNetwork {
    pub networks: Vec<Network>,
}
impl EnsembleNetwork {
    pub fn new(networks: Vec<Network>) -> Self {
        if networks.len() == 0 {
            panic!("Ensemble network must have at least one network");
        }

        Self { networks }
    }
}

impl Predict for EnsembleNetwork {
    fn predict(&self, input: ndarray::ArrayView1<f32>, hot_outputs: usize) -> Array1<f32> {
        let mut all_predictions = Vec::new();

        for network in &self.networks {
            let prediction = network.predict(input, hot_outputs);
            all_predictions.push(prediction);
        }

        get_majority_prediction(all_predictions, hot_outputs)
    }
}

impl Test for EnsembleNetwork {
    fn test(&self, config: &TestingConfig) -> TestingResult {
        let stat_time = std::time::Instant::now();
        let mut correct_predictions = 0;

        let data_sets = (config.inputs)
            .outer_iter()
            .zip(config.expected_outputs.outer_iter());

        let data_set_length = data_sets.len();

        println!("Testing on {data_set_length} data points...");

        for (input, expected_output) in data_sets {
            let predicted_output = self.predict(input.view(), config.hot_outputs);

            let predicted_output_n_hot =
                output_to_n_hot(predicted_output.view(), config.hot_outputs);

            if arrayview1_eqauls(expected_output, predicted_output_n_hot.view()) {
                correct_predictions += 1;
            }
        }

        let accuracy = correct_predictions as f32 / data_set_length as f32;

        TestingResult {
            duration: stat_time.elapsed(),
            accuracy,
        }
    }
}

fn get_majority_prediction(predictions: Vec<Array1<f32>>, hot_outputs: usize) -> Array1<f32> {
    let classes = predictions.to_owned()[0].len();

    let mut prediction_votes = HashMap::new();

    for prediction in predictions {
        let labels = n_hot_to_indexes(prediction.view(), hot_outputs);
        let entry = prediction_votes.entry(labels).or_insert(0);
        *entry += 1;
    }

    let majority_prediction = prediction_votes
        .iter()
        .max_by(|(_, a), (_, b)| a.cmp(b))
        .unwrap()
        .0
        .to_owned();

    indexes_to_n_hot(majority_prediction, classes)
}
