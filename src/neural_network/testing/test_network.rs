use crate::{
    neural_network::{Network, Predict},
    utils::{arrayview1_eqauls, n_hot_to_indexes, output_to_n_hot},
};

use super::{Test, TestingConfig, TestingResult};

impl Test for Network {
    fn test(&self, config: &TestingConfig) -> TestingResult {
        let stat_time = std::time::Instant::now();
        let mut correct_predictions = 0;

        let data_sets = (config.inputs)
            .outer_iter()
            .zip(config.expected_outputs.outer_iter());

        let data_set_length = data_sets.len();

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
