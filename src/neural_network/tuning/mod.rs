use crate::utils::to_percent;

use super::{
    logging::FileLogger, training::TrainingData, ErrorFunctionType, LayerType, Network,
    NetworkConfigBuilder, Train, TrainingConfigBuilder,
};
use itertools::iproduct;
use rayon::prelude::*;
use std::{path::Path, slice::Iter};

pub struct Tuner<'a> {
    pub folder: &'a str,
    pub training_data: &'a TrainingData<'a>,
    pub testing_data: &'a TrainingData<'a>,
    pub seed: &'a str,
    pub epochs: usize,
    pub samples: usize,
    pub input_size: usize,
    pub output_size: usize,
    pub hidden_layer_types: Iter<'a, fn(usize, usize) -> LayerType>,
    pub hidden_layer_counts: Iter<'a, usize>,
    pub hidden_layer_size_params: Iter<'a, (f32, f32)>,
    pub output_layer_types: Iter<'a, fn(usize, usize) -> LayerType>,
    pub error_function_types: Iter<'a, ErrorFunctionType>,
    pub learning_rates: Iter<'a, f32>,
    pub batch_sizes: Iter<'a, usize>,
}

impl<'a> Tuner<'a> {
    pub fn run(&mut self, skip: usize) {
        let combined_iterator = iproduct!(
            self.hidden_layer_types.to_owned(),
            self.hidden_layer_counts.to_owned(),
            self.hidden_layer_size_params.to_owned(),
            self.output_layer_types.to_owned(),
            self.error_function_types.to_owned(),
            self.learning_rates.to_owned(),
            self.batch_sizes.to_owned()
        );

        combined_iterator
            .enumerate()
            .skip(skip)
            .par_bridge()
            .for_each(|(i, params)| {
                for sample in 0..self.samples {
                    let (
                        hidden_layer_type,
                        hidden_layer_count,
                        hidden_layer_size_params,
                        output_layer_type,
                        error_function_type,
                        learning_rate,
                        batch_size,
                    ) = params;

                    let file_path = format!("{}/{}-{}", self.folder, i, sample);

                    if Path::new(&file_path).exists() {
                        println!("Skipping {}", file_path);

                        continue;
                    }

                    let mut layers = vec![];

                    let mut prev_size = self.input_size;

                    for i in 0..*hidden_layer_count {
                        let next_size = if (i + 1) == *hidden_layer_count {
                            self.output_size
                        } else {
                            hidden_layer_size_formula(
                                self.input_size as f32,
                                *hidden_layer_count as f32,
                                (i + 1) as f32,
                                hidden_layer_size_params.0,
                                hidden_layer_size_params.1,
                            ) as usize
                        };

                        layers.push(hidden_layer_type(prev_size, next_size));
                        prev_size = next_size;
                    }

                    layers.push(output_layer_type(prev_size, self.output_size));

                    let seed = format!("{}-{}-{}", self.seed.to_owned(), i, sample);

                    let network_config = NetworkConfigBuilder::new()
                        .layers(layers)
                        .error_function(*error_function_type)
                        .seed(seed.as_str())
                        .build();

                    let training_config = TrainingConfigBuilder::new()
                        .training_data(
                            self.training_data.inputs,
                            self.training_data.expected_outputs,
                        )
                        .testing_data(self.testing_data.inputs, self.testing_data.expected_outputs)
                        .epochs(self.epochs)
                        .learning_rate(*learning_rate)
                        .batch_size(*batch_size)
                        .logger(Box::new(FileLogger::new(file_path.as_str())))
                        .build();

                    let mut network = Network::new(network_config);
                    let result = network.train(&training_config);

                    println!(
                        "Completed {} ({}/{})",
                        file_path,
                        to_percent(result.accuracy.unwrap_or(0.0)),
                        to_percent(result.best_accuracy.unwrap_or(0.0))
                    );
                }
            });
    }
}

fn hidden_layer_size_formula(inputs: f32, layers: f32, layer: f32, a: f32, b: f32) -> f32 {
    (inputs - (inputs / ((layers + 1.0).powf(a))) * (layer.powf(a))) + b
}
