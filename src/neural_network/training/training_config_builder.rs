use super::{TrainingConfig, TrainingData};
use crate::neural_network::{logging::EmptyLogger, Logger, TestingConfigBuilder};
use ndarray::ArrayView2;

#[derive(Clone)]
pub struct TrainingConfigBuilder<'a> {
    min_epochs: Option<usize>,
    max_epochs: Option<usize>,
    error_goal: Option<f32>,
    accuracy_goal: Option<f32>,
    learning_rates: Option<Vec<(usize, f32)>>,
    batch_size: Option<usize>,
    training_inputs: Option<ArrayView2<'a, f32>>,
    training_expected_outputs: Option<ArrayView2<'a, f32>>,
    testing_inputs: Option<ArrayView2<'a, f32>>,
    testing_expected_outputs: Option<ArrayView2<'a, f32>>,
    logger: Option<Box<dyn Logger>>,
    parallel: Option<bool>,
    hot_outputs: Option<usize>,
}

impl<'a> TrainingConfigBuilder<'a> {
    pub fn new() -> Self {
        Self {
            min_epochs: None,
            max_epochs: None,
            error_goal: None,
            accuracy_goal: None,
            learning_rates: None,
            batch_size: None,
            training_inputs: None,
            training_expected_outputs: None,
            testing_inputs: None,
            testing_expected_outputs: None,
            logger: None,
            parallel: None,
            hot_outputs: None,
        }
    }

    pub fn min_epochs(&mut self, min_epochs: usize) -> &mut Self {
        self.min_epochs = Some(min_epochs);
        self
    }

    pub fn max_epochs(&mut self, max_epochs: usize) -> &mut Self {
        self.max_epochs = Some(max_epochs);
        self
    }

    pub fn epochs(&mut self, epochs: usize) -> &mut Self {
        self.min_epochs = Some(epochs);
        self.max_epochs = Some(epochs);
        self
    }

    pub fn error_goal(&mut self, error_goal: f32) -> &mut Self {
        self.error_goal = Some(error_goal);
        self
    }

    pub fn accuracy_goal(&mut self, accuracy_goal: f32) -> &mut Self {
        self.accuracy_goal = Some(accuracy_goal);
        self
    }

    pub fn learning_rate(&mut self, learning_rate: f32) -> &mut Self {
        self.learning_rates = Some(vec![(0, learning_rate)]);
        self
    }

    pub fn learning_rates(&mut self, learning_rates: Vec<(usize, f32)>) -> &mut Self {
        self.learning_rates = Some(learning_rates);
        self
    }

    pub fn batch_size(&mut self, batch_size: usize) -> &mut Self {
        self.batch_size = Some(batch_size);
        self
    }

    pub fn training_data(
        &mut self,
        inputs: ArrayView2<'a, f32>,
        expected_outputs: ArrayView2<'a, f32>,
    ) -> &mut Self {
        if inputs.shape()[0] != expected_outputs.shape()[0] {
            panic!("training data inputs and expected outputs must have the same number items");
        }

        self.training_inputs = Some(inputs);
        self.training_expected_outputs = Some(expected_outputs);

        self
    }

    pub fn testing_data(
        &mut self,
        inputs: ArrayView2<'a, f32>,
        expected_outputs: ArrayView2<'a, f32>,
    ) -> &mut Self {
        if inputs.shape()[0] != expected_outputs.shape()[0] {
            panic!("testing data inputs and expected outputs must have the same number items");
        }

        self.testing_inputs = Some(inputs);
        self.testing_expected_outputs = Some(expected_outputs);
        self
    }

    pub fn logger(&mut self, logger: Box<dyn Logger>) -> &mut Self {
        self.logger = Some(logger);
        self
    }

    pub fn parallel(&mut self) -> &mut Self {
        self.parallel = Some(true);
        self
    }

    pub fn hot_outputs(&mut self, hot_outputs: usize) -> &mut Self {
        self.hot_outputs = Some(hot_outputs);
        self
    }

    pub fn draft(&self) -> Self {
        self.clone()
    }

    pub fn build(&self) -> TrainingConfig<'a> {
        // Epochs

        if self.max_epochs == Some(0) {
            panic!("maximum epochs must be greater than 0");
        }

        if self.min_epochs == Some(0) {
            panic!("minimum epochs must be greater than 0");
        }

        if self.max_epochs < self.min_epochs {
            panic!("maximum epochs must be greater than or equal to minimum epochs");
        }

        // Error goal

        if let Some(error_goal) = self.error_goal {
            if error_goal <= 0.0 {
                panic!("error goal must be greater than 0");
            }
        }

        // Accuracy goal

        if self.accuracy_goal.is_some() && self.testing_inputs.is_none() {
            panic!("if accuracy goal is set, testing data must also be set");
        }

        let is_accuracy_goal_valid =
            self.accuracy_goal <= Some(1.0) && self.accuracy_goal >= Some(0.0);

        if self.accuracy_goal.is_some() && !is_accuracy_goal_valid {
            panic!("accuracy goal must be between 0.0 and 1.0");
        }

        // End conditions

        let any_goals_set = self.error_goal.is_some() || self.accuracy_goal.is_some();

        if !any_goals_set && self.max_epochs.is_none() {
            panic!("if maximum epochs aren't set, at least one goal must be set");
        }

        // Training data

        let training_inputs = self
            .training_inputs
            .expect("training data inputs must be set");

        let training_expected_outputs = self
            .training_expected_outputs
            .expect("training data expected outputs must be set");

        if training_inputs.shape()[0] != training_expected_outputs.shape()[0] {
            panic!("training data inputs and expected outputs must have the same number items");
        }

        // Testing data

        match (self.testing_inputs, self.testing_expected_outputs) {
            (Some(testing_inputs), Some(testing_expected_outputs)) => {
                if testing_inputs.shape()[0] != testing_expected_outputs.shape()[0] {
                    panic!(
                        "testing data inputs and expected outputs must have the same number items"
                    );
                }
            }
            (Some(_), None) => panic!(
                "if testing data inputs are set, testing data expected outputs must also be set"
            ),
            (None, Some(_)) => panic!(
                "if testing data expected outputs are set, testing data inputs must also be set"
            ),
            (None, None) => {}
        }

        // Batch size

        if let Some(batch_size) = self.batch_size {
            if batch_size == 0 {
                panic!("batch size must be greater than 0");
            }

            if batch_size > training_inputs.shape()[0] {
                panic!("batch size must be less than or equal to the number of training items");
            }
        }

        // Training data

        let training_data = TrainingData::<'a> {
            inputs: training_inputs,
            expected_outputs: training_expected_outputs,
        };

        // Hot outputs

        let hot_outputs = self.hot_outputs.unwrap_or(1).to_owned();

        let amount_of_ones = training_expected_outputs
            .outer_iter()
            .next()
            .unwrap()
            .iter()
            .filter(|&&x| x == 1.0)
            .count();

        if amount_of_ones > hot_outputs {
            println!("WARNING: configured amount of hot outputs is seemingly less than the amount of hot outputs in the training data");
        }

        // Testing config

        let testing_config = match (self.testing_inputs, self.testing_expected_outputs) {
            (Some(testing_inputs), Some(testing_expected_outputs)) => Some(
                TestingConfigBuilder::new()
                    .testing_data(testing_inputs, testing_expected_outputs)
                    .hot_outputs(hot_outputs)
                    .build(),
            ),
            (Some(_), None) => panic!(
                "if testing data inputs are set, testing data expected outputs must also be set"
            ),
            (None, Some(_)) => panic!(
                "if testing data expected outputs are set, testing data inputs must also be set"
            ),
            (None, None) => None,
        };

        // Logger

        let logger = self.logger.to_owned().unwrap_or(Box::new(EmptyLogger {}));

        // Learning rates

        let learning_rates = self
            .learning_rates
            .to_owned()
            .expect("learning rate(s) must be set");
        if learning_rates.len() == 0 {
            panic!("learning rate(s) must be set");
        }

        // Training config

        TrainingConfig {
            min_epochs: self.min_epochs,
            max_epochs: self.max_epochs,
            error_goal: self.error_goal,
            accuracy_goal: self.accuracy_goal,
            learning_rates,
            batch_size: self.batch_size,
            training_data,
            testing_config,
            logger,
            parallel: self.parallel.unwrap_or(false),
        }
    }
}
