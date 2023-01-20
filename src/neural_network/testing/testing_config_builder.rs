use super::TestingConfig;
use ndarray::ArrayView2;

#[derive(Clone)]
pub struct TestingConfigBuilder<'a> {
    inputs: Option<ArrayView2<'a, f32>>,
    expected_outputs: Option<ArrayView2<'a, f32>>,
    hot_outputs: Option<usize>,
}

impl<'a> TestingConfigBuilder<'a> {
    pub fn new() -> Self {
        Self {
            inputs: None,
            expected_outputs: None,
            hot_outputs: None,
        }
    }

    pub fn testing_data(
        &mut self,
        inputs: ArrayView2<'a, f32>,
        expected_outputs: ArrayView2<'a, f32>,
    ) -> &mut Self {
        if inputs.shape()[0] != expected_outputs.shape()[0] {
            panic!("training data inputs and expected outputs must have the same number items");
        }

        self.inputs = Some(inputs);
        self.expected_outputs = Some(expected_outputs);
        self
    }

    pub fn hot_outputs(&mut self, hot_outputs: usize) -> &mut Self {
        self.hot_outputs = Some(hot_outputs);
        self
    }

    pub fn draft(&self) -> Self {
        self.clone()
    }

    pub fn build(&self) -> TestingConfig<'a> {
        let inputs = self.inputs.expect("training data inputs must be set");

        let expected_outputs = self
            .expected_outputs
            .expect("training data expected outputs must be set");

        if inputs.shape()[0] != expected_outputs.shape()[0] {
            panic!("training data inputs and expected outputs must have the same number items");
        }

        let hot_outputs = self.hot_outputs.unwrap_or(1);

        let amount_of_ones = expected_outputs
            .outer_iter()
            .next()
            .unwrap()
            .iter()
            .filter(|&&x| x == 1.0)
            .count();

        if amount_of_ones > hot_outputs {
            println!("WARNING: configured amount of hot outputs is seemingly less than the amount of hot outputs in the testing data");
        }

        TestingConfig {
            inputs,
            expected_outputs,
            hot_outputs,
        }
    }
}
