use super::TrainingConfigBuilder;
use crate::neural_network::{Logger, TestingConfig};
use ndarray::ArrayView2;

#[derive(Clone)]
pub struct TrainingData<'a> {
    pub inputs: ArrayView2<'a, f32>,
    pub expected_outputs: ArrayView2<'a, f32>,
}

#[derive(Clone)]
pub struct TrainingConfig<'a> {
    pub min_epochs: Option<usize>,
    pub max_epochs: Option<usize>,
    pub error_goal: Option<f32>,
    pub accuracy_goal: Option<f32>,
    pub learning_rates: Vec<(usize, f32)>,
    pub batch_size: Option<usize>,
    pub training_data: TrainingData<'a>,
    pub testing_config: Option<TestingConfig<'a>>,
    pub logger: Box<dyn Logger>,
    pub parallel: bool,
}

impl<'a> TrainingConfig<'a> {
    pub fn builder() -> TrainingConfigBuilder<'a> {
        TrainingConfigBuilder::new()
    }
}
