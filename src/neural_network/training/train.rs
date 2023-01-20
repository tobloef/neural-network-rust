use crate::neural_network::Network;

use super::TrainingConfig;
use std::time::Duration;

pub struct TrainingResult {
    pub error: f32,
    pub accuracy: Option<f32>,
    pub epochs: usize,
    pub duration: Duration,
    pub best_network: Option<Network>,
    pub best_accuracy: Option<f32>,
}

pub trait Train {
    fn train(&mut self, config: &TrainingConfig) -> TrainingResult;
}
