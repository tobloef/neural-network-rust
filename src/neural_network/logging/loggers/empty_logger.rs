use crate::neural_network::{
    training::{self, BatchResult},
    Logger, Network, TrainingConfig, TrainingResult,
};

#[derive(Clone)]
pub struct EmptyLogger {}

impl EmptyLogger {
    pub fn new() -> Self {
        EmptyLogger {}
    }
}

impl Logger for EmptyLogger {
    fn log_network_start(&self, config: &TrainingConfig, network: &Network) {}

    fn log_network_end(&self, result: &TrainingResult) {}

    fn log_epoch_start(&self, config: &TrainingConfig, epoch: usize) {}

    fn log_epoch_end(&self, result: &training::EpochResult) {}

    fn log_batch_start(&self, config: &TrainingConfig, batch: usize, batch_size: usize) {}

    fn log_batch_end(&self, result: &BatchResult) {}
}
