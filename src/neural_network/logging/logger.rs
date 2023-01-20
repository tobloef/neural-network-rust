use dyn_clone::{clone_trait_object, DynClone};

use crate::neural_network::{
    training::{BatchResult, EpochResult},
    Network, TrainingConfig, TrainingResult,
};

pub trait Logger: DynClone {
    fn log_network_start(&self, config: &TrainingConfig, network: &Network);
    fn log_network_end(&self, result: &TrainingResult);
    fn log_epoch_start(&self, config: &TrainingConfig, epoch: usize);
    fn log_epoch_end(&self, result: &EpochResult);
    fn log_batch_start(&self, config: &TrainingConfig, batch: usize, batch_size: usize);
    //fn log_batch_progress(&self, index: usize, error: f32);
    fn log_batch_end(&self, result: &BatchResult);
}

clone_trait_object!(Logger);
