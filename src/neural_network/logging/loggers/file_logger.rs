use std::{cell::RefCell, fs, path::Path};

use training::EpochResult;

use crate::{
    neural_network::{
        training::{self, BatchResult},
        Logger, Network, TrainingConfig, TrainingResult,
    },
    utils::{get_accuracy_plot, get_config_string, get_result_string},
};

#[derive(Clone)]
pub struct FileLogger {
    epoch_results: RefCell<Vec<EpochResult>>,
    string: RefCell<String>,
    path: String,
}

impl FileLogger {
    pub fn new(path: &str) -> Self {
        Self {
            epoch_results: RefCell::new(Vec::new()),
            string: RefCell::new(String::new()),
            path: path.to_string(),
        }
    }
}

impl Logger for FileLogger {
    fn log_network_start(&self, config: &TrainingConfig, network: &Network) {
        let config_string = get_config_string(config, network);

        self.string.borrow_mut().push_str(&config_string);
    }

    fn log_network_end(&self, result: &TrainingResult) {
        let accuracy_plot = get_accuracy_plot(&self.epoch_results.borrow());
        let result_string = get_result_string(result);

        self.string.borrow_mut().push_str("\n\n");
        self.string.borrow_mut().push_str(&accuracy_plot);
        self.string.borrow_mut().push_str("\n\n");
        self.string.borrow_mut().push_str(&result_string);

        if Path::new(&self.path).exists() {
            fs::remove_file(&self.path).unwrap();
        }
        fs::write(&self.path, self.string.borrow().as_str()).unwrap();
    }

    fn log_epoch_start(&self, config: &TrainingConfig, epoch: usize) {}

    fn log_epoch_end(&self, result: &training::EpochResult) {
        self.epoch_results.borrow_mut().push(result.clone());
        //println!("Epoch #{}", result.epoch);
    }

    fn log_batch_start(&self, config: &TrainingConfig, batch: usize, batch_size: usize) {}

    fn log_batch_end(&self, result: &BatchResult) {}
}
