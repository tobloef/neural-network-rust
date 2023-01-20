use std::{cell::RefCell, fmt::Write};

use crate::{
    neural_network::{
        training::{self, BatchResult, EpochResult},
        Logger, Network, TrainingConfig, TrainingResult,
    },
    utils::{get_accuracy_plot, get_config_string, get_result_string, hhmmss, to_percent},
};
use indicatif::{ProgressBar, ProgressState, ProgressStyle};

pub struct GraphLogger {
    epoch_results: RefCell<Vec<EpochResult>>,
    progress_bar: ProgressBar,
}

impl GraphLogger {
    pub fn new() -> Self {
        let progress_bar = ProgressBar::new(0);
        progress_bar.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [{elapsed_precise}] ({eta}) Epoch #{pos}\n{msg}",
            )
            .unwrap()
            .with_key("eta", |state: &ProgressState, w: &mut dyn Write| {
                write!(w, "{}", hhmmss(state.eta())).unwrap()
            }),
        );
        progress_bar.enable_steady_tick(std::time::Duration::from_millis(100));

        Self {
            epoch_results: RefCell::new(Vec::new()),
            progress_bar,
        }
    }
}

impl Clone for GraphLogger {
    fn clone(&self) -> Self {
        Self::new()
    }
}

impl Logger for GraphLogger {
    fn log_network_start(&self, config: &TrainingConfig, network: &Network) {
        self.epoch_results.borrow_mut().clear();

        let epochs = config.max_epochs.unwrap_or(0);

        self.progress_bar.reset();
        self.progress_bar.set_length(epochs as u64);
        self.progress_bar.set_position(0);

        self.progress_bar
            .println(get_config_string(config, network));
    }

    fn log_network_end(&self, result: &TrainingResult) {
        self.progress_bar
            .println(get_accuracy_plot(&self.epoch_results.borrow()));

        self.progress_bar.println(get_result_string(result));

        self.progress_bar.finish_and_clear();
    }

    fn log_epoch_start(&self, config: &TrainingConfig, epoch: usize) {
        self.progress_bar.set_position(epoch as u64 + 1);
    }

    fn log_epoch_end(&self, result: &training::EpochResult) {
        self.epoch_results.borrow_mut().push(result.clone());

        let mut results: Vec<EpochResult> = self.epoch_results.borrow().clone();

        if results.len() == 1 {
            let a = results[0].clone();
            let mut b = results[0].clone();
            b.epoch = 1;

            results = vec![a, b];
        }

        let plot = get_accuracy_plot(&results);
        self.progress_bar.set_message(format!(
            "Current accuracy: {}. Accuracy over epochs:\n{}",
            result.accuracy.map(to_percent).unwrap_or("N/A".to_string()),
            plot
        ));
    }

    fn log_batch_start(&self, config: &TrainingConfig, batch: usize, batch_size: usize) {}

    fn log_batch_end(&self, result: &BatchResult) {}
}
