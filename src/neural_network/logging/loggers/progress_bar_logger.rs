use std::fmt::Write;

use crate::{
    neural_network::{
        training::{BatchResult, EpochResult},
        Logger, Network, TrainingConfig, TrainingResult,
    },
    utils::{get_config_string, get_result_string, hhmmss},
};
use indicatif::{MultiProgress, ProgressBar, ProgressState, ProgressStyle};

#[derive(Clone)]
pub struct ProgressBarLogger {
    network_bar: ProgressBar,
    epoch_bar: ProgressBar,
    log_epoch: bool,
}

pub enum LogLevel {
    Network,
    Epoch,
    Batch,
}

impl ProgressBarLogger {
    pub fn new(log_level: LogLevel) -> Self {
        let multi_progress = MultiProgress::new();

        let network_bar_style = ProgressStyle::with_template(
            "Network: [{elapsed_precise}] ({eta}) {pos:>4}/{len:4} {bar:30.blue} {msg:20}",
        )
        .unwrap()
        .with_key("eta", |state: &ProgressState, w: &mut dyn Write| {
            write!(w, "{}", hhmmss(state.eta())).unwrap()
        })
        .progress_chars("█░░");

        let epoch_bar_style = ProgressStyle::with_template(
            "Epoch:   [{elapsed_precise}] ({eta}) {pos:>4}/{len:4} {bar:30.cyan} {msg:20}",
        )
        .unwrap()
        .with_key("eta", |state: &ProgressState, w: &mut dyn Write| {
            write!(w, "{}", hhmmss(state.eta())).unwrap()
        })
        .progress_chars("█░░");

        let batch_bar_style = ProgressStyle::with_template(
            "Batch:   [{elapsed_precise}] ({eta}) {pos:>4}/{len:4} {bar:30.green} {msg:20}",
        )
        .unwrap()
        .with_key("eta", |state: &ProgressState, w: &mut dyn Write| {
            write!(w, "{}", hhmmss(state.eta())).unwrap()
        })
        .progress_chars("█░░");

        let network_bar = multi_progress.add(ProgressBar::new(0));
        network_bar.enable_steady_tick(std::time::Duration::from_millis(100));
        network_bar.set_style(network_bar_style);

        let epoch_bar = multi_progress.insert_after(&network_bar, ProgressBar::new(0));
        network_bar.enable_steady_tick(std::time::Duration::from_millis(100));
        epoch_bar.set_style(epoch_bar_style);

        let batch_bar = multi_progress.insert_after(&epoch_bar, ProgressBar::new(0));
        network_bar.enable_steady_tick(std::time::Duration::from_millis(100));
        batch_bar.set_style(batch_bar_style);

        let log_epoch = match log_level {
            LogLevel::Network => false,
            LogLevel::Epoch => true,
            LogLevel::Batch => true,
        };

        let log_batch = match log_level {
            LogLevel::Network => false,
            LogLevel::Epoch => false,
            LogLevel::Batch => true,
        };

        Self {
            network_bar,
            epoch_bar,
            log_epoch,
        }
    }
}

impl Logger for ProgressBarLogger {
    fn log_network_start(&self, config: &TrainingConfig, network: &Network) -> () {
        let network_length = config.max_epochs.unwrap_or(0);

        self.network_bar.reset();
        self.network_bar.set_length(network_length as u64);
        self.network_bar.set_position(0);

        self.network_bar.println(get_config_string(config, network));
    }

    fn log_network_end(&self, result: &TrainingResult) -> () {
        self.network_bar.println(get_result_string(result));

        self.network_bar.finish_and_clear();
    }

    fn log_epoch_start(&self, config: &TrainingConfig, epoch: usize) {
        if config.batch_size.is_none() {
            return;
        }

        if config.max_epochs.is_none() {
            self.network_bar.set_length(epoch as u64);
        }

        let data_set_size = config.training_data.inputs.dim().0;
        let batch_count = data_set_size as f32 / config.batch_size.unwrap() as f32;

        self.network_bar.set_position(epoch as u64);

        if self.log_epoch {
            self.epoch_bar.reset();
            self.epoch_bar.set_length(batch_count as u64);
        }
    }

    fn log_epoch_end(&self, result: &EpochResult) {
        let error_string = format!("E: {:.3}", result.error);
        let accuracy_strig = result
            .accuracy
            .map(|x| format!(" | A: {:.2}%", x * 100.0))
            .unwrap_or("".to_string());

        self.network_bar
            .set_message(format!("{}{}", error_string, accuracy_strig));

        if self.log_epoch {
            self.epoch_bar.finish_and_clear();
        }
    }

    fn log_batch_start(&self, config: &TrainingConfig, batch: usize, batch_size: usize) {
        if self.log_epoch {
            self.epoch_bar.set_position(batch as u64 + 1);
        }
    }

    fn log_batch_end(&self, result: &BatchResult) {
        if self.log_epoch {
            self.epoch_bar
                .set_message(format!("E: {:.3}", result.error));
        }
    }
}
