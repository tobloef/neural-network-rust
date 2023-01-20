use crate::{
    neural_network::{
        training::{BatchResult, EpochResult},
        Logger, Network, TrainingConfig, TrainingResult,
    },
    utils::{min_max_string, println_if_some, to_percent},
};

#[derive(Clone)]
pub struct SimpleLogger {
    pub log_epochs: bool,
    pub log_batches: bool,
}

impl Logger for SimpleLogger {
    fn log_network_start(&self, config: &TrainingConfig, network: &Network) {
        let testing_data_size = config
            .testing_config
            .as_ref()
            .map(|data| data.inputs.shape()[0]);

        println!("Network Training");
        println!("\tConfiguration");
        println!(
            "\t\tTraining set size: {}",
            config.training_data.inputs.shape()[0]
        );
        println_if_some("\t\tTesting set size: ", testing_data_size);
        println!(
            "\t\tEpochs: {}",
            min_max_string(config.min_epochs, config.max_epochs)
        );
        println_if_some("\t\tBatch size: ", config.batch_size);
        println_if_some("\t\tError goal: ", config.error_goal);
        println_if_some("\t\tAccuracy goal: ", config.accuracy_goal.map(to_percent));
        println!("\t\tLearning rate(s): {:?}", config.learning_rates);
    }

    fn log_network_end(&self, result: &TrainingResult) {
        println!("\tResult");
        println!("\t\tEpochs: {}", result.epochs);
        println!("\t\tError: {}", result.error);
        println_if_some("\t\tAccuracy: ", result.accuracy.map(to_percent));
        println!("\t\tDuration: {}s", result.duration.as_secs());
    }

    fn log_epoch_start(&self, config: &TrainingConfig, epoch: usize) {
        if !self.log_epochs {
            return;
        }

        println!("\tEpoch #{}", epoch);
    }

    fn log_epoch_end(&self, result: &EpochResult) {
        if !self.log_epochs {
            return;
        }

        println!("\t\tResult");
        println!("\t\t\tError: {}", result.error);
        println_if_some("\t\t\tAccuracy: ", result.accuracy.map(to_percent));
        println!("\t\t\tDuration: {}s", result.duration.as_secs());
    }

    fn log_batch_start(&self, config: &TrainingConfig, batch: usize, batch_size: usize) {
        if !self.log_batches {
            return;
        }

        let training_data_size = config.training_data.inputs.shape()[0];

        let batches_count = config
            .batch_size
            .map(|batch_size| (training_data_size as f32 / batch_size as f32).ceil() as usize);

        let batches_count_str = batches_count
            .map(|batches_count| format!(" of {}", batches_count))
            .unwrap_or("".to_string());

        println!(
            "\t\tBatch #{}{} ({} items)",
            batch, batches_count_str, batch_size
        );
    }

    fn log_batch_end(&self, result: &BatchResult) {
        if !self.log_batches {
            return;
        }

        println!("\t\t\tResult");
        println!("\t\t\t\tError: {}", result.error);
        println!("\t\t\t\tDuration: {}s", result.duration.as_secs());
    }
}
