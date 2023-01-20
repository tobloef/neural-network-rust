use crate::neural_network::{EpochResult, Network, TrainingConfig, TrainingResult};
use chrono::Local;
use std::{fmt::Display, time::Duration};
use textplots::{Chart, Plot, Shape};

pub fn value_or_string(value: Option<impl Display>, string: &str) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or(string.to_string())
}

pub fn to_percent(value: f32) -> String {
    format!("{:.2}%", value * 100.0)
}

pub fn println_if_some(prefix: &str, value: Option<impl Display>) {
    if let Some(value) = value {
        println!("{}{}", prefix, value);
    }
}

pub fn min_max_string(min: Option<usize>, max: Option<usize>) -> String {
    match (min, max) {
        (Some(min), Some(max)) => {
            if max == min {
                format!("{}", max)
            } else {
                format!("{} - {}", min, max)
            }
        }
        (None, Some(max)) => format!("{}", max),
        (Some(min), None) => format!("{} - ∞", min),
        (None, None) => "∞".to_string(),
    }
}

pub fn hhmmss(duration: Duration) -> String {
    let seconds = duration.as_secs();
    let hours = seconds / 3600;
    let minutes = (seconds % 3600) / 60;
    let seconds = seconds % 60;

    format!("{:02}:{:02}:{:02}", hours, minutes, seconds)
}

pub fn get_config_string(config: &TrainingConfig, network: &Network) -> String {
    let mut string = "".to_string();

    let training_data_size = config.training_data.inputs.dim().0;
    let testing_data_size = config.testing_config.to_owned().map(|c| c.inputs.dim().0);

    string.push_str(&"Training started with the following configuration:\n");
    string.push_str(&format!("  - Layers:\n"));
    for layer_type in network.layer_types.iter() {
        string.push_str(&format!("    - {}\n", layer_type));
    }
    string.push_str(&format!("  - Training data size: {}\n", training_data_size));
    string.push_str(&format!(
        "  - Testing data size: {}\n",
        value_or_string(testing_data_size, "None")
    ));
    string.push_str(&format!("  - Learning rate: {:?}\n", config.learning_rates));
    string.push_str(&format!(
        "  - Min epochs: {}\n",
        value_or_string(config.min_epochs, "None")
    ));
    string.push_str(&format!(
        "  - Max epochs: {}\n",
        value_or_string(config.max_epochs, "None")
    ));
    string.push_str(&format!(
        "  - Batch size: {}\n",
        value_or_string(config.batch_size, "None")
    ));
    // Network's error function
    string.push_str(&format!(
        "  - Error function: {}\n",
        network.error_function_type
    ));
    string.push_str(&format!(
        "  - Error goal: {}\n",
        value_or_string(config.error_goal, "None")
    ));
    string.push_str(&format!(
        "  - Accuracy goal: {}\n",
        value_or_string(
            config
                .accuracy_goal
                .map(|x| x * 100.0)
                .map(|x| x.to_string() + "%"),
            "N/A"
        )
    ));
    string.push_str(&format!("  - Seed: {}\n", network.seed));

    string
}

pub fn get_result_string(result: &TrainingResult) -> String {
    let mut string = "".to_string();

    string.push_str("Training finished with the following results:\n");
    string.push_str(&format!("  - Epochs: {}\n", result.epochs));
    string.push_str(&format!("  - Error: {}\n", result.error));
    string.push_str(&format!(
        "  - Final Accuracy: {}\n",
        result.accuracy.map(to_percent).unwrap_or("N/A".to_string())
    ));
    string.push_str(&format!(
        "  - Best Accuracy: {}\n",
        result
            .best_accuracy
            .map(to_percent)
            .unwrap_or("N/A".to_string())
    ));
    string.push_str(&format!("  - Duration: {}\n", hhmmss(result.duration)));
    string.push_str("\n");

    string
}

pub fn get_accuracy_plot(epoch_results: &Vec<EpochResult>) -> String {
    let epochs = epoch_results.len() as f32;

    let points = epoch_results
        .iter()
        .map(|x| (x.epoch as f32 + 1.0, x.accuracy.unwrap_or(0.0) * 100.0))
        .collect::<Vec<(f32, f32)>>();

    let min_y = points
        .iter()
        .map(|p| p.1)
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let max_y = points
        .iter()
        .map(|p| p.1)
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    let plot = Chart::new_with_y_range(200, 50, 1.0, epochs, min_y - 0.2, max_y + 0.2)
        .lineplot(&Shape::Lines(&points))
        .to_string();

    plot
}

pub fn join_with_slashes_if_needed(a: &str, b: &str) -> String {
    if a.ends_with("/") || b.starts_with("/") {
        format!("{}{}", a, b)
    } else {
        format!("{}/{}", a, b)
    }
}

pub fn network_to_filename(network: &Network, result: &TrainingResult) -> String {
    let date_time_string = &Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();

    let accuracy = result.accuracy.unwrap();
    let accuracy_percent = accuracy * 100.0;
    let accuracy_decimals_only = (accuracy * 100.0) % 1.0;
    let accuracy_string = format!(
        "{:0>3}.{:.0}",
        accuracy_percent as u32,
        accuracy_decimals_only * 100.0
    );

    let layers_string = network
        .layer_types
        .iter()
        .map(|layer| layer.to_string())
        .collect::<Vec<String>>()
        .join("-");

    let seed = &network.seed;

    format!("{accuracy_string}_{date_time_string}_{layers_string}_{seed}")
}
