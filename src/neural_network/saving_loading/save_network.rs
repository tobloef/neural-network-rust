use super::{Save, SavedNetwork, SavingConfig};
use crate::{neural_network::Network, utils::join_with_slashes_if_needed};

impl Save for Network {
    fn save(&self, config: SavingConfig) {
        let saved_network_format: SavedNetwork = to_saved_network_format(&self, &config);

        let base_path = config.base_path.unwrap_or("".to_string());
        let path = join_with_slashes_if_needed(base_path.as_str(), &config.file_name);

        let json = serde_json::to_string(&saved_network_format).unwrap();

        std::fs::write(path, json).unwrap();
    }
}

fn to_saved_network_format(network: &Network, config: &SavingConfig) -> SavedNetwork {
    let weights = network.layers.iter().map(|l| l.weights.clone()).collect();
    let biases = network.layers.iter().map(|l| l.biases.clone()).collect();

    SavedNetwork {
        version: 1,
        layers: network.layer_types.to_owned(),
        seed: network.seed.to_owned(),
        error_function_type: network.error_function_type.to_owned(),
        meta_data: config.meta_data.to_owned(),
        weights,
        biases,
    }
}
