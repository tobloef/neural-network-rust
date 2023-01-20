use network_config_builder::NetworkConfigBuilder;

use super::{Load, LoadingConfig, SavedMetaData, SavedNetwork};
use crate::{
    neural_network::{network_config_builder, Network},
    utils::join_with_slashes_if_needed,
};

impl Load for Network {
    fn load(config: LoadingConfig) -> (Self, Option<SavedMetaData>) {
        let base_path = config.base_path.unwrap_or("".to_string());
        let path = join_with_slashes_if_needed(base_path.as_str(), &config.file_name);

        let json = std::fs::read_to_string(path).unwrap();

        let saved_network_format: SavedNetwork = serde_json::from_str(&json).unwrap();

        let network_config_builder = NetworkConfigBuilder::new()
            //.seed(saved_network_format.seed.as_str())
            .error_function(saved_network_format.error_function_type)
            .layers(saved_network_format.layers)
            .build();

        let mut network = Network::new(network_config_builder);

        for (i, layer) in network.layers.iter_mut().enumerate() {
            layer.weights = saved_network_format.weights[i].clone();
            layer.biases = saved_network_format.biases[i].clone();
        }

        (network, saved_network_format.meta_data)
    }
}
