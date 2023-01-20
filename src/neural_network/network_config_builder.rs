use crate::utils::create_random_seed;

use super::{layers::get_layer_sizes, ErrorFunctionType, LayerType, NetworkConfig};

#[derive(Clone)]
pub struct NetworkConfigBuilder {
    seed: Option<String>,
    layers: Option<Vec<LayerType>>,
    error_function_type: Option<ErrorFunctionType>,
}

impl<'a> NetworkConfigBuilder {
    pub fn new() -> Self {
        NetworkConfigBuilder {
            seed: None,
            layers: None,
            error_function_type: None,
        }
    }

    pub fn seed(mut self, seed: &str) -> Self {
        self.seed = Some(seed.to_string());
        self
    }

    pub fn layers(mut self, layers: Vec<LayerType>) -> Self {
        self.layers = Some(layers);
        self
    }

    pub fn error_function(mut self, error_function_type: ErrorFunctionType) -> Self {
        self.error_function_type = Some(error_function_type);
        self
    }

    pub fn draft(&self) -> NetworkConfigBuilder {
        self.clone()
    }

    pub fn build(self) -> NetworkConfig {
        let layers = self.layers.expect("layers must be set");

        // Check that that each layer has the same number of inputs as the previous layer has outputs
        for (index, (layer, next_layer)) in layers.iter().zip(layers.iter().skip(1)).enumerate() {
            let layer_outputs = get_layer_sizes(layer).1;
            let next_layer_inputs = get_layer_sizes(next_layer).0;

            if layer_outputs != next_layer_inputs {
                panic!(
                    "layer outputs ({}) must match next layer inputs ({}) at index {}",
                    layer_outputs, next_layer_inputs, index
                );
            }
        }

        NetworkConfig {
            seed: self.seed.unwrap_or_else(create_random_seed),
            layers,
            error_function_type: self
                .error_function_type
                .unwrap_or(ErrorFunctionType::MeanSquared),
        }
    }
}
