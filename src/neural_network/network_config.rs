use super::layers::LayerType;
use super::{ErrorFunctionType, NetworkConfigBuilder};

#[derive(Clone)]
pub struct NetworkConfig {
    pub seed: String,
    pub layers: Vec<LayerType>,
    pub error_function_type: ErrorFunctionType,
}

impl NetworkConfig {
    pub fn builder() -> NetworkConfigBuilder {
        NetworkConfigBuilder::new()
    }
}
