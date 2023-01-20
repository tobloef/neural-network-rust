use crate::utils::seeded_rng;

use super::{
    error::{
        binary_cross_entropy_error, binary_cross_entropy_error_derivative,
        categorical_cross_entropy_error, categorical_cross_entropy_error_derivative,
        mean_squared_error, mean_squared_error_derivative,
    },
    layers::{create_layer, Layer},
    network_config::NetworkConfig,
    ErrorFunctionType, LayerType,
};
use ndarray::{Array1, ArrayView1};
use rand::RngCore;

pub struct Network {
    pub layers: Vec<Layer>,
    pub layer_types: Vec<LayerType>,
    pub seed: String,
    pub rng: Box<dyn RngCore>,
    pub error_function: fn(ArrayView1<f32>, ArrayView1<f32>) -> f32,
    pub error_function_derivative: fn(ArrayView1<f32>, ArrayView1<f32>) -> Array1<f32>,
    pub error_function_type: ErrorFunctionType,
}

impl Network {
    pub fn new(config: NetworkConfig) -> Self {
        let mut rng = seeded_rng(&config.seed);

        let layers = config_layers_to_network_layers(config.layers.to_owned(), &mut rng);

        let ErrorFunctions {
            error_function,
            error_function_derivative,
        } = config_error_function_to_network_error_functions(config.error_function_type);

        Network {
            rng,
            seed: config.seed,
            layers,
            layer_types: config.layers,
            error_function,
            error_function_derivative,
            error_function_type: config.error_function_type,
        }
    }
}

impl Clone for Network {
    fn clone(&self) -> Self {
        Network {
            layers: self.layers.clone(),
            layer_types: self.layer_types.clone(),
            seed: self.seed.clone(),
            rng: seeded_rng(&self.seed),
            error_function: self.error_function.clone(),
            error_function_derivative: self.error_function_derivative.clone(),
            error_function_type: self.error_function_type.clone(),
        }
    }
}

fn config_layers_to_network_layers(
    config_layers: Vec<LayerType>,
    rng: &mut Box<dyn RngCore>,
) -> Vec<Layer> {
    config_layers
        .into_iter()
        .map(|layer_type| create_layer(layer_type, rng))
        .collect()
}

struct ErrorFunctions {
    error_function: fn(ArrayView1<f32>, ArrayView1<f32>) -> f32,
    error_function_derivative: fn(ArrayView1<f32>, ArrayView1<f32>) -> Array1<f32>,
}

fn config_error_function_to_network_error_functions(
    error_function_type: ErrorFunctionType,
) -> ErrorFunctions {
    let error_function = match error_function_type {
        ErrorFunctionType::MeanSquared => mean_squared_error,
        ErrorFunctionType::CategoricalCrossentropy => categorical_cross_entropy_error,
        ErrorFunctionType::BinaryCrossentropy => binary_cross_entropy_error,
    };

    let error_function_derivative = match error_function_type {
        ErrorFunctionType::MeanSquared => mean_squared_error_derivative,
        ErrorFunctionType::CategoricalCrossentropy => categorical_cross_entropy_error_derivative,
        ErrorFunctionType::BinaryCrossentropy => binary_cross_entropy_error_derivative,
    };

    ErrorFunctions {
        error_function,
        error_function_derivative,
    }
}
