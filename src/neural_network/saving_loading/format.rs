use std::time::Duration;

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

use crate::neural_network::{ErrorFunctionType, LayerType};

#[derive(Serialize, Deserialize)]
pub struct SavedNetwork {
    pub version: u32,
    pub layers: Vec<LayerType>,
    pub seed: String,
    pub error_function_type: ErrorFunctionType,
    pub meta_data: Option<SavedMetaData>,
    pub weights: Vec<Array2<f32>>,
    pub biases: Vec<Array1<f32>>,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct SavedMetaData {
    pub accuracy: Option<f32>,
    pub error: Option<f32>,
    pub epochs: Option<usize>,
    pub duration: Option<Duration>,
}
