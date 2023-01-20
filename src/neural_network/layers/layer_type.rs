use std::fmt::{self, Debug, Display};

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum LayerType {
    // input size, output size
    Tanh(usize, usize),
    Sigmoid(usize, usize),
    Softmax(usize, usize),
    Relu(usize, usize),
    LeakyRelu(usize, usize),
    Elu(usize, usize),
    Swish(usize, usize),
}

impl Display for LayerType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Debug::fmt(self, f)
    }
}

pub fn get_layer_sizes(layer_type: &LayerType) -> (usize, usize) {
    match layer_type {
        LayerType::Tanh(i, o)
        | LayerType::Sigmoid(i, o)
        | LayerType::Softmax(i, o)
        | LayerType::Relu(i, o)
        | LayerType::LeakyRelu(i, o)
        | LayerType::Elu(i, o)
        | LayerType::Swish(i, o) => (*i, *o),
    }
}
