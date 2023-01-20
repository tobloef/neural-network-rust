mod elu;
mod leaky_relu;
mod relu;
mod sigmoid;
mod softmax;
mod swish;
mod tanh;

use super::{Layer, LayerType};
pub use elu::*;
pub use leaky_relu::*;
use rand::RngCore;
pub use relu::*;
pub use sigmoid::*;
pub use softmax::*;
pub use swish::*;
pub use tanh::*;

pub fn create_layer(layer_type: LayerType, rng: &mut Box<dyn RngCore>) -> Layer {
    match layer_type {
        LayerType::Tanh(input_size, output_size) => tanh(input_size, output_size, rng),
        LayerType::Sigmoid(input_size, output_size) => sigmoid(input_size, output_size, rng),
        LayerType::Softmax(input_size, output_size) => softmax(input_size, output_size, rng),
        LayerType::Relu(input_size, output_size) => relu(input_size, output_size, rng),
        LayerType::LeakyRelu(input_size, output_size) => leaky_relu(input_size, output_size, rng),
        LayerType::Elu(input_size, output_size) => elu(input_size, output_size, rng),
        LayerType::Swish(input_size, output_size) => swish(input_size, output_size, rng),
    }
}
