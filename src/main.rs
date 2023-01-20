#![allow(dead_code, unused_variables, unused_imports, unreachable_code)]

mod neural_network;
mod playground;
mod utils;

use crate::{
    neural_network::{
        EmptyLogger, ErrorFunctionType, LayerType, Save, SavedMetaData, SavingConfigBuilder,
    },
    utils::{create_random_seed, network_to_filename, save_input_as_image, value_or_string},
};
use ndarray::{s, Array, Array2, ArrayView2};
use neural_network::{
    EnsembleNetwork, ErrorFunctionType::*, GraphLogger, LayerType::*, Load, LoadingConfigBuilder,
    Network, NetworkConfigBuilder, Test, TestingConfig, TestingConfigBuilder, TestingResult, Train,
    TrainingConfig, TrainingConfigBuilder, TrainingData, Tuner,
};
use playground::train_stuff;
use rand::Rng;
use rayon::prelude::*;
use regex::Regex;
use std::{fs, ops::Mul, time::Instant};
use utils::{
    arrayview1_eqauls, generate_multi_mnist, load_mnist_data, n_hot_to_indexes, output_to_n_hot,
    to_percent, MnistData,
};

fn main() {
    // Set max parallel threads across the whole program
    //rayon::ThreadPoolBuilder::new().num_threads(4).build_global().unwrap();

    train_stuff();
}
