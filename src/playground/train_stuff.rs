use crate::{
    neural_network::{
        ErrorFunctionType::*, GraphLogger, LayerType::*, Load, LoadingConfigBuilder, Network,
        NetworkConfigBuilder, Save, SavedMetaData, SavingConfigBuilder, Train, TrainingConfig,
        TrainingConfigBuilder,
    },
    utils::{
        create_random_seed, generate_multi_mnist, load_mnist_data, network_to_filename, to_percent,
        MnistData,
    },
};

pub fn train_stuff() {
    // Data preparation

    let mnist_data: MnistData = load_mnist_data(
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
    );

    let seed = create_random_seed();
    //let multimnist_data = generate_multi_mnist(&mnist_data, 8, seed.as_str());

    let data = mnist_data;

    let input_size = data.training_inputs.shape()[1];
    let output_size = data.training_expected_outputs.shape()[1];

    // Network and training

    let mut mnist_network = Network::new(
        NetworkConfigBuilder::new()
            .layers(vec![
                Relu(input_size, 512),
                Relu(512, 512),
                Sigmoid(512, output_size),
            ])
            .error_function(BinaryCrossentropy)
            .seed("seed")
            .build(),
    );

    let mut mnist_training_config: TrainingConfig = TrainingConfigBuilder::new()
        .training_data(
            data.training_inputs.view(),
            data.training_expected_outputs.view(),
        )
        .testing_data(
            data.testing_inputs.view(),
            data.testing_expected_outputs.view(),
        )
        .max_epochs(1)
        .learning_rates(vec![(0, 0.1)])
        .logger(Box::new(GraphLogger::new()))
        .batch_size(128)
        .parallel()
        .hot_outputs(1)
        .build();

    let result = mnist_network.train(&mut mnist_training_config);

    println!(
        "Final Accuracy: {}. Best Accuracy: {}",
        result.accuracy.map(to_percent).unwrap_or("N/A".to_string()),
        result
            .best_accuracy
            .map(to_percent)
            .unwrap_or("N/A".to_string()),
    );

    // Saving the network

    if result.accuracy.is_none() {
        return;
    }

    let file_name = format!(
        "networks/cnn/{}",
        network_to_filename(&mnist_network, &result)
    );

    let save_config = SavingConfigBuilder::new()
        .file_name(file_name.as_str())
        .meta_data(SavedMetaData {
            accuracy: result.accuracy,
            duration: Some(result.duration),
            epochs: Some(result.epochs),
            error: Some(result.error),
        })
        .build();

    mnist_network.save(save_config);

    println!("Saved network as \"{file_name}\"");
}
