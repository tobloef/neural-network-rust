use bmp::{px, Image, Pixel};
use mnist::{Mnist, MnistBuilder};
use ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3};
use rand::Rng;

use super::{arrayview1_eqauls, indexes_to_n_hot, n_hot_to_indexes, seeded_rng};

pub const TRAINING_DATA_SIZE: usize = 60_000;
pub const TEST_DATA_SIZE: usize = 10_000;

pub const IMAGE_DIMENSIONS: [usize; 2] = [28, 28];
pub const INPUT_SIZE: usize = IMAGE_DIMENSIONS[0] * IMAGE_DIMENSIONS[1];
pub const OUTPUT_SIZE: usize = 10;

pub const MULTIMNIST_SHIFT_AMOUNT: usize = 4;
pub const MULTIMNIST_IMAGE_DIMENSIONS: [usize; 2] = [
    IMAGE_DIMENSIONS[0] + MULTIMNIST_SHIFT_AMOUNT * 2,
    IMAGE_DIMENSIONS[1] + MULTIMNIST_SHIFT_AMOUNT * 2,
];
pub const MULTIMNIST_INPUT_SIZE: usize =
    MULTIMNIST_IMAGE_DIMENSIONS[0] * MULTIMNIST_IMAGE_DIMENSIONS[1];
pub const MULTIMNIST_OUTPUT_SIZE: usize = OUTPUT_SIZE + 1;

pub struct MnistData {
    pub training_inputs: Array2<f32>,
    pub training_expected_outputs: Array2<f32>,
    pub testing_inputs: Array2<f32>,
    pub testing_expected_outputs: Array2<f32>,
}

pub fn load_mnist_data<'a>(
    training_images_path: &str,
    training_labels_path: &str,
    test_images_path: &str,
    test_labels_path: &str,
) -> MnistData {
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .base_path("data")
        .training_images_filename(training_images_path)
        .training_labels_filename(training_labels_path)
        .test_images_filename(test_images_path)
        .test_labels_filename(test_labels_path)
        .label_format_one_hot()
        .finalize();

    let training_inputs = Array1::from_vec(trn_img)
        .into_shape((TRAINING_DATA_SIZE, INPUT_SIZE))
        .unwrap()
        .mapv(|x| x as f32 / 255.0);

    let training_expected_outputs = Array1::from_vec(trn_lbl)
        .into_shape((TRAINING_DATA_SIZE, OUTPUT_SIZE))
        .unwrap()
        .mapv(|x| x as f32);

    let testing_inputs = Array1::from_vec(tst_img)
        .into_shape((TEST_DATA_SIZE, INPUT_SIZE))
        .unwrap()
        .mapv(|x| x as f32 / 255.0);

    let testing_expected_outputs = Array1::from_vec(tst_lbl)
        .into_shape((TEST_DATA_SIZE, OUTPUT_SIZE))
        .unwrap()
        .mapv(|x| x as f32);

    MnistData {
        training_inputs,
        training_expected_outputs,
        testing_inputs,
        testing_expected_outputs,
    }
}

pub fn generate_multi_mnist(mnist: &MnistData, amount_per_dataset: usize, seed: &str) -> MnistData {
    let training_multimnist = inputs_and_outputs_to_multimnist(
        mnist.training_inputs.view(),
        mnist.training_expected_outputs.view(),
        amount_per_dataset,
        format!("{seed}-training").as_str(),
    );

    let testing_multimnist = inputs_and_outputs_to_multimnist(
        mnist.testing_inputs.view(),
        mnist.testing_expected_outputs.view(),
        amount_per_dataset,
        format!("{seed}-testing").as_str(),
    );

    MnistData {
        training_inputs: training_multimnist.0,
        training_expected_outputs: training_multimnist.1,
        testing_inputs: testing_multimnist.0,
        testing_expected_outputs: testing_multimnist.1,
    }
}

fn inputs_and_outputs_to_multimnist(
    inputs: ArrayView2<f32>,
    expected_outputs: ArrayView2<f32>,
    amount_per_dataset: usize,
    seed: &str,
) -> (Array2<f32>, Array2<f32>) {
    let inputs_outputs_pairs = inputs.outer_iter().zip(expected_outputs.outer_iter());

    let images_labels_pairs = inputs_outputs_pairs.map(|(input, expected_output)| {
        let image = input_to_image(input, IMAGE_DIMENSIONS);
        let label = n_hot_to_indexes(expected_output, 1)[0];

        (image, label)
    });

    let mut multimnist_inputs = Array2::zeros((
        amount_per_dataset * inputs.shape()[0],
        MULTIMNIST_INPUT_SIZE,
    ));

    let mut multimnist_expected_outputs = Array2::zeros((
        amount_per_dataset * expected_outputs.shape()[0],
        MULTIMNIST_OUTPUT_SIZE,
    ));

    let mut rng = seeded_rng(seed);

    for (i, (image, label)) in images_labels_pairs.enumerate() {
        for j in 0..amount_per_dataset {
            // First image
            let first_insert_position = [
                rng.gen_range(0..(MULTIMNIST_SHIFT_AMOUNT * 2 + 1)) as usize,
                rng.gen_range(0..(MULTIMNIST_SHIFT_AMOUNT * 2 + 1)) as usize,
            ];

            let mut first_reszed_image = Array2::zeros((
                MULTIMNIST_IMAGE_DIMENSIONS[0],
                MULTIMNIST_IMAGE_DIMENSIONS[1],
            ));

            first_reszed_image
                .slice_mut(s![
                    first_insert_position[0]..first_insert_position[0] + IMAGE_DIMENSIONS[0],
                    first_insert_position[1]..first_insert_position[1] + IMAGE_DIMENSIONS[1]
                ])
                .assign(&image);

            // Second image
            let random_index = rng.gen_range(0..inputs.dim().0);
            let other_image = input_to_image(inputs.slice(s![random_index, ..]), IMAGE_DIMENSIONS);
            let other_label = n_hot_to_indexes(expected_outputs.slice(s![random_index, ..]), 1)[0];

            let second_insert_position = [
                rng.gen_range(0..(MULTIMNIST_SHIFT_AMOUNT * 2 + 1)) as usize,
                rng.gen_range(0..(MULTIMNIST_SHIFT_AMOUNT * 2 + 1)) as usize,
            ];

            let mut second_reszed_image = Array2::zeros((
                MULTIMNIST_IMAGE_DIMENSIONS[0],
                MULTIMNIST_IMAGE_DIMENSIONS[1],
            ));

            second_reszed_image
                .slice_mut(s![
                    second_insert_position[0]..second_insert_position[0] + IMAGE_DIMENSIONS[0],
                    second_insert_position[1]..second_insert_position[1] + IMAGE_DIMENSIONS[1]
                ])
                .assign(&other_image);

            // Combined

            let mut multimnist_image = first_reszed_image + second_reszed_image;
            multimnist_image.mapv_inplace(|x| if x > 1.0 { 1.0 } else { x });

            let multimnist_input = image_to_input(multimnist_image.view(), MULTIMNIST_INPUT_SIZE);

            let index = i * amount_per_dataset + j;

            multimnist_inputs
                .slice_mut(s![index, ..])
                .assign(&multimnist_input);

            let first_expected_output = indexes_to_n_hot(vec![label], MULTIMNIST_OUTPUT_SIZE);
            let second_expected_output =
                indexes_to_n_hot(vec![other_label], MULTIMNIST_OUTPUT_SIZE);

            let multimnist_output =
                if arrayview1_eqauls(first_expected_output.view(), second_expected_output.view()) {
                    let mut new_expected_output = first_expected_output.clone();
                    new_expected_output[MULTIMNIST_OUTPUT_SIZE - 1] = 1.0;
                    new_expected_output
                } else {
                    first_expected_output + second_expected_output
                };

            multimnist_expected_outputs
                .slice_mut(s![index, ..])
                .assign(&multimnist_output);
        }
    }

    (multimnist_inputs, multimnist_expected_outputs)
}

pub fn image_to_input(image: ArrayView2<f32>, input_size: usize) -> ArrayView1<f32> {
    image.into_shape(input_size).unwrap()
}

pub fn input_to_image(input: ArrayView1<f32>, image_dimensions: [usize; 2]) -> ArrayView2<f32> {
    input.into_shape(image_dimensions).unwrap()
}

pub fn save_input_as_image(input: ArrayView1<f32>, image_dimensions: [usize; 2], file_path: &str) {
    let image_data = input_to_image(input, image_dimensions);

    let width = image_data.shape()[0] as u32;
    let height = image_data.shape()[1] as u32;

    let mut image = Image::new(width, height);

    for (y, col) in image_data.outer_iter().enumerate() {
        for (x, value) in col.iter().enumerate() {
            let pixel = px!(
                (value * 255.0) as u8,
                (value * 255.0) as u8,
                (value * 255.0) as u8
            );

            image.set_pixel(x as u32, y as u32, pixel);
        }
    }

    image.save(file_path).expect("failed to save image");
}

fn save_image(image_data: ArrayView2<f32>, file_path: &str) {
    let width = image_data.shape()[0] as u32;
    let height = image_data.shape()[1] as u32;

    let mut image = Image::new(width, height);

    for (y, col) in image_data.outer_iter().enumerate() {
        for (x, value) in col.iter().enumerate() {
            let pixel = px!(
                (value * 255.0) as u8,
                (value * 255.0) as u8,
                (value * 255.0) as u8
            );

            image.set_pixel(x as u32, y as u32, pixel);
        }
    }

    image.save(file_path).expect("failed to save image");
}

fn inputs_to_images(inputs: ArrayView2<f32>, image_dimensions: [usize; 2]) -> ArrayView3<f32> {
    inputs
        .into_shape((inputs.shape()[0], image_dimensions[0], image_dimensions[1]))
        .unwrap()
}

fn images_to_inputs(images: ArrayView3<f32>, input_size: usize) -> ArrayView2<f32> {
    images.into_shape((images.shape()[0], input_size)).unwrap()
}
