use ndarray::{Array, Array1, Array2};
use ndarray_rand::RandomExt;
use rand::RngCore;
use rand_distr::{Normal, Uniform};

pub fn xavier(
    input_size: usize,
    output_size: usize,
    rng: &mut Box<dyn RngCore>,
) -> (Array2<f32>, Array1<f32>) {
    let weight_bounds = 1.0 / (input_size as f32).sqrt();
    let weights = Array::random_using(
        (output_size, input_size),
        Uniform::new(-weight_bounds, weight_bounds),
        rng,
    );

    let biases = Array::random_using(output_size, Uniform::new(0.0, weight_bounds), rng);

    return (weights, biases);
}

pub fn he(
    input_size: usize,
    output_size: usize,
    rng: &mut Box<dyn RngCore>,
) -> (Array2<f32>, Array1<f32>) {
    let weight_bounds = (2.0 / input_size as f32).sqrt();
    let weights = Array::random_using(
        (output_size, input_size),
        Normal::new(0.0, weight_bounds).unwrap(),
        rng,
    );

    let biases = Array::random_using(output_size, Normal::new(0.0, weight_bounds).unwrap(), rng);

    return (weights, biases);
}
