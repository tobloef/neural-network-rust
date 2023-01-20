use ndarray::{Array1, ArrayView1};

pub fn binary_cross_entropy_error(
    predicted_output: ArrayView1<f32>,
    expected_output: ArrayView1<f32>,
) -> f32 {
    let clipped_predicted = predicted_output.mapv(|x| x.max(f32::EPSILON).min(1.0 - f32::EPSILON));

    let a = (1.0 - &expected_output) * clipped_predicted.mapv(|x| (1.0 - x).ln());
    let b = &expected_output * clipped_predicted.mapv(|x| x.ln());
    let result = -(a + b).mean().unwrap();

    return result;
}

pub fn binary_cross_entropy_error_derivative(
    predicted_output: ArrayView1<f32>,
    expected_output: ArrayView1<f32>,
) -> Array1<f32> {
    let clipped_predicted = predicted_output.mapv(|x| x.max(f32::EPSILON).min(1.0 - f32::EPSILON));

    let a = (1.0 - &expected_output) / (1.0 - &clipped_predicted);
    let b = &expected_output / &clipped_predicted;
    let result = (&a - &b) / expected_output.len() as f32;

    result
}
