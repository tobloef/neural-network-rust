use ndarray::{Array1, ArrayView1};

pub fn mean_squared_error(
    predicted_output: ArrayView1<f32>,
    expected_output: ArrayView1<f32>,
) -> f32 {
    let error = &predicted_output - &expected_output;
    let squared_error = error.mapv(|x| x.powi(2));

    squared_error.mean().unwrap_or(0.0)
}

pub fn mean_squared_error_derivative(
    predicted_output: ArrayView1<f32>,
    expected_output: ArrayView1<f32>,
) -> Array1<f32> {
    let num_samples = predicted_output.len();
    let error = &predicted_output - &expected_output;

    2.0 * error / num_samples as f32
}
