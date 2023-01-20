use ndarray::{Array1, ArrayView1};

pub fn categorical_cross_entropy_error(
    predicted_output: ArrayView1<f32>,
    expected_output: ArrayView1<f32>,
) -> f32 {
    let pairs = predicted_output.iter().zip(expected_output.iter());

    let result: f32 = -pairs
        .map(|(p, e)| *e * (p + f32::EPSILON).ln())
        .sum::<f32>();

    result
}

pub fn categorical_cross_entropy_error_derivative(
    predicted_output: ArrayView1<f32>,
    expected_output: ArrayView1<f32>,
) -> Array1<f32> {
    let a = &predicted_output * (1.0 - &predicted_output);
    let b = &predicted_output - &expected_output;
    let result = &b / (&a + f32::EPSILON);

    result
}
