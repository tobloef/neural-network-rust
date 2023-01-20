use ndarray::{Array1, Array2, ArrayView1, Axis};

pub fn to_2d(input: ArrayView1<f32>) -> Array2<f32> {
    input.to_owned().insert_axis(Axis(0))
}

pub fn to_2d_transposed(input: ArrayView1<f32>) -> Array2<f32> {
    input.to_owned().insert_axis(Axis(1))
}

pub fn arrayview1_eqauls(a: ArrayView1<f32>, b: ArrayView1<f32>) -> bool {
    a.iter().zip(b.iter()).all(|(a, b)| a == b)
}
