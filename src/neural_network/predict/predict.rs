use ndarray::{Array1, ArrayView1};

pub trait Predict {
    fn predict(&self, input: ArrayView1<f32>, hot_outputs: usize) -> Array1<f32>;
}
