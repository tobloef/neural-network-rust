use super::TestingConfigBuilder;
use ndarray::ArrayView2;

#[derive(Clone)]
pub struct TestingConfig<'a> {
    pub inputs: ArrayView2<'a, f32>,
    pub expected_outputs: ArrayView2<'a, f32>,
    pub hot_outputs: usize,
}

impl<'a> TestingConfig<'a> {
    pub fn builder() -> TestingConfigBuilder<'a> {
        TestingConfigBuilder::new()
    }
}
