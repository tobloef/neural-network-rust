use super::TestingConfig;
use std::time::Duration;

pub struct TestingResult {
    pub accuracy: f32,
    pub duration: Duration,
}

pub trait Test {
    fn test(&self, config: &TestingConfig) -> TestingResult;
}
