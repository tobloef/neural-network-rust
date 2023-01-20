use super::LoadingConfigBuilder;

#[derive(Clone)]
pub struct LoadingConfig {
    pub base_path: Option<String>,
    pub file_name: String,
}

impl<'a> LoadingConfig {
    pub fn builder() -> LoadingConfigBuilder {
        LoadingConfigBuilder::new()
    }
}
