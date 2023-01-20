use super::LoadingConfig;

#[derive(Clone)]
pub struct LoadingConfigBuilder {
    base_path: Option<String>,
    file_name: Option<String>,
}

impl LoadingConfigBuilder {
    pub fn new() -> Self {
        Self {
            base_path: None,
            file_name: None,
        }
    }

    pub fn base_path(&mut self, base_path: &str) -> &mut Self {
        self.base_path = Some(base_path.to_string());
        self
    }

    pub fn file_name(&mut self, file_name: &str) -> &mut Self {
        self.file_name = Some(file_name.to_string());
        self
    }

    pub fn draft(&self) -> Self {
        self.clone()
    }

    pub fn build(&self) -> LoadingConfig {
        LoadingConfig {
            base_path: self.base_path.clone(),
            file_name: self.file_name.clone().expect("file name must be set"),
        }
    }
}
