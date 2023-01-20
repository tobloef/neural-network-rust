use super::{format::SavedMetaData, SavingConfig};

#[derive(Clone)]
pub struct SavingConfigBuilder {
    base_path: Option<String>,
    file_name: Option<String>,
    meta_data: Option<SavedMetaData>,
}

impl SavingConfigBuilder {
    pub fn new() -> Self {
        Self {
            base_path: None,
            file_name: None,
            meta_data: None,
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

    pub fn meta_data(&mut self, meta_data: SavedMetaData) -> &mut Self {
        self.meta_data = Some(meta_data);
        self
    }

    pub fn draft(&self) -> Self {
        self.clone()
    }

    pub fn build(&self) -> SavingConfig {
        SavingConfig {
            base_path: self.base_path.to_owned(),
            file_name: self.file_name.to_owned().expect("file name must be set"),
            meta_data: self.meta_data.clone(),
        }
    }
}
