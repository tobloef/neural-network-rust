use super::{format::SavedMetaData, SavingConfigBuilder};

#[derive(Clone)]
pub struct SavingConfig {
    pub base_path: Option<String>,
    pub file_name: String,
    pub meta_data: Option<SavedMetaData>,
}

impl<'a> SavingConfig {
    pub fn builder() -> SavingConfigBuilder {
        SavingConfigBuilder::new()
    }
}
