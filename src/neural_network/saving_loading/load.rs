use super::{LoadingConfig, SavedMetaData};

pub trait Load {
    fn load<'a>(config: LoadingConfig) -> (Self, Option<SavedMetaData>)
    where
        Self: Sized;
}
