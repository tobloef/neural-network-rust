use super::SavingConfig;

pub trait Save {
    fn save(&self, config: SavingConfig);
}
