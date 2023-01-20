use std::fmt::{self, Debug, Display};

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum ErrorFunctionType {
    MeanSquared,
    CategoricalCrossentropy,
    BinaryCrossentropy,
}

impl Display for ErrorFunctionType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Debug::fmt(self, f)
    }
}
