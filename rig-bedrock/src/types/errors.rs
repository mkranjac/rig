use std::{
    error::Error,
    fmt::{Display, Formatter},
};

use aws_sdk_bedrockruntime::error::BuildError;
use tracing::error;

#[derive(Debug)]
pub enum IntegrationError {
    UnsupportedFeature(&'static str),
    UnsupportedFormat(String),
    BuildError(BuildError),
    ConversionError(String),
    ModelError(&'static str),
}

impl Display for IntegrationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use IntegrationError::*;
        match self {
            UnsupportedFeature(err) => {
                error!("Unsupported feature: {}", err);
                write!(f, "Unsupported feature: {}", err)
            }
            UnsupportedFormat(err) => {
                error!("Unsupported format: {}", err);
                write!(f, "Unsupported format: {}", err)
            }
            BuildError(err) => {
                error!("Failed to build: {}", err);
                write!(f, "Failed to build: {}", err)
            }
            ConversionError(err) => {
                error!("Failed to convert: {}", err);
                write!(f, "Failed to convert: {}", err)
            }
            ModelError(err) => {
                error!("Model error: {}", err);
                write!(f, "Model error: {}", err)
            }
        }
    }
}

impl Error for IntegrationError {}
