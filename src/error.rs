#[cfg(feature = "cuda")]
use cudarc::driver::safe::DriverError;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[cfg(feature = "cuda")]
    #[error("cudarc error: {0}")]
    CudaError(#[from] DriverError),

    #[error("Other error: {0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;
