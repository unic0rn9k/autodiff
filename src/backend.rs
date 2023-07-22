use crate::error::Result;
use cudarc::driver::safe;
use std::sync::Arc;

/// A backend is a device that can allocate and transfer data.
/// Specifically all operations are implemented for a specific backend.
/// So you might have multiple different backends that all use the CPU,
/// but one might use BLAS and another might use the matrixmultiply crate.
pub trait Backend<T> {
    type DevicePtr;

    /// Allocate `size_of::<T>() * elements` bytes of memory on self.
    fn alloc(&self, elements: usize) -> Result<Self::DevicePtr>;

    /// Copy `host` into `device`.
    fn htod_into(&self, host: Vec<T>, device: &mut Self::DevicePtr) -> Result<()>;

    /// Copy `host` into a new allocation on `self`.
    fn htod(&self, host: Vec<T>) -> Result<Self::DevicePtr> {
        let mut data = self.alloc(host.len())?;
        self.htod_into(host, &mut data)?;
        Ok(data)
    }

    /// Copy value in `src` and return it.
    fn dtoh(&self, src: &Self::DevicePtr) -> Result<Box<[T]>>;
}

/// A backend that uses the CPU, and allocated in Boxes.
pub struct CpuHeap;

impl<T: Clone> Backend<T> for CpuHeap {
    type DevicePtr = Box<[T]>;

    fn alloc(&self, elements: usize) -> Result<Self::DevicePtr> {
        #[allow(clippy::uninit_assumed_init)]
        Ok(
            vec![unsafe { std::mem::MaybeUninit::zeroed().assume_init() }; elements]
                .into_boxed_slice(),
        )
    }

    fn htod_into(&self, host: Vec<T>, dst: &mut Self::DevicePtr) -> Result<()> {
        *dst = host.into_boxed_slice();
        Ok(())
    }

    fn dtoh(&self, src: &Self::DevicePtr) -> Result<Box<[T]>> {
        Ok(src.clone())
    }
}

/// A backend that uses the cudarc, for CUDA GPU support.
pub struct Cuda(Arc<safe::CudaDevice>);

impl<T: cudarc::driver::ValidAsZeroBits + cudarc::driver::DeviceRepr + Unpin> Backend<T> for Cuda {
    type DevicePtr = safe::CudaSlice<T>;

    fn alloc(&self, elements: usize) -> Result<Self::DevicePtr> {
        Ok(self.0.alloc_zeros::<T>(elements)?)
    }

    fn htod_into(&self, host: Vec<T>, dst: &mut Self::DevicePtr) -> Result<()> {
        Ok(self.0.htod_copy_into(host, dst)?)
    }

    fn dtoh(&self, src: &Self::DevicePtr) -> Result<Box<[T]>> {
        Ok(self.0.dtoh_sync_copy(src)?.into_boxed_slice())
    }
}
