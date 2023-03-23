pub struct Bit<const B: bool>;
pub type Zero = Bit<false>;
pub type One = Bit<true>;

pub trait Scalar: From<Zero> + From<One> {}

macro_rules! impl_scalar {
    ($($t: ty)*) => {$(
        impl From<Zero> for $t {
            fn from(_: Zero) -> Self {
                0 as $t
            }
        }
        impl From<One> for $t {
            fn from(_: One) -> Self {
                1 as $t
            }
        }
    )*};
}

impl_scalar!(u8 i8 u16 i16 u32 i32 u64 i64 u128 i128 f32 f64);
