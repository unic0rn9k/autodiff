pub trait ElemMul<Rhs = Self> {
    type Output;
    fn elem_mul(self, rhs: Rhs) -> Self::Output;
}

macro_rules! impl_elem_mul {
    ($($t: ty)*) => {$(
        impl ElemMul for $t {
            type Output = $t;
            fn elem_mul(self, rhs: Self) -> Self::Output {
                self * rhs
            }
        }
    )*};
}

impl_elem_mul!(f32 f64 u8 i8 u16 i16 u32 i32 u64 i64 u128 i128);
