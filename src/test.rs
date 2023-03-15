use flamer::flame;
extern crate test;
use test::{black_box, Bencher};

use crate::*;

#[flame]
#[bench]
fn basic(b: &mut Bencher) {
    let x = black_box(2f32.symbol("x"));
    let y = black_box(3f32.symbol("y"));
    b.iter(|| {
        black_box(Add(Mul(&x, &y), Mul(&x, &x))).eval();
        //assert_eq!(f.eval(), 10.);
        //black_box(f.derivative(&["x", "y"])[0].eval());
        //println!("dx={:?}", delta[0]);
        //println!("dy={:?}", delta[1]);
        //assert_eq!(delta[0].eval(), 7.);
        //assert_eq!(delta[1].eval(), 2.);
    });
}

#[bench]
fn hand_written(b: &mut Bencher) {
    b.iter(|| {
        let x = black_box(2f32);
        let y = black_box(3f32);
        black_box(x * x + x * y);
    });
}
