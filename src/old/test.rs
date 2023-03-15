use flamer::flame;

#[test]
#[flame]
fn basic() {
    let x = Arc::new(2f32);
    let y = Arc::new(3f32);
    let f = Arc::new(Add(
        Arc::new(Mul(x.clone(), y.clone())),
        Arc::new(Mul(x.clone(), x.clone())),
    ));
    assert_eq!(f.eval(), 10.);
    let delta = f.derivative(&[x.id(), y.id()]);
    println!("dx={:?}", delta[0]);
    println!("dy={:?}", delta[1]);
    assert_eq!(delta[0].eval(), 7.);
    assert_eq!(delta[1].eval(), 2.);
}

macro_rules! test_graph {
    ($x: expr, $y: expr) => {
        Arc::new(Add(
            Arc::new(Mul($x.clone(), $y.clone())),
            Arc::new(Mul($x.clone(), $x.clone())),
        ))
    };
}

extern crate test;
use std::sync::Arc;

use test::{black_box, Bencher};

use crate::{Add, DiffNode, Mul};
#[bench]
fn ad_f(b: &mut Bencher) {
    let x = Arc::new(black_box(2f32));
    let y = Arc::new(black_box(3f32));
    let f = test_graph!(x, y);
    b.iter(|| {
        let f = f.clone();
        black_box(f).eval();
    });
}

#[bench]
fn ad_dfdx(b: &mut Bencher) {
    let x = Arc::new(black_box(2f32));
    let y = Arc::new(black_box(3f32));
    let f = test_graph!(x, y).derivative(&[x.id()])[0].clone();
    b.iter(|| {
        let f = f.clone();
        black_box(f).eval();
    });
}

#[bench]
fn ad_dfdy(b: &mut Bencher) {
    let x = Arc::new(black_box(2f32));
    let y = Arc::new(black_box(3f32));
    let f = test_graph!(x, y).derivative(&[y.id()])[0].clone();
    b.iter(|| {
        let f = f.clone();
        black_box(f).eval();
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
