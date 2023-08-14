# A library for automatic differentiation using GATs [![Rust](https://github.com/unic0rn9k/autodiff/actions/workflows/rust.yml/badge.svg)](https://github.com/unic0rn9k/autodiff/actions/workflows/rust.yml)

This library was developed as a part of a school projekt, therefore it also includes a [repport written in Danish](https://github.com/unic0rn9k/autodiff/blob/master/rapport.pdf).

As a part of the projekt i also wrote an [mnist classifier](https://github.com/unic0rn9k/autodiff/blob/master/examples/mnist/src/main.rs), which is described in the repport.

To run the mnist classifier example, clone the repository, and run `cargo r --package mnist_classifier`.

# Example from unit test in src/mat.rs
```rust
    let x = mat(DMatrix::<f32>::new_random(3, 1)).symbol("input");

    let w1 = mat(DMatrix::<f32>::new_random(2, 3)).symbol("w1");
    let b1 = mat(DMatrix::<f32>::new_random(2, 1)).symbol("b1");

    let w2 = mat(DMatrix::<f32>::new_random(4, 2)).symbol("w2");
    let b2 = mat(DMatrix::<f32>::new_random(4, 1)).symbol("b2");

    let dl2 = mat(DMatrix::<f32>::repeat(4, 1, 1.));

    let l1 = &w1 * &x + &b1;
    let l2 = &w2 * &l1 + &b2;
    let [dw1, dw2] = l2.derivative(["w1", "w2"], &dl2);

    l2.eval();

    println!("dw1 = {dw1:?}");
    println!();
    println!("dw2 = {dw2:?}");
    println!();

    assert_eq!(dw2.eval().shape(), w2.eval().shape());
    assert_eq!(dw1.eval().shape(), w1.eval().shape());
```
