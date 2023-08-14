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
# Long compile times
The repport also details some of the issues with leveraging GATs for autodiff, which lead me to try to rewrite the projekt on a new branch (`runtime-compiled-graphs`), where I instead use a structure for representing expressions, which can then be compiled into a evaluatable graph, at run time. The graph can then be reused for identical functions, with different values. This also allows for optimization of the mathematical expression, caching of computed values, an pruning of identical graph segments.

One thing I wanted to work into this project, was GPU support, so `runtime-compiled-graphs` is build with multiple backends in mind, and is meant to have GPU support out of the box.
