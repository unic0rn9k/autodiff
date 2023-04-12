use autodiff::mat::nalgebra::DMatrix;
use autodiff::prelude::*;
use mnist::MnistBuilder;

// softmax
// sum = exp(x_0) + exp(x_1) + ... + exp(x_n)
// softmax(x_n) = exp(x_n) / sum
//
// softmax'(x_n) = softmax(x_n) * (1 - softmax(x_n))
//
// f(x)/g(x) -> (f'(x)g(x) - f(x)g'(x)) / g(x)^2
// exp(x) / sum -> (exp(x) * sum - exp(x) * exp(x)) / sum^2

fn main() {
    let ntrain: usize = 1000;
    let ntest: usize = 100;
    let s = 28 * 28;

    let mnist::Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .base_path("data/")
        .training_set_length(ntrain as u32)
        .test_set_length(ntest as u32)
        .download_and_extract()
        .finalize();

    let mut w = mat(DMatrix::<f32>::new_random(10, s)).symbol("w");
    let mut b = mat(DMatrix::<f32>::new_random(10, 1)).symbol("b");

    for n in 0..1000 {
        let x = mat(DMatrix::<f32>::from_iterator(
            s,
            1,
            trn_img[n * s..n * s + s].iter().map(|x| *x as f32 / 255.),
        ))
        .symbol("x");

        let y = &w * &x + &b;
        let dy = mat(2.
            * (y.eval().0.unwrap()
                - DMatrix::<f32>::from_iterator(
                    10,
                    1,
                    (0..10).map(|i| if i == trn_lbl[n] as usize { 1. } else { 0. }),
                )));

        let [dw, db] = y.derivative([("w", &dy), ("b", &dy)]);
        let dw = dw.eval().0.unwrap();
        let db = db.eval().0.unwrap();

        *w.0.node.0.as_mut().unwrap() -= dw * 0.01;
        *b.0.node.0.as_mut().unwrap() -= db * 0.01;

        let mut correct = 0;
        for t in 0..ntest {
            let x = mat(DMatrix::<f32>::from_iterator(
                s,
                1,
                tst_img[t * s..t * s + s].iter().map(|x| *x as f32 / 255.),
            ))
            .symbol("x");
            let y = &w * x + &b;

            if tst_lbl[t] as usize == y.eval().argmax() {
                correct += 1;
            }
        }

        println!("{n} accuracy: {:.2}%", correct as f32 / ntest as f32 * 100.);
    }
}
