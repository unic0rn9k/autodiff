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

fn argmax(x: &[f32]) -> usize {
    let mut max = 0.;
    let mut max_idx = 0;
    for (i, &x) in x.iter().enumerate() {
        if x > max {
            max = x;
            max_idx = i;
        }
    }
    max_idx
}

fn main() {
    let ntrain: usize = 20000;
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

    let mut w =
        mat(DMatrix::<f32>::new_random(10, s).map(|n| n - 0.5) / (28. * 28. / 2.)).symbol("w");
    let mut b =
        mat(DMatrix::<f32>::new_random(10, 1).map(|n| n - 0.5) / (28. * 28. / 2.)).symbol("b");

    for n in 0..ntrain {
        let x = mat(DMatrix::<f32>::from_iterator(
            s,
            1,
            trn_img[n * s..n * s + s].iter().map(|x| *x as f32 / 255.),
        ))
        .symbol("x");

        let y = &w * &x + &b;
        let y = y.exp() / Sum(y.exp());

        let out = y.eval().0.unwrap();

        let target = DMatrix::<f32>::from_iterator(
            10,
            1,
            (0..10).map(|i| if i == trn_lbl[n] as usize { 1. } else { 0. }),
        );

        // let dy = (y - mat(target)) * 2f32; // Compiletime er over en time, hvis dette også skal være autodiff

        let dy = mat(2. * (out.clone() - target));

        let [dw, db] = y.derivative(["w", "b"], dy);
        let dw = dw.eval().0.unwrap();
        let db = db.eval().0.unwrap();

        *w.0.node.0.as_mut().unwrap() -= dw * 0.001;
        *b.0.node.0.as_mut().unwrap() -= db * 0.001;

        let mut correct = 0;
        for t in 0..ntest {
            let x = mat(DMatrix::<f32>::from_iterator(
                s,
                1,
                tst_img[t * s..t * s + s].iter().map(|x| *x as f32 / 255.),
            ))
            .symbol("x");
            let y = &w * x + &b;

            if tst_lbl[t] as usize == argmax(y.eval().0.unwrap().as_slice()) {
                correct += 1;
            }
        }

        println!("{n} accuracy: {}%", correct as f32 / ntest as f32 * 100.,);
    }
}
