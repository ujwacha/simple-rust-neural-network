use image_recognizer_nn::{init_network, Network};
use mnist::*;
use ndarray::prelude::*;
use ndarray::{self, Array, OwnedRepr};

fn main() {
    println!(" ::: hello, enter 0 to train, 1 to see predictions ::: ");
    let mut input = String::new();

    std::io::stdin()
        .read_line(&mut input)
        .expect("couldn't read line");

    let choice: i32 = input.trim().parse().expect("enter a number");

    if choice == 0 {
        train()
    } else {
        predictor()
    }
}

fn train() {
    let first = 28 * 28;
    let second = 500;
    let third = 10;

    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    let mut train_images = Vec::new();
    for chunks in trn_img.chunks(first) {
        let image_matrix = Array::from_shape_vec((first, 1), chunks.to_vec())
            .unwrap()
            .map(|x| *x as f64 / 256.0);
        train_images.push(image_matrix);
    }

    // for k in 0..20 {
    //     print_image(&train_images[k]);

    //     println!("{}", trn_lbl[k]);
    // }
    let mut network = init_network(first, second, third);

    let mut count: u64 = 0;

    println!("training the model, it's gonna take a lot of time");

    for (img, lbl) in train_images.iter().zip(trn_lbl) {
        network = network.forward_propagate(img.clone()).unwrap();

        let (net, loss) = network.backprop(lbl, EPSILON);

        network = net;

        if count % 100 == 0 {
            println!("loss: {loss}");
        }

        count += 1;
    }

    let mut test_image = Vec::new();
    for chunks in tst_img.chunks(first) {
        let image_matrix = Array::from_shape_vec((first, 1), chunks.to_vec())
            .unwrap()
            .map(|x| *x as f64 / 256.0);
        test_image.push(image_matrix);
    }

    let mut corrects = 0.0;

    for thing in test_image.iter().zip(tst_lbl).take(300) {
        let (img, lbl) = thing;

        let i = network.predict(img.clone());
        if i == lbl as usize {
            corrects += 1.0;
        }
    }

    let percentage = (corrects * 100.0) / 300.0;
    println!("works {} % of the time", percentage);

    println!("saving the model....");
    network.save("model.cbor").unwrap();
}

fn predictor() {
    let first = 28 * 28;
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    let mut test_image = Vec::new();
    for chunks in tst_img.chunks(first) {
        let image_matrix = Array::from_shape_vec((first, 1), chunks.to_vec())
            .unwrap()
            .map(|x| *x as f64 / 256.0);
        test_image.push(image_matrix);
    }

    let net = Network::load("model.cbor").unwrap();

    let mut input = String::new();

    loop {
        println!("select the number from 1 to 10,000 to check the model");

        std::io::stdin()
            .read_line(&mut input)
            .expect("error reading the line");

        let index: usize = input.trim().parse().expect("give a number");
        let index = index - 1;

        print_image(&test_image[index]);

        println!("correct label: {}", trn_lbl[index]);

        println!(
            "predicted label: {}",
            net.predict(test_image[index].clone())
        )
    }
}

fn getputput_matrix(data: u8) -> ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> {
    let mut outtest = vec![0.0; 10];
    outtest[data as usize] = 1.0;
    Array::from_shape_vec((1, 10), outtest).unwrap()
}

fn print_image(train_image: &ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>) {
    for i in 0..28 {
        for j in 0..28 {
            if train_image[[i * 28 + j, 0]] == 0.0 {
                print!(" ");
            } else {
                print!("#");
            }
        }

        print!("\n");
    }
}

const EPSILON: f64 = 1e-3;
// end
