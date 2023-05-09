use std::{
    fs::File,
    io::{Read, Write},
};

use serde::{Deserialize, Serialize};

use ndarray::{Array, ArrayBase, Dim, OwnedRepr};
use rand::{self, Rng};

fn relu(x: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

fn d_relu(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else {
        0.0
    }
}

#[derive(Serialize, Deserialize)]
pub struct Network {
    a1: Matrix,
    a2: Matrix,
    a3: Matrix,
    w1: Matrix,
    w2: Matrix,
    b1: Matrix,
    b2: Matrix,
}

type Matrix = ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>;

pub fn init_network(first: usize, second: usize, third: usize) -> Network {
    let a1 = Array::from_elem((first, 1), 0.0);
    let a2 = Array::from_elem((second, 1), 0.0);
    let a3 = Array::from_elem((third, 1), 0.0);

    let w1 = Array::from_shape_fn((second, first), |(_i, _j)| {
        rand::thread_rng().gen_range(-0.5..=0.5)
    });

    let w2 = Array::from_shape_fn((third, second), |(_i, _j)| {
        rand::thread_rng().gen_range(-0.5..=0.5)
    });

    let b1 = Array::from_shape_fn((second, 1), |(_i, _j)| {
        rand::thread_rng().gen_range(-0.5..=0.5)
    });

    let b2 = Array::from_shape_fn((third, 1), |(_i, _j)| {
        rand::thread_rng().gen_range(-0.5..=0.5)
    });

    Network {
        a1,
        a2,
        a3,
        w1,
        w2,
        b1,
        b2,
    }
}

impl Network {
    pub fn save(&self, filename: &str) -> Result<(), std::io::Error> {
        // searilize the binary to a cbor file
        let bytes = serde_cbor::to_vec(self).unwrap();

        let mut file = File::create(filename)?;

        file.write_all(&bytes)?;

        Ok(())
    }

    pub fn predict(&self, image: Matrix) -> usize {
        let network = self.forward_propagate(image);

        let probabilty_distribution = network.unwrap().a3.clone();

        let mut number = 0.0;
        let mut index = 0;
        let mut i = 0;

        for element in probabilty_distribution {
            if element > number {
                index = i;
                number = element;
            }
            i += 1;
        }

        index
    }

    pub fn forward_propagate(&self, input: Matrix) -> Option<Self> {
        if input.dim() != self.a1.dim() {
            return None;
        }

        let a1 = input;

        let a2 = self.w1.dot(&a1) + self.b1.clone();

        let a2 = a2.map(|x| relu(*x));

        let a3 = self.w2.dot(&a2) + self.b2.clone();

        let a3 = probability_distribution(a3);

        Some(Network {
            a1,
            a2,
            a3,
            w1: self.w1.clone(),
            w2: self.w2.clone(),
            b1: self.b1.clone(),
            b2: self.b2.clone(),
        })
    }

    pub fn backprop(&self, correct_lbl: u8, epsilon: f64) -> (Network, f64) {
        let y = encode_lbl_to_matrix(correct_lbl);
        let m = y.len() as f64;

        let dz2 = &self.a3 - &y;

        let loss = dz2.map(|x| x.powf(2.0)).sum() * 100.0;

        let dw2 = &dz2.dot(&self.a2.t()) / m;
        let db2 = dz2.sum() / m;

        let dz1 = self.w2.t().dot(&dz2) * self.a2.map(|x| d_relu(*x));
        let dw1 = &dz1.dot(&self.a1.t()) / m;
        let db1 = dz1.sum() / m;

        let net = Network {
            a1: self.a1.clone(),
            a2: self.a2.clone(),
            a3: self.a3.clone(),
            w1: self.w1.clone() - (epsilon * dw1),
            w2: self.w2.clone() - (epsilon * dw2),
            b1: self.b1.clone() - (epsilon * db1),
            b2: self.b2.clone() - (epsilon * db2),
        };
        (net, loss)
    }

    pub fn load(filename: &str) -> Result<Self, std::io::Error> {
        // Read the binary data from a file
        let mut file = File::open(filename)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        // Deserialize the binary data to a Network struct
        let network: Network = serde_cbor::from_slice(&buffer).unwrap();

        Ok(network)
    }
}

fn encode_lbl_to_matrix(x: u8) -> Matrix {
    let mut return_array = Array::from_elem((10, 1), 0.0);
    *return_array.get_mut((x as usize, 0)).unwrap() = 1.0;
    return_array
}

fn probability_distribution(matrix: Matrix) -> Matrix {
    // i guess this is gonna be slow
    // but i trust my compiler
    matrix.map(|x| x.exp() / matrix.map(|x| x.exp()).sum())
}
