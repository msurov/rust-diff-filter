use core::time;
use std::mem::Discriminant;

mod utils;
use nalgebra::{dmatrix, Matrix, DMatrix, DMatrixViewMut, VecStorage, Dyn};
type MatX = nalgebra::DMatrix<f64>;
type VecX = nalgebra::DVector<f64>;

#[derive(Clone)]
#[allow(non_snake_case)]
pub struct StateSpace {
  A : MatX,
  B : MatX,
  C : MatX,
  D : MatX
}

#[allow(unused)]
pub struct TransferFunction {
  num : VecX,
  denom : VecX
}

impl TransferFunction {
  pub fn make_diff_filter(order : u32, time_constant : f64) -> TransferFunction {
    let mut denom = VecX::zeros(order as usize + 1);
    let mut time_constant_powk : f64 = 1.0;
  
    for k in 0..=order {
      denom[k as usize] = utils::binom_f64(order, k) * time_constant_powk;
      time_constant_powk *= time_constant;
    }

    let num = VecX::from_element(1, 1.0);
    TransferFunction {
      num, denom
    }
  }
}

impl StateSpace {
  pub fn create(ninputs : usize, noutputs : usize, nstates : usize) -> StateSpace {
    StateSpace {
      A : MatX::zeros(nstates, nstates),
      B : MatX::zeros(nstates, ninputs),
      C : MatX::zeros(noutputs , nstates),
      D : MatX::zeros( noutputs, ninputs),
    }
  }

  #[allow(non_snake_case)]
  pub fn make_diff_filter(order : u32, time_constant : f64) -> StateSpace {
    let tf: TransferFunction = TransferFunction::make_diff_filter(order, time_constant);
    let nstates = order as usize;
    let noutputs = order as usize + 1;
    let ninputs : usize = 1;
    let mut ss = StateSpace::create(ninputs, noutputs, nstates);
    let an = tf.denom[nstates];

    // fill A
    utils::fill_upper_diag(&mut ss.A, 1.0, 1);
    let row = tf.denom.transpose().view((0, 0), (1, nstates)) / an;
    ss.A.view_mut((nstates - 1, 0), (1, nstates)).copy_from(&row);

    // fill B
    ss.B[(nstates - 1, 0)] = 1.0 / an;

    // fill C
    utils::fill_upper_diag(&mut ss.C, 1.0, 0);
    ss.C.view_mut((noutputs - 1, 0), (1, nstates)).copy_from(&row);

    // fill D
    ss.D[(noutputs - 1, 0)] = 1.0 / an;

    ss
  }

  #[allow(non_snake_case)]
  pub fn discretize(&self, step : f64) -> StateSpace {
    let expAdt = (&self.A * step).exp();
    let tmp = &expAdt * &self.B - &self.B;
    let Bd = self.A.clone().qr().solve(&tmp).expect("matrix A is singular, can't discretize");

    StateSpace { 
      A: expAdt,
      B: Bd,
      C: self.C.clone(),
      D: self.D.clone()
    }
  }
}

fn main() {
  let filter = StateSpace::make_diff_filter(2, 0.1);
  let discrete = filter.discretize(0.01);
  println!("A = {}", discrete.A);
  println!("B = {}", discrete.B);
  println!("C = {}", discrete.C);
  println!("D = {}", discrete.D);
}
