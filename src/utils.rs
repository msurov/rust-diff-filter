use nalgebra::DMatrix;


pub fn fill_upper_diag(mat : &mut nalgebra::DMatrix<f64>, val : f64, displacement : i32) {
  let (ny, nx) = mat.shape();
  if displacement > 0 {
    let displacement = displacement as usize;
    let xmax = usize::min(nx, ny + displacement);
    for ix in displacement..xmax {
      let iy = ix - displacement;
      mat[(iy, ix)] = val;
    }
  } else {
    let displacement = (-displacement) as usize;
    let xmax = usize::min(nx + displacement, ny);
    for iy in displacement..xmax {
      let ix = iy - displacement;
      mat[(iy, ix)] = val;
    }
  }
}

pub fn binom_u32(n : u32, k : u32) -> u32 {
  let mut num = 1;
  let mut den = 1;

  if k < n - k {
    for i in 1..=k {
      num *= n - k + i;
      den *= i;
    }
  } else {
    for i in 1..=n - k {
      num *= k + i;
      den *= i;
    }
  }
  num / den
}

pub fn binom_f64(n : u32, k : u32) -> f64 {
  let mut val : f64 = 1.0;

  if k < n - k {
    for i in 1..=k {
      val *= f64::from((n - k + i) / i);
    }
  } else {
    for i in 1..=n - k {
      val *= f64::from((k + i) / i);
    }
  }

  val
}
