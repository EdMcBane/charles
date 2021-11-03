use std::ops::{Neg, Index, AddAssign, MulAssign, DivAssign, IndexMut};

const WIDTH: usize = 256;
const HEIGHT: usize = 256;

struct Vec3([f32; 3]);

impl Default for Vec3 {
    fn default() -> Self {
        Vec3([0., 0., 0.])
    }
}

impl Vec3 {
    fn new(x: f32, y: f32, z: f32) -> Self {
        Vec3([x, y, z])
    }

    fn x(self) -> f32 {
        self.0[0]
    }
    fn y(self) -> f32 {
        self.0[1]
    }
    fn z(self) -> f32 {
        self.0[2]
    }

    fn length_squared(&self) -> f32 {
        self.0.into_iter().map(|e| e.powi(2)).sum()
    }

    fn length(&self) -> f32 {
        self.length_squared().sqrt()
    }
}

impl Neg for Vec3 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Vec3([-self.0[0], -self.0[1], -self.0[2]])
    }
}

impl Index<usize> for Vec3 {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl IndexMut<usize> for Vec3 {

    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

impl AddAssign for Vec3 {
    fn add_assign(&mut self, rhs: Self) {
        for (l, r) in self.0.iter_mut().zip(rhs.0.into_iter()) {
            *l += r;
        }
    }
}

impl MulAssign<f32> for Vec3 {
    fn mul_assign(&mut self, t: f32) {
        for l in self.0.iter_mut() {
            *l *= t;
        }
    }
}

impl DivAssign<f32> for Vec3 {
    fn div_assign(&mut self, t: f32) {
        for l in self.0.iter_mut() {
            *l /= t;
        }
    }
}

fn main() {
    println!("P3");
    println!("{} {}", WIDTH, HEIGHT);
    println!("{}", u8::MAX);
    for j in (0..WIDTH).rev() {
        eprint!("\rScanlines remaining: {}", j);
        for i in 0..HEIGHT {
            let (r, g, b) = (
                (255.999 * i as f32 / (WIDTH - 1) as f32) as u8,
                (255.999 * j as f32 / (WIDTH - 1) as f32) as u8,
                (255.999 * 0.25) as u8);
            println!("{} {} {}", r, g, b);
        }
    }
}
