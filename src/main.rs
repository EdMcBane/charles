use std::ops::{Neg, Index, AddAssign, MulAssign, DivAssign, IndexMut, Add, Sub, Mul, Div};
use std::fmt::{Display, Formatter};
use std::io::{stdout, Write};


#[derive(Copy, Clone)]
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

    fn x(&self) -> f32 {
        self.0[0]
    }
    fn y(&self) -> f32 {
        self.0[1]
    }
    fn z(&self) -> f32 {
        self.0[2]
    }

    fn length_squared(&self) -> f32 {
        self.0.into_iter().map(|e| e.powi(2)).sum()
    }

    fn length(&self) -> f32 {
        self.length_squared().sqrt()
    }

    fn dot(&self, rhs: &Self) -> f32 {
        self.0.into_iter().zip(rhs.0.into_iter()).map(|(l, r)| l * r).sum()
    }

    fn cross(&self, rhs: Self) -> Self {
        Vec3([self.0[1] * rhs.0[2] - self.0[2] * rhs.0[1],
            self.0[2] * rhs.0[0] - self.0[0] * rhs.0[2],
            self.0[0] * rhs.0[1] - self.0[1] * rhs.0[0]])
    }

    fn unit_vector(self) -> Self {
        let len = self.length();
        self / len
    }

    fn write_color<W: Write>(&self, fmt: &mut W) {
        writeln!(fmt, "{} {} {}",
                 (255.999 * self.x()) as u8,
                 (255.999 * self.y()) as u8,
                 (255.999 * self.z()) as u8).unwrap();
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

impl Add for Vec3 {
    type Output = Vec3;

    fn add(self, rhs: Self) -> Self::Output {
        Vec3([self.0[0] + rhs.0[0],
            self.0[1] + rhs.0[1],
            self.0[2] + rhs.0[2]])
    }
}

impl Sub for Vec3 {
    type Output = Vec3;

    fn sub(self, rhs: Self) -> Self::Output {
        Vec3([self.0[0] - rhs.0[0],
            self.0[1] - rhs.0[1],
            self.0[2] - rhs.0[2]])
    }
}

impl Mul<Vec3> for Vec3 {
    type Output = Vec3;

    fn mul(self, rhs: Self) -> Self::Output {
        Vec3([self.0[0] * rhs.0[0],
            self.0[1] * rhs.0[1],
            self.0[2] * rhs.0[2]])
    }
}

impl Mul<Vec3> for f32 {
    type Output = Vec3;

    fn mul(self, rhs: Vec3) -> Self::Output {
        rhs * self
    }
}

impl Mul<f32> for Vec3 {
    type Output = Vec3;

    fn mul(self, rhs: f32) -> Self::Output {
        Vec3([self.0[0] * rhs,
            self.0[1] * rhs,
            self.0[2] * rhs])
    }
}

impl Div<f32> for Vec3 {
    type Output = Vec3;

    fn div(self, rhs: f32) -> Self::Output {
        self.mul(1. / rhs)
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

impl Display for Vec3 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{:?}", self.0))
    }
}

type Point3 = Vec3;
type Color = Vec3;

struct Ray {
    origin: Point3,
    dir: Vec3,
}

impl Ray {
    fn index(&self, t: f32) -> Vec3 {
        self.origin + self.dir * t
    }
}

fn ray_color(r: &Ray) -> Color {
    if hit_sphere(&Point3::new(0., 0., -1.), 0.5, r) {
        return Color::new(1., 0., 0.);
    }
    let unit_dir = r.dir.unit_vector();
    let t = 0.5 * unit_dir.y() + 1.0;
    (1.0 - t) * Color::new(1., 1., 1.) + t * Color::new(0.5, 0.7, 1.0)
}

fn test_image() {

    const WIDTH: usize = 256;
    const HEIGHT: usize = 256;

    println!("P3");
    println!("{} {}", WIDTH, HEIGHT);
    println!("{}", u8::MAX);
    for j in (0..WIDTH).rev() {
        eprint!("\rScanlines remaining: {}", j);
        for i in 0..HEIGHT {
            Color::new(i as f32 / (WIDTH - 1) as f32,
                       j as f32 / (HEIGHT - 1) as f32,
                       0.25).write_color(&mut stdout());
        }
    }
}

fn hit_sphere(center: &Point3, radius: f32, r: &Ray) -> bool {
    let oc = r.origin - *center;
    let a = Vec3::dot(&r.dir, &r.dir);
    let b = 2. * Vec3::dot(&oc, &r.dir);
    let c = Vec3::dot(&oc, &oc) - radius.powi(2);
    let discriminant = b*b - 4.*a*c;
    discriminant > 0.
}

fn main() {
    let aspect_ratio = 16.0 / 9.0;
    let image_width = 400usize;
    let image_height = (image_width as f32 / aspect_ratio) as usize;
    // Image
    let viewport_height = 2.0;
    let viewport_width = aspect_ratio * viewport_height;
    let focal_length = 1.0;
    // Camera
    let origin = Point3::new(0.,0.,0.);
    let horizontal = Vec3::new(viewport_width, 0., 0.);
    let vertical = Vec3::new(0., viewport_height, 0.);
    let lower_left_corner = origin - horizontal / 2. - vertical / 2. - Vec3::new(0.,0.,focal_length);
    // Render
    println!("P3");
    println!("{} {}", image_width, image_height);
    println!("{}", u8::MAX);
    for j in (0..image_height).rev() {
        eprint!("\rScanlines remaining: {}", j);
        for i in 0..image_width {
            let u = i as f32 / (image_width - 1) as f32;
            let v = j as f32 / (image_height - 1) as f32;
            let ray = Ray {
                origin: origin,
                dir: lower_left_corner + u*horizontal + v*vertical - origin,
            };
            let color = ray_color(&ray);
            color.write_color(&mut stdout());
        }
    }



}