#![feature(total_cmp)]

use std::ops::{Neg, Index, AddAssign, MulAssign, DivAssign, IndexMut, Add, Sub, Mul, Div, Range, Deref};
use std::fmt::{Display, Formatter};
use std::io::{stdout, Write};
use std::cmp::Ordering;
use rand::Rng;
use rand::rngs::ThreadRng;
use std::thread::Thread;
use rand::distributions::Uniform;
use rand::distributions::Distribution;

#[derive(Copy, Clone)]
struct Vec3([f64; 3]);

impl Default for Vec3 {
    fn default() -> Self {
        Vec3([0., 0., 0.])
    }
}

impl Vec3 {
    fn new(x: f64, y: f64, z: f64) -> Self {
        Vec3([x, y, z])
    }

    fn x(&self) -> f64 {
        self.0[0]
    }
    fn y(&self) -> f64 {
        self.0[1]
    }
    fn z(&self) -> f64 {
        self.0[2]
    }

    fn length_squared(&self) -> f64 {
        self.0.into_iter().map(|e| e.powi(2)).sum()
    }

    fn length(&self) -> f64 {
        self.length_squared().sqrt()
    }

    fn dot(&self, rhs: &Self) -> f64 {
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

    fn write_color<W: Write>(&self, fmt: &mut W, samples_per_pixel: usize) {
        let scale = 1.0 / samples_per_pixel as f64;
        writeln!(fmt, "{} {} {}",
                 (256. * (self.x() * scale).sqrt().clamp(0., 0.999)) as u8, // SQRT for gamma correction
                 (256. * (self.y() * scale).sqrt().clamp(0., 0.999)) as u8,
                 (256. * (self.z() * scale).sqrt().clamp(0., 0.999)) as u8).unwrap();
    }
}


impl Neg for Vec3 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Vec3([-self.0[0], -self.0[1], -self.0[2]])
    }
}

impl Index<usize> for Vec3 {
    type Output = f64;

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

impl Mul<Vec3> for f64 {
    type Output = Vec3;

    fn mul(self, rhs: Vec3) -> Self::Output {
        rhs * self
    }
}

impl Mul<f64> for Vec3 {
    type Output = Vec3;

    fn mul(self, rhs: f64) -> Self::Output {
        Vec3([self.0[0] * rhs,
            self.0[1] * rhs,
            self.0[2] * rhs])
    }
}

impl Div<f64> for Vec3 {
    type Output = Vec3;

    fn div(self, rhs: f64) -> Self::Output {
        self.mul(1. / rhs)
    }
}

impl MulAssign<f64> for Vec3 {
    fn mul_assign(&mut self, t: f64) {
        for l in self.0.iter_mut() {
            *l *= t;
        }
    }
}

impl DivAssign<f64> for Vec3 {
    fn div_assign(&mut self, t: f64) {
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
    fn at(&self, t: f64) -> Vec3 {
        self.origin + self.dir * t
    }
}

const SPHERE_CENTER: Point3 = Vec3([0., 0., -1.]);
const SPHERE_RADIUS: f64 = 0.5;
const COLOR_WHITE: Color = Vec3([1., 1., 1.]);
const COLOR_BLACK: Color = Vec3([0., 0., 0.]);





struct HitRecord {
    p: Point3,
    normal: Vec3,
    t: f64,
    front_face: bool,
}

impl HitRecord {
    fn new(t: f64, p: Point3, r: &Ray, outward_normal: Vec3) -> Self {
        let front_face = Vec3::dot(&r.dir, &outward_normal) < 0.;
        HitRecord {
            t,
            p,
            front_face,
            normal: if front_face { outward_normal } else { -outward_normal },
        }
    }
}

trait Hittable {
    fn hit(&self, r: &Ray, t_range: &Range<f64>) -> Option<HitRecord>;
}

struct Sphere {
    center: Point3,
    radius: f64,
}

impl Hittable for &Sphere {
    fn hit(&self, r: &Ray, t_range: &Range<f64>) -> Option<HitRecord> {
        (*self).hit(r, t_range)
    }
}

impl Hittable for Sphere {
    fn hit(&self, r: &Ray, t_range: &Range<f64>) -> Option<HitRecord> {
        let oc = r.origin - self.center;
        let a = r.dir.length_squared();
        let half_b = Vec3::dot(&oc, &r.dir);
        let c = oc.length_squared() - self.radius.powi(2);
        let discriminant = half_b.powi(2) - a * c;
        if discriminant < 0. {
            return None;
        }
        // nearest acceptable root
        let sqrtd = discriminant.sqrt();
        [-1., 1.].into_iter()
            .map(|sign| (-half_b + sqrtd * sign) / a)
            .find(|root| t_range.contains(root))
            .map(|root| {
                let p = r.at(root);
                HitRecord::new(root, p, r, (p - self.center) / self.radius)
            })
    }
}

impl Hittable for &Vec<Box<dyn Hittable>> {
    fn hit(&self, r: &Ray, t_range: &Range<f64>) -> Option<HitRecord> {
        self.iter()
            .filter_map(|hittable| hittable.hit(r, t_range))
            .min_by(|hr1, hr2| hr1.t.total_cmp(&hr2.t))
    }
}

fn degrees_to_radians(degrees: f64) -> f64 {
    degrees * std::f64::consts::PI / 180.
}

struct Camera {
    origin: Point3,
    lower_left_corner: Point3,
    horizontal: Vec3,
    vertical: Vec3,
}

impl Camera {
    fn new() -> Self {
        let aspect_ratio = 16.0 / 9.0;
        let viewport_height = 2.0;
        let viewport_width = aspect_ratio * viewport_height;
        let focal_length = 1.0;

        let origin = Point3::default();
        let horizontal = Vec3::new(viewport_width, 0., 0.);
        let vertical = Vec3::new(0., viewport_height, 0.);
        Camera {
            origin,
            horizontal,
            vertical,
            lower_left_corner: origin - horizontal / 2. - vertical / 2. - Vec3::new(0., 0., focal_length),
        }
    }
    fn get_ray(&self, u: f64, v: f64) -> Ray {
        Ray {
            origin: self.origin,
            dir: self.lower_left_corner + u * self.horizontal + v * self.vertical - self.origin,
        }
    }
}

struct Raycaster {
    rng: ThreadRng,
    unit: Uniform<f64>,
}

impl Raycaster {
    fn new() -> Self {
        Raycaster {
            rng: rand::thread_rng(),
            unit: Uniform::from(-1.0..1.0),
        }
    }
    fn random_in_unit_sphere(&mut self) -> Vec3 {
        loop {
            let p = Vec3::new(
                self.unit.sample(&mut self.rng),
                self.unit.sample(&mut self.rng),
                self.unit.sample(&mut self.rng));
            if p.length_squared() < 1. {
                return p
            }
        }
    }
    

    fn ray_color<W: Hittable>(&mut self, r: &Ray, world: W, depth: usize) -> Color {
        if depth == 0 {
            return COLOR_BLACK;
        }
        let range = 0.001f64..f64::INFINITY;
        if let Some(rec) = world.hit(r, &range) {
            let target = rec.p + rec.normal + self.random_in_unit_sphere();
            return 0.5 * self.ray_color(&Ray { origin: rec.p, dir: target - rec.p}, world, depth - 1);
        }
        let unit_dir = r.dir.unit_vector();
        let t = 0.5 * (unit_dir.y() + 1.0);
        (1.0 - t) * COLOR_WHITE + t * Color::new(0.5, 0.7, 1.0)
    }

    fn main(&mut self) {
        // Image
        let aspect_ratio = 16.0 / 9.0;
        let image_width = 400usize;
        let image_height = (image_width as f64 / aspect_ratio) as usize;
        let samples_per_pixel = 100;
        let max_depth = 50;
        // World
        let world: Vec<Box<dyn Hittable>> = vec![Box::new(Sphere {
            center: Point3::new(0., 0., -1.),
            radius: 0.5,
        }), Box::new(Sphere {
            center: Point3::new(0., -100.5, -1.),
            radius: 100.,
        })];

        // Camera
        let cam = Camera::new();
        let mut rand = rand::thread_rng();
        // Render
        println!("P3");
        println!("{} {}", image_width, image_height);
        println!("{}", u8::MAX);
        for j in (0..image_height).rev() {
            eprint!("\rScanlines remaining: {}", j);
            for i in 0..image_width {
                let mut pixel_color = COLOR_BLACK;
                for s in 0..samples_per_pixel {
                    let u = (i as f64 + rand.gen_range(0.0..1.0)) / (image_width - 1) as f64;
                    let v = (j as f64 + rand.gen_range(0.0..1.0)) / (image_height - 1) as f64;
                    let r = cam.get_ray(u, v);
                    pixel_color += self.ray_color(&r, &world, max_depth);
                }
                pixel_color.write_color(&mut stdout(), samples_per_pixel);
            }
        }
    }
}
fn main() {
    Raycaster::new().main()
}