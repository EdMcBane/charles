#![feature(total_cmp)]

mod rng;

use std::ops::{Neg, Index, AddAssign, MulAssign, DivAssign, IndexMut, Add, Sub, Mul, Div, Range};
use std::fmt::{Display, Formatter};
use std::io::{stdout, Write};
use rand::distributions::Uniform;
use rand::distributions::Distribution;
use rayon::prelude::*;
use std::sync::Arc;
use crate::rng::my_rng;
use rand::Rng;

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
    const ZERO_THRESH: f64 = 1e-8;
    fn near_zero(&self) -> bool {
        self.0.into_iter().all(|e| e.abs() < Vec3::ZERO_THRESH)
    }

    fn reflect(&self, n: &Vec3) -> Self {
        *self - 2. * Vec3::dot(self, n) * (*n)
    }

    fn refract(&self, n: &Vec3, etai_over_etat: f64) -> Self {
        let cos_theta = Vec3::dot(&-*self, n).min(1.0);
        let r_out_perp = etai_over_etat * (*self + cos_theta * *n);
        let r_out_parallel = -(1.0 - r_out_perp.length_squared()).abs().sqrt() * *n;
        r_out_perp + r_out_parallel
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

const COLOR_WHITE: Color = Vec3([1., 1., 1.]);
const COLOR_BLACK: Color = Vec3([0., 0., 0.]);


struct HitRecord {
    p: Point3,
    normal: Vec3,
    mat: Arc<dyn Material + Sync>,
    t: f64,
    front_face: bool,
}

impl HitRecord {
    fn new(t: f64, p: Point3, r: &Ray, outward_normal: Vec3, mat: Arc<dyn Material + Sync>) -> Self {
        let front_face = Vec3::dot(&r.dir, &outward_normal) < 0.;
        HitRecord {
            t,
            p,
            mat,
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
    mat: Arc<dyn Material + Sync + Send>,
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
                HitRecord::new(root, p, r, (p - self.center) / self.radius, self.mat.clone())
            })
    }
}

impl Hittable for &Vec<Box<dyn Hittable + Sync>> {
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
    caster: Raycaster,
    origin: Point3,
    lower_left_corner: Point3,
    horizontal: Vec3,
    vertical: Vec3,
    u: Vec3,
    v: Vec3,
    _w: Vec3,
    lens_radius: f64,
}

impl Camera {
    fn new(caster: Raycaster, look_from: Point3, look_at: Point3, vup: Vec3, vfov: f64, aspect_ratio: f64, aperture: f64, focus_dist: f64) -> Self {
        let theta = degrees_to_radians(vfov);
        let h = (theta / 2.0).tan();
        let viewport_height = 2.0 * h;
        let viewport_width = aspect_ratio * viewport_height;

        let w = (look_from - look_at).unit_vector();
        let u = vup.cross(w).unit_vector();
        let v = w.cross(u);

        let origin = look_from;
        let horizontal = focus_dist * viewport_width * u;
        let vertical = focus_dist * viewport_height * v;
        Camera {
            caster,
            origin,
            horizontal,
            vertical,
            lower_left_corner: origin - horizontal / 2. - vertical / 2. - focus_dist * w,
            u,
            v,
            _w: w,
            lens_radius: aperture / 2.,
        }
    }
    fn get_ray(&self, s: f64, t: f64) -> Ray {
        let disk_pos = self.lens_radius * self.caster.random_in_unit_disk();
        let offset = self.u * disk_pos.x() + self.v * disk_pos.y();
        Ray {
            origin: self.origin + offset,
            dir: self.lower_left_corner + s * self.horizontal + t * self.vertical - self.origin - offset,
        }
    }
}

enum Scatter {
    Absorbed,
    Scattered {
        ray: Ray,
        attenuation: Color,
    },
}

trait Material {
    fn scatter(&self, r_in: &Ray, rec: &HitRecord) -> Scatter;
}

struct Lambertian {
    caster: Raycaster,
    albedo: Color,
}

impl Lambertian {
    fn new(caster: Raycaster, albedo: Color) -> Self {
        Lambertian {
            caster,
            albedo,
        }
    }
}

impl Material for Lambertian {
    fn scatter(&self, _r_in: &Ray, rec: &HitRecord) -> Scatter {
        let scatter_direction = rec.normal + self.caster.random_unit_vector();
        Scatter::Scattered {
            ray: Ray {
                origin: rec.p,
                dir: if !scatter_direction.near_zero() { scatter_direction } else { rec.normal }, // Avoid zero vec and NaNs
            },
            attenuation: self.albedo,
        }
    }
}

struct Metal {
    caster: Raycaster,
    albedo: Color,
    fuzz: f64,
}

impl Metal {
    fn new(caster: Raycaster, albedo: Color, fuzz: f64) -> Self {
        assert!(fuzz <= 1.);
        Metal {
            caster,
            albedo,
            fuzz,
        }
    }
}

impl Material for Metal {
    fn scatter(&self, r_in: &Ray, rec: &HitRecord) -> Scatter {
        let reflected = r_in.dir.unit_vector().reflect(&rec.normal);
        let scattered = reflected + self.fuzz * self.caster.random_in_unit_sphere();
        if Vec3::dot(&scattered, &rec.normal) <= 0. {
            Scatter::Absorbed
        } else {
            Scatter::Scattered {
                ray: Ray {
                    origin: rec.p,
                    dir: scattered,
                },
                attenuation: self.albedo,
            }
        }
    }
}

struct VantaBlack;

impl Material for VantaBlack {
    fn scatter(&self, _r_in: &Ray, _rec: &HitRecord) -> Scatter {
        Scatter::Absorbed
    }
}

struct Dielectric {
    ir: f64,
}

impl Dielectric {
    fn new(ir: f64) -> Self {
        Self {
            ir,
        }
    }

    fn reflectance(&self, cosine: f64, ref_idx: f64) -> f64 {
        // Schlick's approximation
        let r0 = ((1.0 - ref_idx) / (1. + ref_idx)).powi(2);
        r0 + (1. - r0) * (1.0 - cosine).powi(5)
    }
}

impl Material for Dielectric {
    fn scatter(&self, r_in: &Ray, rec: &HitRecord) -> Scatter {
        let refraction_ratio = if rec.front_face { 1.0 / self.ir } else { self.ir };
        let unit_direction = r_in.dir.unit_vector();
        let cos_theta = Vec3::dot(&-unit_direction, &rec.normal).min(1.0);
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
        let cannot_refract = refraction_ratio * sin_theta > 1.0;
        let refracted = if cannot_refract || self.reflectance(cos_theta, refraction_ratio) > my_rng().gen_range(0.0..1.0) {
            unit_direction.reflect(&rec.normal)
        } else {
            unit_direction.refract(&rec.normal, refraction_ratio)
        };
        Scatter::Scattered {
            ray: Ray {
                origin: rec.p,
                dir: refracted,
            },
            attenuation: COLOR_WHITE,
        }
    }
}

struct Raycaster {
    unit: Uniform<f64>,
}

impl Raycaster {
    fn new() -> Self {
        Raycaster {
            unit: Uniform::from(-1.0..1.0),
        }
    }
    fn random_in_unit_sphere(&self) -> Vec3 {
        loop {
            let mut rng = my_rng();
            let p = Vec3::new(
                self.unit.sample(&mut rng),
                self.unit.sample(&mut rng),
                self.unit.sample(&mut rng));
            if p.length_squared() < 1. {
                return p;
            }
        }
    }
    fn random(&self) -> Vec3 {
        let mut rng = my_rng();
        Vec3::new(
            self.unit.sample(&mut rng),
            self.unit.sample(&mut rng),
            self.unit.sample(&mut rng),
        )
    }

    fn random_in_range(&self, range: Range<f64>) -> Vec3 {
        let mut rng = my_rng();
        Vec3::new(
            my_rng().gen_range(range.clone()),
            my_rng().gen_range(range.clone()),
            my_rng().gen_range(range),
        )
    }

    fn random_in_unit_disk(&self) -> Vec3 {
        let mut rng = my_rng();
        loop {
            let p = Vec3::new(
                self.unit.sample(&mut rng),
                self.unit.sample(&mut rng),
                0.);
            if p.length_squared() < 1. {
                return p;
            }
        }
    }

    fn random_unit_vector(&self) -> Vec3 {
        self.random_in_unit_sphere().unit_vector()
    }
    #[allow(dead_code)]
    fn random_in_hemisphere(&mut self, normal: &Vec3) -> Vec3 {
        let in_unit_sphere = self.random_in_unit_sphere();
        if Vec3::dot(&in_unit_sphere, normal) > 0. {
            in_unit_sphere
        } else {
            -in_unit_sphere
        }
    }

    fn ray_color<W: Hittable>(&self, r: &Ray, world: W, depth: usize) -> Color {
        if depth == 0 {
            return COLOR_BLACK;
        }
        let range = 0.001f64..f64::INFINITY;
        if let Some(rec) = world.hit(r, &range) {
            match rec.mat.scatter(r, &rec) {
                Scatter::Absorbed => return COLOR_BLACK,
                Scatter::Scattered {
                    ray,
                    attenuation
                } => {
                    return attenuation * self.ray_color(&ray, world, depth - 1);
                }
            }
        }
        let unit_dir = r.dir.unit_vector();
        let t = 0.5 * (unit_dir.y() + 1.0);
        (1.0 - t) * COLOR_WHITE + t * Color::new(0.5, 0.7, 1.0)
    }

    fn main(self: &Raycaster) {
        // Image
        let aspect_ratio = 3.0 / 2.0;
        let image_width = 1200usize;
        let image_height = (image_width as f64 / aspect_ratio) as usize;
        let samples_per_pixel = 500;
        let max_depth = 50;

        // World
        let world = self.random_scene();

        // Camera
        let look_from = Point3::new(13., 2., 3.);
        let look_at = Point3::new(0., 0., 0.);
        let focus_dist = (look_from - look_at).length();
        let cam = Camera::new(
            Raycaster::new(),
            look_from,
            look_at,
            Vec3::new(0., 1., 0.),
            20.,
            aspect_ratio,
            0.1,
            10.,
        );

        // Render
        rayon::ThreadPoolBuilder::new().num_threads(6).build_global().unwrap();
        println!("P3");
        println!("{} {}", image_width, image_height);
        println!("{}", u8::MAX);
        let mut output: Vec<Vec<Color>> = Vec::new();
        output.par_extend(
            (0..image_height).into_par_iter().rev()
            .map(|j| {
                let mut rand = my_rng();
                (0..image_width).into_iter().map(|i| {
                    let mut pixel_color = COLOR_BLACK;
                    for _ in 0..samples_per_pixel {
                        let u = (i as f64 + rand.gen_range(0.0..1.0)) / (image_width - 1) as f64;
                        let v = (j as f64 + rand.gen_range(0.0..1.0)) / (image_height - 1) as f64;
                        let r = cam.get_ray(u, v);
                        pixel_color += self.ray_color(&r, &world, max_depth);
                    }
                    pixel_color
                }).collect()
            }));
        for line in output {
            for pixel_color in line {
                pixel_color.write_color(&mut stdout(), samples_per_pixel);
            }
        }
    }

    fn random_scene(&self) -> Vec<Box<dyn Hittable + Sync>> {
        let mut world: Vec<Box<dyn Hittable + Sync>> = Vec::new();
        let material_ground = Arc::new(Lambertian::new(Raycaster::new(), Color::new(0.5, 0.5, 0.5)));
        world.push(Box::new(Sphere {
            center: Point3::new(0., -1000., 0.),
            radius: 1000.,
            mat: material_ground.clone(),
        }));

        for a in -11..11 {
            for b in -11..11 {
                let center = Point3::new(
                    a as f64 + 0.9 * my_rng().gen_range(0.0..1.0),
                    0.2,
                    b as f64 + 0.9 * my_rng().gen_range(0.0..1.0));
                if (center - Point3::new(4.,0.2, 0.)).length() < 0.9 {
                    continue;
                }
                let sphere_material: Arc<dyn Material + Sync + Send> = match my_rng().gen_range::<f64, Range<f64>>(0.0..1.0) {
                    f if f < 0.8 =>  {
                        let albedo =  self.random() * self.random();
                        Arc::new(Lambertian::new(Raycaster::new(), albedo))
                    },
                    f if f < 0.95 => {
                        let albedo = self.random_in_range(0.5..1.0);
                        let fuzz = my_rng().gen_range(0.0..0.5);
                        Arc::new(Metal::new(Raycaster::new(), albedo, fuzz))
                    },
                    _ => {
                        Arc::new(Dielectric::new(1.5))
                    },
                };
                world.push(Box::new(Sphere {
                    center,
                    radius: 0.2,
                    mat: sphere_material
                }))
            }
        }
        world.push(Box::new(Sphere{
            center: Point3::new(0.,1.,0.),
            radius: 1.0,
            mat: Arc::new( Dielectric::new(1.5)),
        }));
        world.push(Box::new(Sphere{
            center: Point3::new(-4.,1.,0.),
            radius: 1.0,
            mat: Arc::new( Lambertian::new(Raycaster::new(), Color::new(0.4, 0.2, 0.1))),
        }));
        world.push(Box::new(Sphere{
            center: Point3::new(4.,1.,0.),
            radius: 1.0,
            mat: Arc::new( Metal::new(Raycaster::new(), Color::new(0.7,0.6,0.5), 0.0)),
        }));
        world
    }
}

fn main() {
    Arc::new(Raycaster::new()).main()
}