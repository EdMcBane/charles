const WIDTH: usize = 256;
const HEIGHT: usize = 256;

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
