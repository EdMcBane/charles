use core::cell::UnsafeCell;
use std::rc::Rc;
use std::thread_local;
use rand::rngs::SmallRng;
use rand::{Error, RngCore};
use rand::SeedableRng;

#[derive(Clone, Debug)]
pub struct MyRng {
    // Rc is explictly !Send and !Sync
    rng: Rc<UnsafeCell<SmallRng>>,
}

thread_local!(
    // We require Rc<..> to avoid premature freeing when thread_rng is used
    // within thread-local destructors. See #968.
    static THREAD_RNG_KEY: Rc<UnsafeCell<SmallRng>> = {
        let rng = SmallRng::from_seed([0u8;32]);
        Rc::new(UnsafeCell::new(rng))
    }
);

#[cfg_attr(doc_cfg, doc(cfg(all(feature = "std", feature = "std_rng"))))]
pub fn my_rng() -> MyRng {
    let rng = THREAD_RNG_KEY.with(|t| t.clone());
    MyRng { rng }
}

impl Default for MyRng {
    fn default() -> MyRng {
        my_rng()
    }
}

impl RngCore for MyRng {
    #[inline(always)]
    fn next_u32(&mut self) -> u32 {
        // SAFETY: We must make sure to stop using `rng` before anyone else
        // creates another mutable reference
        let rng = unsafe { &mut *self.rng.get() };
        rng.next_u32()
    }

    #[inline(always)]
    fn next_u64(&mut self) -> u64 {
        // SAFETY: We must make sure to stop using `rng` before anyone else
        // creates another mutable reference
        let rng = unsafe { &mut *self.rng.get() };
        rng.next_u64()
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        // SAFETY: We must make sure to stop using `rng` before anyone else
        // creates another mutable reference
        let rng = unsafe { &mut *self.rng.get() };
        rng.fill_bytes(dest)
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Error> {
        // SAFETY: We must make sure to stop using `rng` before anyone else
        // creates another mutable reference
        let rng = unsafe { &mut *self.rng.get() };
        rng.try_fill_bytes(dest)
    }
}