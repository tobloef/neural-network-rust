use rand::{Rng, RngCore};
use rand_pcg::Pcg64;
use rand_seeder::Seeder;

pub fn create_random_seed() -> String {
    let mut rng = rand::thread_rng();
    rng.gen::<u32>().to_string()
}

pub fn seeded_rng(seed: &str) -> Box<dyn RngCore> {
    Box::new(Seeder::from(seed).make_rng::<Pcg64>())
}
