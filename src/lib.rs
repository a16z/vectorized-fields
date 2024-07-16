use std::{fs::{File, OpenOptions}, io::{BufReader, BufWriter, Read}};

use ark_ff::PrimeField;

pub fn save_vec_to_file<F: PrimeField>(file_name: &str, vec: &[F]) {
    let file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true) // This ensures the file is always replaced if it exists
        .open(file_name)
        .expect("Unable to create or open file");
    let mut writer = BufWriter::new(file);

    for element in vec {
        element.serialize_compressed(&mut writer).expect("should serialize");
    }
}

pub fn read_vec_from_file<F: PrimeField>(file_name: &str) -> Vec<F> {

    let file = File::open(file_name).expect("Unable to open file");
    let mut reader = BufReader::new(file);
    let mut buffer = Vec::new();

    reader.read_to_end(&mut buffer).expect("Unable to read data");

    let mut deserializer = buffer.as_slice();
    let mut vec = Vec::new();
    while let Ok(element) = F::deserialize_compressed(&mut deserializer) {
        vec.push(element);
    }
    vec
}