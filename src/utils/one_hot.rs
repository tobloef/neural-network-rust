use ndarray::{Array1, ArrayView1};

pub fn n_hot_to_indexes(n_hot: ArrayView1<f32>, hot_count: usize) -> Vec<usize> {
    let mut indexes = Vec::new();
    let mut n_hot = n_hot.to_owned();

    let min_index = n_hot
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    let min_value = n_hot[[min_index]];

    for _ in 0..hot_count {
        let mut max_index = 0;
        let mut max_value = n_hot[[0]];

        for (index, value) in n_hot.iter().enumerate() {
            if *value > max_value {
                max_index = index;
                max_value = *value;
            }
        }

        indexes.push(max_index);
        n_hot[max_index] = min_value;
    }

    indexes
}

pub fn indexes_to_n_hot(indexes: Vec<usize>, output_size: usize) -> Array1<f32> {
    let mut n_hot = Array1::zeros(output_size);
    for index in indexes {
        n_hot[index] = 1.0;
    }
    n_hot
}

pub fn output_to_n_hot(output: ArrayView1<f32>, hot_count: usize) -> Array1<f32> {
    let indexes = n_hot_to_indexes(output, hot_count);
    indexes_to_n_hot(indexes, output.len())
}
