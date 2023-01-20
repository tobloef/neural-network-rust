use std::ops::Mul;

use ndarray::{arr1, arr2, arr3, s, Array, Array1, Array2, Array3, ArrayView2};

pub fn rotate_by_180_degrees(input: &ArrayView2<f32>) -> Array2<f32> {
    let mut rotated = input.to_owned();
    rotated.swap_axes(0, 1);
    rotated.reversed_axes()
}

pub fn valid_correlate(input: &ArrayView2<f32>, kernel: &ArrayView2<f32>) -> Array2<f32> {
    let input_shape = [input.shape()[0], input.shape()[1]];
    let kernel_shape = [kernel.shape()[0], kernel.shape()[1]];

    let output_shape = valid_correlate_output_shape(&input_shape, &kernel_shape);

    Array::from_shape_fn((output_shape[0], output_shape[1]), |(x, y)| {
        let slice = input.slice(s![x..x + kernel.shape()[0], y..y + kernel.shape()[1]]);

        slice.mul(kernel).sum()
    })
}

pub fn pad_by_kernel_size(input: &ArrayView2<f32>, kernel_shape: [usize; 2]) -> Array2<f32> {
    let input_shape = [input.shape()[0], input.shape()[1]];

    let output_shape = full_correlate_output_shape(&input_shape, &kernel_shape);

    let mut padded_slice = Array::zeros(output_shape);

    let coords_on_slice = [kernel_shape[0] - 1, kernel_shape[1] - 1];

    padded_slice
        .slice_mut(s![
            coords_on_slice[0]..coords_on_slice[0] + input.shape()[0],
            coords_on_slice[1]..coords_on_slice[1] + input.shape()[1]
        ])
        .assign(input);

    padded_slice
}

pub fn full_correlate(input: &ArrayView2<f32>, kernel: &ArrayView2<f32>) -> Array2<f32> {
    let kernel_shape: [usize; 2] = [kernel.shape()[0], kernel.shape()[1]];

    let padded_input = pad_by_kernel_size(input, kernel_shape);

    valid_correlate(&padded_input.view(), kernel)
}

pub fn valid_convolute(input: &ArrayView2<f32>, kernel: &ArrayView2<f32>) -> Array2<f32> {
    let rotated_kernel = rotate_by_180_degrees(kernel);

    valid_correlate(input, &rotated_kernel.view())
}

pub fn full_convolute(input: &ArrayView2<f32>, kernel: &ArrayView2<f32>) -> Array2<f32> {
    let rotated_kernel = rotate_by_180_degrees(kernel);

    full_correlate(input, &rotated_kernel.view())
}

fn layer() {
    let input: Array2<f32> = arr2(&[[1., 2.], [3., 4.], [5., 6.]]);

    // Layer 1

    let kernel1: Array2<f32> = arr2(&[[1., 2.], [3., 4.]]);
    let bias1: f32 = 1.;

    let kernel2: Array2<f32> = arr2(&[[1., 2.], [3., 4.]]);
    let bias2: f32 = 1.;

    let kernels = stack_2d_arrays(vec![kernel1, kernel2]);
    let biases: Array1<f32> = arr1(&[bias1, bias2]);

    let input_shape: [usize; 2] = [input.shape()[0], input.shape()[1]];
    let kernel_shape: [usize; 2] = [kernels.shape()[1], kernels.shape()[2]];

    let output_shape = valid_correlate_output_shape(&input_shape, &kernel_shape);

    let outputs: Array3<f32> = valid_correlate_many(&input.view(), &kernels, &biases);

    // Max pooling
    let max_pooling_outputs = max_pooling_many(&outputs, [2, 2]);

    // Layer 2
    // TODO
}

fn max_pooling_many(inputs: &Array3<f32>, kernel_shape: [usize; 2]) -> Array3<f32> {
    let input_shape = [inputs.shape()[1], inputs.shape()[2]];

    let output_shape = valid_correlate_output_shape(&input_shape, &kernel_shape);

    inputs.outer_iter().fold(
        Array::zeros((inputs.shape()[0], output_shape[0], output_shape[1])),
        |mut outputs, input| {
            for (i, mut output) in outputs.outer_iter_mut().enumerate() {
                output.assign(&max_pooling(&input, kernel_shape));
            }
            outputs
        },
    )
}

fn max_pooling(input: &ArrayView2<f32>, kernel_size: [usize; 2]) -> Array2<f32> {
    let input_shape = [input.shape()[0], input.shape()[1]];

    let output_shape = valid_correlate_output_shape(&input_shape, &kernel_size);

    Array::from_shape_fn((output_shape[0], output_shape[1]), |(x, y)| {
        let slice = input.slice(s![x..x + kernel_size[0], y..y + kernel_size[1]]);

        slice.fold(f32::MIN, |acc, &x| acc.max(x))
    })
}

fn valid_correlate_many(
    input: &ArrayView2<f32>,
    kernels: &Array3<f32>,
    biases: &Array1<f32>,
) -> Array3<f32> {
    let input_shape = [input.shape()[0], input.shape()[1]];
    let kernels_shape = [kernels.shape()[1], kernels.shape()[2]];

    let output_shape = valid_correlate_output_shape(&input_shape, &kernels_shape);

    kernels.outer_iter().fold(
        Array::zeros((kernels.shape()[0], output_shape[0], output_shape[1])),
        |mut outputs, kernel| {
            for (i, mut output) in outputs.outer_iter_mut().enumerate() {
                output.assign(&valid_correlate(&input.view(), &kernel.view()));
                output += biases[i];
            }
            outputs
        },
    )
}

fn valid_correlate_output_shape(input_shape: &[usize; 2], kernel_shape: &[usize; 2]) -> [usize; 2] {
    [
        input_shape[0] - kernel_shape[0] + 1,
        input_shape[1] - kernel_shape[1] + 1,
    ]
}

fn full_correlate_output_shape(input_shape: &[usize; 2], kernel_shape: &[usize; 2]) -> [usize; 2] {
    [
        input_shape[0] + (kernel_shape[0] - 1) * 2,
        input_shape[1] + (kernel_shape[1] - 1) * 2,
    ]
}

// Try not to use this, as vectors need to be allocated on the heap
fn stack_2d_arrays(arrays: Vec<Array2<f32>>) -> Array3<f32> {
    let shape = arrays[0].shape();
    for array in arrays.iter() {
        assert_eq!(array.shape(), shape);
    }

    let mut stacked = Array::zeros((arrays.len(), shape[0], shape[1]));
    for (i, array) in arrays.iter().enumerate() {
        stacked.slice_mut(s![i, .., ..]).assign(array);
    }
    stacked
}

fn stack_1d_arrays(arrays: Vec<Array1<f32>>) -> Array2<f32> {
    let shape = arrays[0].shape();
    for array in arrays.iter() {
        assert_eq!(array.shape(), shape);
    }

    let mut stacked = Array::zeros((arrays.len(), shape[0]));
    for (i, array) in arrays.iter().enumerate() {
        stacked.slice_mut(s![i, ..]).assign(array);
    }
    stacked
}

#[cfg(test)]
mod tests {
    use ndarray::{arr2, aview2};

    use super::*;

    #[test]
    fn test_rotate_by_180_degrees() {
        let input = aview2(&[[1., 2.], [-1., 0.]]);

        let expected_output = arr2(&[[1., 2.], [-1., 0.]]);

        let actual_output = rotate_by_180_degrees(&input);

        assert_eq!(actual_output, expected_output);
    }

    #[test]
    fn test_valid_correlate() {
        let input = aview2(&[[1., 6., 2.], [5., 3., 1.], [7., 0., 4.]]);

        let kernel = aview2(&[[1., 2.], [-1., 0.]]);

        let expected_output = arr2(&[[8., 7.], [4., 5.]]);

        let actual_output = valid_correlate(&input, &kernel);

        assert_eq!(actual_output, expected_output);
    }

    #[test]
    fn test_pad_by_kernel_size() {
        assert_eq!(
            pad_by_kernel_size(&aview2(&[[1., 2.], [3., 4.]]), [1, 1]),
            arr2(&[[1., 2.], [3., 4.],])
        );

        assert_eq!(
            pad_by_kernel_size(&aview2(&[[1., 2.], [3., 4.]]), [2, 2]),
            arr2(&[
                [0., 0., 0., 0.],
                [0., 1., 2., 0.],
                [0., 3., 4., 0.],
                [0., 0., 0., 0.],
            ])
        );

        assert_eq!(
            pad_by_kernel_size(&aview2(&[[1., 2.], [3., 4.]]), [3, 3]),
            arr2(&[
                [0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0.],
                [0., 0., 1., 2., 0., 0.],
                [0., 0., 3., 4., 0., 0.],
                [0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0.],
            ])
        );
    }

    #[test]
    fn test_full_correlate() {
        let input = aview2(&[[1., 6., 2.], [5., 3., 1.], [7., 0., 4.]]);

        let kernel = aview2(&[[1., 2.], [-1., 0.]]);

        let expected_output = arr2(&[
            [0., -1., -6., -2.],
            [2., 8., 7., 1.],
            [10., 4., 5., -3.],
            [14., 7., 8., 4.],
        ]);

        let actual_output = full_correlate(&input, &kernel);

        assert_eq!(actual_output, expected_output);
    }

    #[test]
    fn test_valid_convolute() {
        let input = aview2(&[[1., 6., 2.], [5., 3., 1.], [7., 0., 4.]]);

        let kernel = aview2(&[[1., 2.], [-1., 0.]]);

        let expected_output = arr2(&[[7., 5.], [11., 3.]]);

        let actual_output = valid_convolute(&input, &kernel);

        assert_eq!(actual_output, expected_output);
    }
}
