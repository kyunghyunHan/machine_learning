use ndarray::prelude::*;
use smartcore::metrics::*;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::neighbors::knn_classifier::*;
use smartcore::model_selection::train_test_split;
use ndarray::concatenate;
use ndarray::stack;

pub fn main(){
    let bream_length = arr1(&[
        25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
        31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
        35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0
    ]);

    let bream_weight = arr1(&[
        242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
        500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
        700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0
    ]);

    // 빙어 데이터
    let smelt_length = arr1(&[
        9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0
    ]);

    let smelt_weight = arr1(&[
        6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9
    ]);
    let length: ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 1]>>=concatenate![Axis(0), bream_length, smelt_length];
    let weight: ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 1]>>=concatenate![Axis(0), bream_weight, smelt_weight];

    let  fish_data: ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 2]>> = stack![Axis(1), length, weight];

    println!("{}",fish_data);

    let mut fish_target: Vec<i32> = vec![1; 35];
    // let rust_array: Vec<f64> = fish_data.into_raw_vec();
    fish_target.extend(vec![0; 14]);
    let mut data: Vec<Vec<_>> = Vec::new();
    for row in fish_data.outer_iter() {
        let row_vec: Vec<_> = row.iter().cloned().collect();
        data.push(row_vec);
    }
    let fish_dense_data: DenseMatrix<f64> = DenseMatrix::from_2d_vec(&data);
    // DenseMatrix를 초기화합니다.
    let (train_input, test_input, tarin_target, test_target) = train_test_split(&fish_dense_data, &fish_target, 0.2,true,None);

    // 2차원 배열의 슬라이스로 DenseMatrix를 초기화합니다.
    let knn= KNNClassifier::fit(&train_input, &tarin_target, Default::default()).unwrap();
    let y_pred = knn.predict(&test_input).unwrap();

    let acc: f64 = ClassificationMetricsOrd::accuracy().get_score(&test_target, &y_pred);
    println!("{:?}",knn.predict(&test_input).unwrap());
  
    }