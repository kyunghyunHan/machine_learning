use plotters::prelude::*;
use ndarray::{array, stack, s,Array1, Array2};
use ndarray::Axis;
use ndarray::Array;
use linfa::traits::Fit;
use linfa::prelude::*;
use linfa_linear::LinearRegression;
use linfa::Float;
use ndarray::ArrayBase;
use ndarray::OwnedRepr;
use ndarray::Dim;
// use smartcore::numbers::realnum::RealNumber;
// use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::neighbors::knn_classifier::*;
use smartcore::neighbors::knn_regressor::{KNNRegressor, KNNRegressorParameters};
use smartcore::neighbors::knn_classifier::KNNClassifier;
// use smartcore::metrics::distance::Distances;
use smartcore::model_selection::train_test_split;
use smartcore::metrics::roc_auc_score;
use smartcore::metrics::completeness_score;

fn main() {
    // 도미 데이터
    let bream_length = array![
        25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
        31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
        35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0
    ];

    let bream_weight = array![
        242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
        500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
        700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0
    ];

    // 빙어 데이터
    let smelt_length = array![
        9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0
    ];

    let smelt_weight = array![
        6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9
    ];

    // Create a scatter plot
    let root = BitMapBackend::new("scatter_plot.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let x_range = 0.0..45.0; // Adjust the x-axis range based on your data
    let y_range = 0.0..1050.0; // Adjust the y-axis range based on your data

    let mut chart = ChartBuilder::on(&root)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(x_range, y_range)
        .unwrap();

    chart
        .configure_mesh()
        .x_desc("Length")
        .y_desc("Weight")
        .draw()
        .unwrap();

    chart
        .draw_series(
            bream_length
                .iter()
                .zip(bream_weight.iter())
                .map(|(x, y)| {
                    Circle::new((*x, *y), 5, Into::<ShapeStyle>::into(&RGBColor(255, 0, 0)))
                }),
        )
        .unwrap();

    chart
        .draw_series(
            smelt_length
                .iter()
                .zip(smelt_weight.iter())
                .map(|(x, y)| {
                    Circle::new((*x, *y), 5, Into::<ShapeStyle>::into(&RGBColor(0, 0, 255)))
                }),
        )
        .unwrap();

    // Prepare the data for K-nearest neighbors
    // let bream_data = array![bream_length, bream_weight].reversed_axes();
    // let smelt_data = array![smelt_length, smelt_weight].reversed_axes();

    // let fish_data = stack![Axis(0), &[bream_data, smelt_data]];

    // // Create the target labels
    // let fish_target = Array1::from_elem(35, 1.0).to_owned();
    // let smelt_target = Array1::from_elem(14, 0.0).to_owned();
    // let fish_target = stack![Axis(0), &[fish_target, smelt_target]];

    // // Create and train the K-nearest neighbors classifier

    // // Test the classifier
    // let test_data = array![[30.0, 600.0]].reversed_axes();
    // println!("{:?}",fish_target);

    let data: Vec<Vec<f64>> = bream_length.iter().zip(bream_weight.iter()).map(|(&l, &w)| vec![l, w]).collect();
    let num_rows = data.len();
    let num_cols = if num_rows > 0 { data[0].len() } else { 0 };
    let mut flat_data = Vec::new();
    for row in data.iter() {
        for &value in row.iter() {
            flat_data.push(value);
        }
    }
    let data: Vec<Vec<f64>> = bream_length.iter().zip(bream_weight.iter()).map(|(&l, &w)| vec![l, w]).collect();
    let data_slices: Vec<&[f64]> = data.iter().map(|row| row.as_slice()).collect();

    // Create a DenseMatrix from the 2D array
    let x: DenseMatrix<f64> = DenseMatrix::from_2d_array(&data_slices);  
        // Combine 'length' and 'weight' vectors into a 2D array
        let mut fish_data: Vec<[f64; 2]> = Vec::new();
        fish_data.extend(bream_length.iter().zip(bream_weight.iter()).map(|(&l, &w)| [l, w]));
        fish_data.extend(smelt_length.iter().zip(smelt_weight.iter()).map(|(&l, &w)| [l, w]));
    
        // Create 'fish_target' with 1s and 0s
        let mut fish_target: Vec<i32> = Vec::new();
        fish_target.extend(vec![1; bream_length.len()]);
        fish_target.extend(vec![0; smelt_length.len()]);
    
        // println!("{:?}", fish_data);
        // println!("{:?}", fish_target);
        let fish_data_2d: Vec<Vec<f64>> = fish_data.iter().map(|arr| arr.to_vec()).collect();
        let fish_target_vec: Vec<f64> = fish_target.iter().map(|&x| x as f64).collect();
    
        let x: DenseMatrix<f64> = DenseMatrix::from_2d_array(
            &fish_data_2d.iter().map(|row| row.as_slice()).collect::<Vec<_>>()
        );
        let y: Vec<f64> = fish_target_vec;
        // let bream_length: Array1<f64> = Array::from(vec![
        //     25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
        //     31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
        //     35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0,
        // ]);
        
        // let smelt_length: Array1<f64> = Array::from(vec![
        //     9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0,
        // ]);
        
        // let bream_weight: Array1<f64> = Array::from(vec![
        //     242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
        //     500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
        //     700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0,
        // ]);
        
        // let smelt_weight: Array1<f64> = Array::from(vec![
        //     6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9,
        // ]);
       // Prepare the data as DenseMatrix
let bream_length: Vec<f64> = vec![25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0];

let bream_weight: Vec<f64> = vec![242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0];

let smelt_length: Vec<f64> = vec![9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0];
let smelt_weight: Vec<f64> = vec![6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9];

// Prepare feature matrix and target vector
let mut data = vec![];
data.extend(bream_length.iter().map(|&x| vec![x, 1.0]));
data.extend(smelt_length.iter().map(|&x| vec![x, 0.0]));
let features: Vec<f64> = data.iter().flat_map(|x| x.clone()).collect();
let target: Vec<f64> = data.iter().map(|x| x[1]).collect();
let num_samples = data.len();
let num_features = data[0].len();



// Split the dataset into training and test sets
// let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true);

// Fit a KNN classifier
// let knn: KNNClassifier<f64, f64, DenseMatrix<f64>, Vec<f64>, smartcore::metrics::distance::euclidian::Euclidian<f64>> = KNNClassifier::fit(
// &x_train,
// &y_train,
// Default::default(),
// ).unwrap();

// Make predictions
// let y_hat_knn = knn.predict(&x_test);

// Calculate test error (AUC)
// let auc = roc_auc_score(&y_test, &y_hat_knn);
// println!("AUC: {}", auc);

let x: DenseMatrix<f32> = DenseMatrix::from_2d_array(&[
    &[25.4, 242.0],
     &[26.3, 290.0],
      &[26.5, 340.0],&[29.0, 363.0], &[29.0, 430.0],& [29.7, 450.0],& [29.7, 500.0], &[30.0, 390.0],& [30.0, 450.0], &[30.7, 500.0], &[31.0, 475.0], &[31.0, 500.0], &[31.5, 500.0], &[32.0, 340.0], &[32.0, 600.0], &[32.0, 600.0],& [33.0, 700.0],& [33.0, 700.0], &[33.5, 610.0],& [33.5, 650.0], &[34.0, 575.0], &[34.0, 685.0], &[34.5, 620.0], &[35.0, 680.0], &[35.0, 700.0], &[35.0, 725.0],& [35.0, 720.0],& [36.0, 714.0],&[36.0, 850.0], &[37.0, 1000.0], &[38.5, 920.0],& [38.5, 955.0], &[39.5, 925.0],& [41.0, 975.0], &[41.0, 950.0], &[9.8, 6.7],&[10.5, 7.5], &[10.6, 7.0],& [11.0, 9.7],&[11.2, 9.8],& [11.3, 8.7], &[11.8, 10.0],& [11.8, 9.9],& [12.0, 9.8],& [12.2, 12.2],& [12.4, 13.4], &[13.0, 12.2], &[14.3, 19.7],&[15.0, 19.9]]
    );

// let y:Vec<f32> =vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
let y:Vec<i32> =vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true,None);

let mut knn = KNNClassifier::fit(&x_train, &y_train, Default::default()).and_then(|knn| knn.predict(&x_test)).unwrap();    
let dd: ArrayBase<OwnedRepr<i32>, Dim<[usize; 1]>>= Array::from(knn);
// Calculate test error
let array1_f32: Array1<f64> = Array1::from(y_test.iter().map(|&x| x as f64).collect::<Vec<f64>>());
// let array2_f32: Array1<f64> = Array1::from(knn.iter().map(|&x| x as f64).collect::<Vec<f64>>());
let y_true: Vec<i32> = vec![0, 1, 0, 1, 1];
let y_pred_probabilities: Vec<f64> = vec![0.2, 0.8, 0.3, 0.7, 0.9];

// Vec를 ndarray::Array1로 변환
let y_true_array: Array1<f64> = Array::from(y_true.iter().map(|&x| x as f64).collect::<Vec<f64>>());
let y_pred_array: Array1<f64> = Array::from(y_pred_probabilities.clone());
// ROC AUC score 계산
let array1_f64: Vec<f64> = y_test.iter().map(|&x| x as f64).collect();
// let y_true: Vec<i32> = vec![0, 1, 0, 1, 1];

// Vec를 ndarray::Array1로 변환
let y_true_array: Vec<f64> = y_true.iter().map(|&x| x as f64).collect();
let y_pred_array: Vec<f64> = y_pred_probabilities.clone();

// ROC AUC score 계산
let auc = roc_auc_score(&y_true_array, &y_pred_array);

// 결과 출력
println!("ROC AUC: {:?}", auc);
// 결과 출력
}
// fn main(){
//     use smartcore::dataset::*;
// // DenseMatrix wrapper around Vec
// use smartcore::linalg::naive::dense_matrix::DenseMatrix;
// // Imports for KNN classifier
// use smartcore::neighbors::knn_classifier::KNNClassifier;
// // Model performance
// use smartcore::metrics::roc_auc_score;
// use smartcore::model_selection::train_test_split;
// // Load dataset
// let cancer_data = breast_cancer::load_dataset();
// // Transform dataset into a NxM matrix
// let x: DenseMatrix<f32> = DenseMatrix::from_array(
//     cancer_data.num_samples,
//     cancer_data.num_features,
//     &cancer_data.data,
// );
// // These are our target class labels
// let y = cancer_data.target;
// // Split dataset into training/test (80%/20%)
// let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true);
// // KNN classifier
// let y_hat_knn = KNNClassifier::fit(
//     &x_train,
//     &y_train,        
//     Default::default(),
// ).and_then(|knn| knn.predict(&x_test)).unwrap();    
// // Calculate test error
// println!("AUC: {}", roc_auc_score(&y_test, &y_hat_knn));
// }