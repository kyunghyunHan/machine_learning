use std::fs::File;
use ndarray::{Axis, iter::Axes,concatenate,stack,array,arr1,Array};
use ndarray::prelude::*;
use polars::prelude::*;
use polars_core::prelude::*;
use polars::prelude::{CsvWriter, CsvReader,DataFrame, NamedFrom, SerWriter, Series};
use polars_io::prelude::*;
use smartcore::linalg::basic::arrays::Array2;
use smartcore::neighbors::knn_classifier::*;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::model_selection::train_test_split;
use smartcore::linear::logistic_regression::LogisticRegression;
use smartcore::metrics::*;
use smartcore::svm::svc::{SVC,SVCParameters};
use smartcore::ensemble::random_forest_classifier::{RandomForestClassifier,RandomForestClassifierParameters};
use std::str;
use linfa::DatasetBase;
use linfa_linear::LinearRegression;
use linfa::traits::{Fit, Predict};
use linfa::prelude::*;
pub fn main(){
let  mut iris_df: DataFrame = CsvReader::from_path("./datasets/iris/iris.csv").unwrap().finish().unwrap();
println!("{}",iris_df);
println!("데이터 미리보기:{}",iris_df.head(None));
println!("데이터 정보 확인:{:?}",iris_df.schema());
println!("null count확인{}",iris_df.null_count());
/*데이터 아이리스 변환 */
fn replace(str_val: &Series) -> Series {
    str_val.utf8()
        .unwrap()
        .into_iter()
        .map(|opt_name: Option<&str>| {
            opt_name.map(|name: &str| if  name =="Iris-setosa" {
                0 as i64
            }else if name =="Iris-virginica"{
                1
            }else{
                2
            })
         })
        .collect::<Int64Chunked>()
        .into_series()
}
iris_df.apply("Species", replace).unwrap();
//데이터 나눈기

let  data= iris_df.drop("Species").unwrap();//훈련데이터
let target= iris_df.column("Species").unwrap();//타겟데이터
println!("{}",data);



let data= data.to_ndarray::<Float64Type>(IndexOrder::Fortran).unwrap();

let mut x_train: Vec<Vec<_>> = Vec::new();
for row in data.outer_iter() {
    let row_vec: Vec<_> = row.iter().cloned().collect();
    x_train.push(row_vec);
}
let x_train: DenseMatrix<f64> = DenseMatrix::from_2d_vec(&x_train);

let y_train: Vec<i64> = target.i64().unwrap().into_no_null_iter().collect();
let y_train:Vec<i32>= y_train.iter().map(|x|*x as i32).collect();

let (train_input, test_input, tarin_target, test_target) = train_test_split(&x_train, &y_train, 0.2,true,None);


println!("{}",train_input);

let logreg= LogisticRegression::fit(&train_input, &tarin_target, Default::default()).unwrap();
let y_pred: Vec<i32> = logreg.predict(&test_input).unwrap();
let acc: f64 = ClassificationMetricsOrd::accuracy().get_score(&test_target, &y_pred);
println!("{:?}",acc);

//파일저장
let mut d:Vec<f64>= vec![];
 let d= test_input.copy_col_as_vec(1, &mut d);
let d= test_input.get_col(0);
println!("{:?}",d);
let dense_matrix = DenseMatrix::from_2d_array(&[
    &[1.0, 2.0, 3.0],
    &[4.0, 5.0, 6.0],
]);




let survived_series = Series::new("Species", y_pred.into_iter().collect::<Vec<i32>>());

// let mut df: DataFrame = DataFrame::new(vec![passenger_id_series, survived_series]).unwrap();

// let mut output_file: File = File::create("./datasets/iris/out.csv").unwrap();

// CsvWriter::new(&mut output_file)
//     .has_header(true)
//     .finish(&mut df)
//     .unwrap();
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

let dataset = Dataset::new(fish_data, array![1., 2.]).with_feature_names(vec!["test 1", "test 2"]);
let (train, test) = dataset.split_with_ratio(0.5);

let model = LinearRegression::default().fit(&train).unwrap();
let predictions = model.predict(test.records());
println!("{}",predictions);
let  mut fish_data = stack![Axis(1), length, weight];
let mut fish_target: Vec<i32> = vec![1; 35];
// let rust_array: Vec<f64> = fish_data.into_raw_vec();
fish_target.extend(vec![0; 14]);

let fish_target = Array::from(fish_target);
let dataset: DatasetBase<ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<ndarray::OwnedRepr<i32>, Dim<[usize; 1]>>>= Dataset::new(fish_data,fish_target);
let (train, test) = dataset.split_with_ratio(0.8);

}
