use polars::prelude::*;
use linfa::prelude::*;
use ndarray::prelude::*;
use ndarray::concatenate;
use ndarray::stack;

pub fn main(){
    let df = CsvReader::from_path("./datasets/iris.csv")
    .unwrap()
    .finish()
    .unwrap();
//SepalLengthCm가 5이상인 것들
let mask= df.column("SepalLengthCm").unwrap().f64().unwrap().gt(5.0);
//filter
let df_small = df.filter(&mask).unwrap();
//SepalWidthCm와 PetalLengthCm만 출력
let df_agg = df_small
    .select(["SepalWidthCm","PetalLengthCm"])
    .unwrap();
// println!("{}", df_agg);
//row
println!("{}", df.height());
//col
println!("{}", df.width());
//ndarray 로 변환
let ndarray: ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 2]>> = df_agg.to_ndarray::<Float64Type>(IndexOrder::Fortran).unwrap();
// println!("{}", ndarray);
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
let length: ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 1]>>=concatenate![Axis(0), bream_length, smelt_length];
let weight: ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 1]>>=concatenate![Axis(0), bream_weight, smelt_weight];

let  mut fish_data = stack![Axis(1), length, weight];

let arr: ArrayBase<ndarray::OwnedRepr<i32>, Dim<[usize; 1]>>= arr1(&[1,1,1,1,1]);
let dataset= Dataset::new(fish_data,arr);



}
