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
let a_as_vec: Vec<i32> = d.iterator(0).map(|&x| x as i32).collect();

println!("{:?}",d);
let survived_series = Series::new("Species", y_pred.into_iter().collect::<Vec<i32>>());


let passenger_id_series = Series::new("Id", a_as_vec.into_iter().collect::<Vec<i32>>());


let mut df: DataFrame = DataFrame::new(vec![passenger_id_series, survived_series]).unwrap();

let mut output_file: File = File::create("./datasets/iris/out.csv").unwrap();

CsvWriter::new(&mut output_file)
    .has_header(true)
    .finish(&mut df)
    .unwrap();


}
