use std::fs::File;
use ndarray::{Axis, iter::Axes,concatenate,stack};
use polars::prelude::*;
use polars_core::prelude::*;
use polars::prelude::{CsvWriter, CsvReader,DataFrame, NamedFrom, SerWriter, Series};
use polars_io::prelude::*;
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
                0 as u32
            }else if name =="Iris-virginica"{
                1
            }else{
                2
            })
         })
        .collect::<UInt32Chunked>()
        .into_series()
}
iris_df.apply("Species", replace).unwrap();

    
println!("{}", &iris_df);
}