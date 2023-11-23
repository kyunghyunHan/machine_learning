use std::fs::File;
use ndarray::{Axis, iter::Axes,concatenate,stack,array,arr1,Array};
use ndarray::prelude::*;
use polars::prelude::*;
use polars_core::prelude::*;
use polars::prelude::{CsvWriter, CsvReader,DataFrame, NamedFrom, SerWriter, Series};
use polars_io::prelude::*;
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
let target: Vec<i32> = target.iter().map(|x|x.to_string().parse::<i32>().unwrap()).collect();
let target=Array::from(target);
let data= data.to_ndarray::<Float64Type>(IndexOrder::Fortran).unwrap();
let mut dataset= Dataset::new(data, target).with_feature_names(vec!["test 1", "test 2"]);
let (train, test) = dataset.split_with_ratio(0.8);

// println!("d{:?}",train);


}
