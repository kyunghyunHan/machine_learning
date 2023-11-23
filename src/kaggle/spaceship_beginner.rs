/*

제츨파일

PassengerId :PassengerId

Transported : target
*/
use std::fs::File;
use ndarray::prelude::*;
use polars::prelude::*;
use polars_core::prelude::*;
use polars_io::prelude::*;

use std::str;
pub fn main(){
    /*데이터불러오기 */
    let mut train_df: DataFrame = CsvReader::from_path("./datasets/spaceship_titanic/train.csv")
    .unwrap()
    .finish().unwrap();

    let  test_df: DataFrame = CsvReader::from_path("./datasets/spaceship_titanic/test.csv")
    .unwrap()
    .finish().unwrap();



}