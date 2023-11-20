
use ndarray::{Axis, iter::Axes};
use polars::prelude::*;
use polars_core::prelude::*;
use polars_io::prelude::*;
use std::fs::File;
use chrono::prelude::*;

use polars::prelude::{CsvWriter, DataFrame, NamedFrom, SerWriter, Series};

pub fn main(){
    let mut train_df: DataFrame = CsvReader::from_path("./datasets/titanic_beginner/train.csv")
    .unwrap()
    .finish().unwrap().drop_many(&["PassengerId", "Name", "Ticket"]);
   let test_df = CsvReader::from_path("./datasets/titanic_beginner/test.csv")
   .unwrap()
   .finish().unwrap().drop_many(&["Name", "Ticket"]);
   //데이터 미리보기
   println!("데이터 미리보기:{}",train_df.head(None));
   println!("데이터 정보 확인:{:?}",train_df.schema());
  

  //필요 없는 데이터 삭제
  //train_df.drop_many(&["PassengerId", "Name", "Ticket"]);
  println!("데이터 미리보기:{}",train_df.head(None));
  println!("훈련 데이터 정보 확인:{:?}",train_df.schema());
  println!("{}","----------------------------------------");
  println!("테스트 데이터 정보 확인:{:?}",test_df.schema());
  /*Pclass서수형데이터 1등석,2등석,3등석과 같은정보value에대한 카운팅 */
println!("{}",train_df.column("Pclass").unwrap().value_counts(true,false).unwrap());
let mut output_file: File = File::create("./datasets/titanic_beginner/out.csv").unwrap();

println!("{:?}",train_df.column("Pclass").unwrap().to_dummies(None, false));
println!("{:?}",test_df.column("Pclass").unwrap().to_dummies(None, false));

CsvWriter::new(&mut output_file)
    .has_header(true)
    .finish(&mut train_df)
    .unwrap();
}
