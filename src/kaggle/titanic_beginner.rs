
use ndarray::{Axis, iter::Axes};
use polars::prelude::*;
use polars_core::prelude::*;
use polars_io::prelude::*;
use std::fs::File;
use chrono::prelude::*;

use polars::prelude::{CsvWriter, DataFrame, NamedFrom, SerWriter, Series};

pub fn main(){
    let  train_df: DataFrame = CsvReader::from_path("./datasets/titanic_beginner/train.csv")
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
  

/*Pclass */
let mut pclass_train_dummies= train_df.column("Pclass").unwrap().to_dummies(None, false).unwrap();
let mut pclass_test_dummies= test_df.column("Pclass").unwrap().to_dummies(None, false).unwrap();


let  pclass_train_dummies= pclass_train_dummies.rename("Pclass_1","Pclass_1").unwrap();
let  pclass_test_dummies= pclass_test_dummies.rename("Pclass_1","Pclass_1").unwrap();

println!("train_dummies:{:?}",pclass_train_dummies);
println!("test_dummies{:?}",pclass_test_dummies);

let  mut train_df: DataFrame = train_df.drop("Pclass").unwrap();
let mut test_df= test_df.drop("Pclass").unwrap();
println!("train:{:?}",train_df);
println!("test:{:?}",test_df);

for series in pclass_train_dummies.iter(){
    train_df = train_df.hstack(&[series.clone()]).unwrap();
}

for series in pclass_test_dummies.iter(){
    test_df = test_df.hstack(&[series.clone()]).unwrap();
}
println!("train:{:?}",train_df);
println!("test:{:?}",test_df);

/*Sex */

let  mut sex_train_dummies= train_df.column("Sex").unwrap().to_dummies(None, false).unwrap();
let  mut sex_test_dummies= test_df.column("Sex").unwrap().to_dummies(None, false).unwrap();


let  sex_train_dummies= sex_train_dummies.rename("Sex_female","Female").unwrap().rename("Sex_male","male").unwrap();
let  sex_test_dummies= sex_test_dummies.rename("Sex_female","Female").unwrap().rename("Sex_male","male").unwrap();
println!("train_dummies:{:?}",sex_train_dummies);
println!("test_dummies{:?}",sex_test_dummies);

let  mut train_df = train_df.drop("Sex").unwrap();
let  mut test_df = test_df.drop("Sex").unwrap();
println!("train:{:?}",train_df);
println!("test:{:?}",test_df);
//join
for series in sex_train_dummies.iter(){
    train_df = train_df.hstack(&[series.clone()]).unwrap();
}
for series in sex_test_dummies.iter(){
    test_df = test_df.hstack(&[series.clone()]).unwrap();
}
println!("train:{:?}",train_df);
println!("test:{:?}",test_df);
/*Age */

println!("train_df:{:?}",train_df);

let  train_df: &mut DataFrame= train_df.with_column(  train_df.column("Age").unwrap().fill_null(FillNullStrategy::Mean).unwrap()).unwrap();
let  test_df: &mut DataFrame= test_df.with_column(  test_df.column("Age").unwrap().fill_null(FillNullStrategy::Mean).unwrap()).unwrap();

println!("train_df:{:?}",train_df);
println!("test_df:{:?}",test_df);


/*SibSp & Panch 
바꿀필요 x
*/


/*Fare 
무단탑승으로 생각하고 0으로 입력
*/
println!("Fare:{}",test_df.column("Fare").unwrap().value_counts(true,false).unwrap());

let mut test_df: &mut DataFrame= test_df.with_column(  test_df.column("Fare").unwrap().fill_null(FillNullStrategy::Zero).unwrap()).unwrap();
println!("Fare:{}",test_df.column("Fare").unwrap().value_counts(true,false).unwrap());

println!("train_df:{:?}",train_df);
println!("test_df:{:?}",test_df);
/*Cabin */
let  mut train_df = train_df.drop("Cabin").unwrap();
let mut test_df= test_df.drop("Cabin").unwrap();
println!("train_df:{:?}",train_df);
println!("test_df:{:?}",test_df);
/*Embarked
탑승구를 의미
*/

println!("{}",train_df.column("Embarked").unwrap().value_counts(true,false).unwrap());
let mut  train_df= train_df.with_column(  train_df.column("Embarked").unwrap().fill_null(FillNullStrategy::Min).unwrap()).unwrap();
println!("{}",train_df.column("Embarked").unwrap().value_counts(true,false).unwrap());

/*데이터 나누기 */

/*알고리즘 적용 */


/*제출용 파일 */
let mut output_file: File = File::create("./datasets/titanic_beginner/out.csv").unwrap();
CsvWriter::new(&mut output_file)
    .has_header(true)
    .finish(&mut train_df)
    .unwrap();
}
