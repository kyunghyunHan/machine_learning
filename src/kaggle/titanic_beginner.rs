use polars::prelude::*;
use polars_core::prelude::*;
use polars_io::prelude::*;
use std::fs::File;
use chrono::prelude::*;

use polars::prelude::{CsvWriter, DataFrame, NamedFrom, SerWriter, Series};

pub fn main(){
    let train_df = CsvReader::from_path("./datasets/titanic_beginner/train.csv")
    .unwrap()
    .finish()
    .unwrap().drop_many(&["PassengerId", "Name", "Ticket"]);
   let test_df = CsvReader::from_path("./datasets/titanic_beginner/test.csv")
   .unwrap()
   .finish()
   .unwrap().drop_many(&["Name", "Ticket"]);

   println!("{}",train_df.head(None));
  
   let value_counts = train_df.column("Pclass").unwrap().value_counts(false,false).unwrap();
   println!("{}",value_counts);

   let mut df: DataFrame = df!(
    "integer" => &[1, 2, 3],
    "date" => &[
           1,1,1
    ],
    "float" => &[4.0, 5.0, 6.0]

    
)
.unwrap();
let mut df: DataFrame = df!(
    "integer" => &[1, 2, 3],
    "date" => &[
            NaiveDate::from_ymd_opt(2022, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap(),
            NaiveDate::from_ymd_opt(2022, 1, 2).unwrap().and_hms_opt(0, 0, 0).unwrap(),
            NaiveDate::from_ymd_opt(2022, 1, 3).unwrap().and_hms_opt(0, 0, 0).unwrap(),
    ],
    "float" => &[4.0, 5.0, 6.0]
)
.unwrap();
println!("{}", df);


let mut output_file: File = File::create("out.csv").unwrap();
//파일생성
CsvWriter::new(&mut output_file)
    .has_header(true)
    .finish(&mut df)
    .unwrap();
}

