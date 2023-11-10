use polars::prelude::*;

pub  fn main(){

    let df = CsvReader::from_path("./datasets/winequalityN.csv")
    .unwrap()
    .finish()
    .unwrap();

    println!("{}",df);
}
