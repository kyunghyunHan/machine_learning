use polars::prelude::*;

pub  fn main(){

    let df = CsvReader::from_path("./datasets/red_wine.csv")
    .unwrap()
    .finish()
    .unwrap();

    println!("{}",df);
    let df_agg = df.select(["alcohol","residual sugar","pH","citric acid"])
    .unwrap();
    println!("{}",df_agg);
 
}
