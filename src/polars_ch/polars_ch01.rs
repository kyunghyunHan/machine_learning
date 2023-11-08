use polars::prelude::*;
use ndarray::prelude::*;
use polars::prelude::DataFrame;
pub fn main(){
    let df = CsvReader::from_path("./datasets/iris.csv")
    .unwrap()
    .finish()
    .unwrap();
// let mask= df.column("SepalLengthCm").unwrap().f64().unwrap().gt(5.0);
// let df_small = df.filter(&mask).unwrap();
// let df_agg = df_small
//     .group_by(["Species"]).unwrap()
//     .select(["SepalWidthCm"])
//     .mean().unwrap();
println!("{}", df);
let ndarray = df.to_ndarray::<Float64Type>(IndexOrder::Fortran).unwrap();
println!("{}", ndarray);

}
