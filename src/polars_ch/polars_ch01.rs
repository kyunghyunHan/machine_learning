use polars::prelude::*;

pub fn main(){
    let df = CsvReader::from_path("./datasets/iris.csv")
    .unwrap()
    .finish()
    .unwrap();
//SepalLengthCm가 5이상인 것들
let mask= df.column("SepalLengthCm").unwrap().f64().unwrap().gt(5.0);
//filter
let df_small = df.filter(&mask).unwrap();
//SepalWidthCm와 PetalLengthCm만 출력
let df_agg = df_small
    .select(["SepalWidthCm","PetalLengthCm"])
    .unwrap();
// println!("{}", df_agg);
//row
println!("{}", df.height());
//col
println!("{}", df.width());
//ndarray 로 변환
let ndarray = df_agg.to_ndarray::<Float64Type>(IndexOrder::Fortran).unwrap();
// println!("{}", ndarray);

}
