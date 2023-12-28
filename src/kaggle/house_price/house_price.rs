use std::fs::File;
use ndarray::prelude::*;
use polars::prelude::*;
use polars_core::prelude::*;
use polars_io::prelude::*;
use plotters::prelude::*;
const OUT_FILE_NAME: &str = "./src/kaggle/house_price/house_price.png";

pub fn main(){
    println!("{}","hello");

    let train_df: DataFrame = CsvReader::from_path("./datasets/house-prices-advanced-regression-techniques/train.csv")
    .unwrap()
    .has_header(true)
    .with_null_values(Some(NullValues::AllColumns(vec!["NA".to_owned()])))
    .finish()
    .unwrap();   
    println!("{}",train_df);

    let a=  train_df.column("SalePrice").unwrap();
    let y_train: Vec<i64> = a.i64().unwrap().into_no_null_iter().collect();
    let y_train:Vec<f32>= y_train.iter().map(|x|*x as f32).collect();
    let b: Vec<&Series>= train_df.columns(["YrSold","SalePrice"]).unwrap();
    println!("{:?}",b);
    let mut c= train_df.select(["YrSold","SalePrice"]).unwrap();

    let  x_train: ArrayBase<ndarray::OwnedRepr<f32>, Dim<[usize; 2]>> = c.to_ndarray::<Float32Type>(IndexOrder::Fortran).unwrap();

    println!("{:?}",x_train);
    let mut data: Vec<Vec<_>> = Vec::new();
    for row in x_train.outer_iter() {
        let row_vec: Vec<_> = row.iter().cloned().collect();
        data.push(row_vec);
    }
    println!("{:?}",data);
    
    let root_area = BitMapBackend::new(OUT_FILE_NAME, (824, 768)).into_drawing_area();
    root_area.fill(&WHITE).unwrap();//배경
    let root_area: DrawingArea<BitMapBackend<'_>, plotters::coord::Shift> = root_area.titled("House Price", ("sans-serif", 60)).unwrap();
    let mut linear_function = ChartBuilder::on(&root_area)
    .margin(5)
    .set_all_label_area_size(50)
    .build_cartesian_2d(147500f32..208500f32, 147500f32..208500f32).unwrap();


   

linear_function.draw_series(PointSeries::of_element(
    data.into_iter().map(|x|x).step_by(1).map(|x| (x[0], x[1])),
    2,
    ShapeStyle::from(&RED).filled(),
    &|coord, size, style| {
        EmptyElement::at(coord)
            + Circle::new((0, 0), size, style)
            // + Text::new(format!("{:?}", coord), (0, 15), ("sans-serif", 15))
    },
)).unwrap();
}