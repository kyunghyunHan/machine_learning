use polars::prelude::*;
use reqwest;
use smartcore::model_selection::train_test_split;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linear::linear_regression::{LinearRegression, LinearRegressionParameters};
use smartcore::metrics::*;/*



*/
#[tokio::main]

pub async fn main()-> Result<(), reqwest::Error> {
    let url = "https://media.githubusercontent.com/media/musthave-ML10/data_source/main/insurance.csv";
    let client = reqwest::Client::new();

    let resp: reqwest::Response = client.get(url).send().await?;

  
    let body = resp.bytes().await?;

    let reader = CsvReader::new(std::io::Cursor::new(body)).has_header(true);
    /*데이터 불러오기 */
    let df = reader.finish().unwrap();

    println!("{:?}", df);
    println!("{:?}", df.head(None));
    println!("{:?}", df.schema());
    println!("{:?}", df.null_count());

    let y= df.column("charges").unwrap().f64().unwrap().into_no_null_iter().map(|x|x as f64).collect::<Vec<f64>>();
    let x= df.drop("charges").unwrap().to_ndarray::<Float64Type>(IndexOrder::Fortran).unwrap();
    let mut x_vec: Vec<Vec<_>> = Vec::new();
    for row in x.outer_iter(){
        let row_vec:Vec<_>=row.iter().cloned().collect();
        x_vec.push(row_vec);
    }
    let x: DenseMatrix<f64> = DenseMatrix::from_2d_vec(&x_vec);

    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true, None);


    let model= LinearRegression::fit(&x_train, &y_train, LinearRegressionParameters::default()).unwrap();
    let y_pred: Vec<f64> = model.predict(&x_test).unwrap();
    // let acc: f64 = ClassificationMetricsOrd::accuracy().get_score(&y_test, &y_pred);
    
    // println!("{}",acc);
    /*mean_squared_error */
    let mean_squared_error= mean_squared_error(&y_test, &y_pred);
    println!("{}",f64::powf(mean_squared_error, 0.5) );

    Ok(())


}