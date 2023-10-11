use datafusion::prelude::*;
use std::sync::Arc;
use datafusion::arrow::datatypes::DataType;
use datafusion::logical_plan::case;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::metrics::roc_auc_score;
use smartcore::model_selection::train_test_split;
use smartcore::tree::decision_tree_classifier::DecisionTreeClassifier;
// use std::fs::File;
use ndarray::{concatenate, Array2, ArrayBase, Axis, Dim, OwnedRepr, ViewRepr};
use std::{
    fs::File,
    io::{BufRead, BufReader},
};
async fn load_data() -> datafusion::error::Result<Arc<DataFrame>> {
    let ctx = SessionContext::new();
    ctx.register_csv("weather", "./datasets/weather.csv", 
      CsvReadOptions::new()).await?;
    ctx.sql("SELECT * FROM weather").await
}



fn clean_data(mut df: Arc<DataFrame>) 
-> datafusion::error::Result<Arc<DataFrame>> {
  let filter_expr = df
      .schema()
      .field_names()
      .iter()
      .enumerate()
      .filter(|(idx, _)| *idx > 0)
      .fold(
          col(&df.schema().field_names()[0]).not_eq(lit("NA")),
          |expr, (_, field)| expr.and(col(field).not_eq(lit("NA"))),
      );
  df = df.filter(filter_expr)?;
  df.with_column(
      "Outcome",
      case(col("RainTomorrow"))
          .when(lit("Yes"), lit(1.0f64))
          .otherwise(lit(0.0f64))?,
  )
}


fn select_data(df: Arc<DataFrame>) 
  -> datafusion::error::Result<Arc<DataFrame>> {
    let x_cols = vec![
        "MinTemp",
        "MaxTemp",
        "Evaporation",
        "Sunshine",
        "WindGustSpeed",
        "WindSpeed9am",
        "WindSpeed3pm",
        "Humidity9am",
        "Humidity3pm",
    ];
    let col_selections = {
        let mut cols = vec!["Outcome"];
        cols.append(&mut x_cols.clone());
        cols
    }
    .iter()
    .map(|field| cast(col(field), DataType::Float64).alias(field))
    .collect();
    df.select(col_selections)
}async fn extract_data(df: Arc<DataFrame>) -> (Vec<Vec<f64>>, Vec<f64>) {
    df.write_csv("./datasets/weather_processed")
        .await
        .unwrap();
    let f = File::open("./datasets/weather_processed/part-0.csv").unwrap();
    let mut reader = csv::Reader::from_reader(f);
    let mut y_data = vec![];
    let mut x_data: Vec<Vec<f64>> = vec![];
    for result in reader.records() {
        let record = result.unwrap();
        y_data.push(record.get(0).unwrap().parse::<f64>().unwrap());
        x_data.push(
            record
                .iter()
                .enumerate()
                .filter(|(idx, _)| *idx > 0)
                .map(|(_, v)| v.parse::<f64>().unwrap())
                .collect(),
        );
    }
    (x_data, y_data)
}

fn create_model(x_data: &Vec<Vec<f64>>, y_data: &Vec<f64>) {
    let x = DenseMatrix::from_2d_vec(&x_data);
    let y = y_data;

    let (x_train, x_test, y_train, y_test) = 
      train_test_split(&x, &y, 0.2, true);

    let predictions = 
      DecisionTreeClassifier::fit(&x_train, &y_train, Default::default())
        .and_then(|tree| tree.predict(&x_test))
        .unwrap();

    println!("{}", roc_auc_score(&y_test, &predictions));
}
#[tokio::main]
pub async fn main(){
    let mut df = load_data().await.unwrap();
    df = clean_data(df).unwrap();
 df = select_data(df).unwrap();
let (x_data, y_data) = extract_data(df).await;
create_model(&x_data, &y_data);
}