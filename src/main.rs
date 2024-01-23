mod polars_ch;
mod chart;
mod fish;
mod ml;
mod sql;
mod ds_note;

// fn main(){

// /*=========chart======== */
// // chart::chart02::main();
// // chart::chart02::main();
// // chart::chart03::main();
// // chart::chart04::main();
// // chart::linear_function::main();
// // chart::step_function::main();
// // chart::sigmoid_function::main();
// // chart::tanh_function::main();
// // chart::relu_function::main();
// // chart::sign_function::main();
// // chart::time_series::main();
// /*=========chart======== */
// /*=========fish======== */
// // fish::fish01::main();
// // fish::wine01::main();
// /*=========fish======== */

// /*=========wine======== */

// /*=========wine======== */
// /*=========polars======== */
// // polars_ch::polars_ch01::main();
// /*=========polars======== */
// /*=========DB======== */

// // sql::sql_connect::main();
// ds_note::ds_note02::main();

// }


use polars::prelude::*;
use reqwest;
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
   

    Ok(())


}