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
use reqwest::Client;
#[tokio::main]

pub async fn main()-> Result<(), reqwest::Error> {
    let url = "https://media.githubusercontent.com/media/musthave-ML10/data_source/main/insurance.csv";
    let client = Client::new();

    // Make a GET request to the specified URL
    let resp = client.get(url).send().await?;

    // Check if the request was successful (status code 2xx)
    if resp.status().is_success() {
        // Read the response body
        let body = resp.text().await?;
        println!("Response body: {}", body);
    } else {
        // Print an error message if the request was not successful
        eprintln!("Request failed with status code: {}", resp.status());
    }

    Ok(())


}