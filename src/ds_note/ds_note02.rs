use polars::prelude::*;
use reqwest;
use reqwest::Client;
#[tokio::main]

pub async fn main()-> Result<(), reqwest::Error> {
    let url = "htts://media.githubusercontent.com/media/must-ML10/data_source/main/insurance.csv";
    let client = Client::new();
//   let tf= CsvReader::from_path(path)
    // Make a GET request to the specified URL
    let resp = client.get(url).send().await?;
     
    // Check if the request was successful (status code 2xx)
    if resp.status().is_success() {
        // Read the response body
        let body = resp.text().await?;
        let reader = CsvReader::new(std::io::Cursor::new(body)).has_header(true);

        // Read the CSV data into a DataFrame
        let df: DataFrame = reader.finish().unwrap();

        println!("Response body: {}", df);
        
    } else {
        // Print an error message if the request was not successful
        eprintln!("Request failed with status code: {}", resp.status());
    }

    Ok(())


}