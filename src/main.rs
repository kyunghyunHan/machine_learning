use chart::matshow;
use polars::prelude::cov::pearson_corr;

mod chart;
mod ds_note;
mod fish;
mod ml;
mod polars_ch;
mod sql;
use polars::prelude::*;
fn main() {
    /*=========chart======== */
    // chart::chart02::main();
    // chart::chart02::main();
    // chart::chart03::main();
    // chart::chart04::main();
    // chart::linear_function::main();
    // chart::step_function::main();
    // chart::sigmoid_function::main();
    // chart::tanh_function::main();
    // chart::relu_function::main();
    // chart::sign_function::main();
    // chart::time_series::main();
    /*=========chart======== */
    /*=========fish======== */
    // fish::fish01::main();
    // fish::wine01::main();
    /*=========fish======== */

    /*=========wine======== */

    /*=========wine======== */
    /*=========polars======== */
    // polars_ch::polars_ch01::main();
    /*=========polars======== */
    /*=========DB======== */

    // sql::sql_connect::main();
    //  ds_note::ds_note02::main().unwrap();
    // chart::matshow::main();
    let mut s0 = Series::new("col1", [1, 2, 3, 4, 5, 6].as_ref());
    let s1 = Series::new("col2", [1, 4, 2, 8, 16, 32].as_ref());
    let s2 = Series::new("col3", [6, 5, 4, 3, 2, 1].as_ref());
    let df = DataFrame::new(vec![s0.clone(), s1.clone(), s2]).unwrap();
    // let df: DataFrame = df
    //     .clone()
    //     .lazy()
    //     .with_columns([
    //         cov(col("col1"), col("col3"), 1).alias("col1").g,
    //         cov(col("col2"), col("col3"), 1).alias("col2"),
    //         cov(col("col1"), col("col3"), 1).alias("col3"),
    //     ])
    //     .collect()
    //     .unwrap();


    let a: f64= pearson_corr(s0.clone().i32().unwrap(), s1.clone().i32().unwrap(), 1).unwrap();

    println!("{}", a);
}
