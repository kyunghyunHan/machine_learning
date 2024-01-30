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
 ds_note::ds_note02::main().unwrap();
}
