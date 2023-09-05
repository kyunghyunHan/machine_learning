use std::io::prelude::*;
use std::io::BufReader;
use std::path::Path;
use std::fs::File;
use std::vec::Vec;


pub struct HousingDataset{
    crim: f64,
    zn: f64,
    indus: f64,
    chas: f64,
    nox: f64,
    rm: f64,
    age: f64,
    dis: f64,
    rad: f64,
    tax: f64,
    ptratio: f64, 
    black: f64,
    lstat: f64, 
    medv: f64,
}
impl HousingDataset {
    pub fn new(v:Vec<&str> )-> HousingDataset {
        let unwrapped_text:Vec<f64> = v.iter().map(|r|r.parse().unwrap()).collect();
    
        }
}
pub fn read_hosing_csv(){
    
}

pub fn main(){

}