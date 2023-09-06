use std::io::prelude::*;
use std::io::BufReader;
use std::path::Path;
use std::fs::File;
use std::vec::Vec;
use rusty_machine::{linalg::{Matrix,BaseMatrix,Vector},learning::{SupModel,lin_reg::LinRegressor}};
use rusty_machine::analysis::score::neg_mean_squared_error;



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
    pub fn new(v:Vec<&str> ) -> HousingDataset {
        let unwrapped_text:Vec<f64> = v.iter().map(|r|r.parse().unwrap()).collect();
        HousingDataset {crim: unwrapped_text[0],
            zn: unwrapped_text[1], 
            indus: unwrapped_text[2],
            chas: unwrapped_text[3],
            nox: unwrapped_text[4],
            rm: unwrapped_text[5],
            age: unwrapped_text[6],
            dis: unwrapped_text[7],
            rad: unwrapped_text[8],
            tax: unwrapped_text[9],
            ptratio: unwrapped_text[10],
            black: unwrapped_text[11],
            lstat: unwrapped_text[12],
            medv: unwrapped_text[13]} 
    }

    pub fn train_features(&self)->Vec<f64>{
        vec![self.crim, 
        self.zn,
        self.indus, 
        self.chas, 
        self.nox, 
        self.rm, 
        self.age, 
        self.dis, 
        self.rad, 
        self.tax, 
        self.ptratio, 
        self.black, 
        self.lstat] // remember we're returning this so no ; 
    }

    pub fn train_target(&self)->f64 {
        self.medv
    }
    
}

fn read_single_line(s: String) -> HousingDataset { 
    // read a single line 
    let raw_vector: Vec<&str> = s.split_whitespace().collect(); 
    // now read the single vector
    let housing_vector: HousingDataset = HousingDataset::new(raw_vector); 
    // now return the vector, no ; 
    housing_vector
}

pub fn read_housing_csv(file_name:impl AsRef<Path>)->Vec<HousingDataset>{
    let file = File::open(file_name).expect("Please, give an input file. Cannot find file");
    let reader: BufReader<File> = BufReader::new(file);

    
    reader.lines().enumerate()
    .map(|(numb, line)| line.expect(&format!("Impossible to read line number {}", numb)))
    .map(|row| read_single_line(row))
    .collect()
}


pub fn main(){
    let ifile = "datasets/boston_house.csv";
    let mut input_data = read_housing_csv(&ifile);

    let test_chunk_size: f64 = input_data.len() as f64 * 0.3; // we are taking the 30% as test data
    let test_chunk_size = test_chunk_size.round() as usize;
    let (test, train) = input_data.split_at(test_chunk_size); 
    let train_size = train.len() ; 
    let test_size = test.len();

    let x_train: Vec<f64> = train.iter().flat_map(|row| row.train_features()).collect();
    let y_train: Vec<f64> = train.iter().map(|row| row.train_target()).collect();

    let x_test: Vec<f64> = test.iter().flat_map(|row| row.train_features()).collect();
    let y_test: Vec<f64> = test.iter().map(|row| row.train_target()).collect();

    let x_train_matrix = Matrix::new(train_size, 13, x_train); // 13 is the number of features
    let y_train_vector = Vector::new(y_train);
    let x_test_matrix = Matrix::new(test_size, 13, x_test);

     // MODEL! 
     let mut linearRegression = LinRegressor::default(); 
     // train 
     linearRegression.train(&x_train_matrix, &y_train_vector);
     // predictions
     let preds = linearRegression.predict(&x_test_matrix).unwrap();
     // convert to matrix both preds and y_test 
     let preds_matrix = Matrix::new(test_size, 1, preds); 
     let y_preds_matrix = Matrix::new(test_size, 1, y_test);
     // compute the mse
     let mse = neg_mean_squared_error(&preds_matrix, &y_preds_matrix); 
 
     println!("Final negMSE (the higher the better) {:?}", mse);
     // return the mse 
}