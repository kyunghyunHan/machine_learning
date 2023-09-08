use std::result::Result;
use std::error::Error;
use mnist::*;
use tch::{kind,no_grad,Kind,Tensor};
use ndarray::{Array3,Array2};
const LABELS: i64 = 10; // number of distinct labels
    const HEIGHT: usize = 28; 
    const WIDTH: usize = 28;
    
    const TRAIN_SIZE: usize = 50000;
    const VAL_SIZE: usize = 10000;
    const TEST_SIZE: usize =10000;
    
    const N_EPOCHS: i64 = 200;
    
    const THRES: f64 = 0.001;





    pub fn image_to_tensor(){

    }

pub fn labels_to_tensor(){}


fn print_type_of(){}
pub fn main(){
    


}