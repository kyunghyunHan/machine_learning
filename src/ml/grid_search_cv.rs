use smartcore::model_selection::{KFold,cross_validate};
use smartcore::metrics::accuracy;
use smartcore::linear::logistic_regression::LogisticRegression;
use smartcore::api::SupervisedEstimator;
use smartcore::linalg::basic::matrix::DenseMatrix;


fn grid_search_cv(
    x:&DenseMatrix<f64>,
    y:&Vec<i32>,
    param_grid:Vec<(f64,f64)>,
    cv:&KFold
)->(f64, (f64, f64)){
   let mut best_score= 0.0;
   let mut best_params= (0.0,0.0);
    // penalty 설정
    // solver_tolerance 설정
    //cross_validate
    (best_score, best_params)
}

pub fn main(){
   
}

