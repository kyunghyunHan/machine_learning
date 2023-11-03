use plotters::prelude::*;
use ndarray::prelude::*;
use smartcore::metrics::*;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::neighbors::knn_classifier::*;


pub fn main() {
    // 도미 데이터
    let bream_length = array![
        25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
        31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
        35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0
    ];

    let bream_weight = array![
        242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
        500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
        700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0
    ];

    // 빙어 데이터
    let smelt_length = array![
        9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0
    ];

    let smelt_weight = array![
        6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9
    ];

    // Create a scatter plot
    let root = BitMapBackend::new("scatter_plot.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let x_range = 0.0..45.0; // Adjust the x-axis range based on your data
    let y_range = 0.0..1050.0; // Adjust the y-axis range based on your data

    let mut chart = ChartBuilder::on(&root)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(x_range, y_range)
        .unwrap();

    chart
        .configure_mesh()
        .x_desc("Length")
        .y_desc("Weight")
        .draw()
        .unwrap();

    chart
        .draw_series(
            bream_length
                .iter()
                .zip(bream_weight.iter())
                .map(|(x, y)| {
                    Circle::new((*x, *y), 5, Into::<ShapeStyle>::into(&RGBColor(255, 0, 0)))
                }),
        )
        .unwrap();

    chart
        .draw_series(
            smelt_length
                .iter()
                .zip(smelt_weight.iter())
                .map(|(x, y)| {
                    Circle::new((*x, *y), 5, Into::<ShapeStyle>::into(&RGBColor(0, 0, 255)))
                }),
        )
        .unwrap();


let fish_data: DenseMatrix<f32> = DenseMatrix::from_2d_array(&[
     &[25.4, 242.0],
     &[26.3, 290.0],
     &[26.5, 340.0],&[29.0, 363.0], &[29.0, 430.0],& [29.7, 450.0],& [29.7, 500.0], &[30.0, 390.0],& [30.0, 450.0], &[30.7, 500.0], &[31.0, 475.0], &[31.0, 500.0], &[31.5, 500.0], &[32.0, 340.0], &[32.0, 600.0], &[32.0, 600.0],& [33.0, 700.0],& [33.0, 700.0], &[33.5, 610.0],& [33.5, 650.0], &[34.0, 575.0], &[34.0, 685.0], &[34.5, 620.0], &[35.0, 680.0], &[35.0, 700.0], &[35.0, 725.0],& [35.0, 720.0],& [36.0, 714.0],&[36.0, 850.0], &[37.0, 1000.0], &[38.5, 920.0],& [38.5, 955.0], &[39.5, 925.0],& [41.0, 975.0], &[41.0, 950.0], &[9.8, 6.7],&[10.5, 7.5], &[10.6, 7.0],& [11.0, 9.7],&[11.2, 9.8],& [11.3, 8.7], &[11.8, 10.0],& [11.8, 9.9],& [12.0, 9.8],& [12.2, 12.2],& [12.4, 13.4], &[13.0, 12.2], &[14.3, 19.7],&[15.0, 19.9]]
    );

// let y:Vec<f32> =vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
let mut fish_target: Vec<i32> = vec![1; 35];
fish_target.extend(vec![0; 14]);
//90프로를 학습데이터로 사용
// let (x_train, x_test, y_train, y_test) = train_test_split(&fish_data, &fish_target, 0.2,true,None);

let  knn: KNNClassifier<f32, i32, DenseMatrix<f32>, Vec<i32>, smartcore::metrics::distance::euclidian::Euclidian<f32>> = KNNClassifier::fit(&fish_data, &fish_target, Default::default()).unwrap();

let y_pred = knn.predict(&fish_data).unwrap();

let acc: f64 = ClassificationMetricsOrd::accuracy().get_score(&fish_target, &y_pred);

println!("Accuracy: {}", acc);
}
