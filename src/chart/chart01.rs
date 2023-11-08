use plotters::prelude::*;
use ndarray::prelude::*;


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
    let root = BitMapBackend::new("./src/chart/chart01.png", (800, 600)).into_drawing_area();
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

}
