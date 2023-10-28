use plotters::prelude::*;

pub fn main() {
    let bream_length = vec![
        25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
        31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
        35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0,
    ];
    let bream_weight = vec![
        242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
        500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
        700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0,
    ];

    // Create a scatter plot
    let root = BitMapBackend::new("scatter_plot.png", (800, 600))
        .into_drawing_area();
    root.fill(&WHITE).unwrap();

    let x_range = 25.0..45.0;
    let y_range = 0.0..1050.0;

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
}
