use plotters::prelude::*;

const OUT_FILE_NAME: &str = "./src/chart/matshow.png";

pub fn main() {
    let root = BitMapBackend::new(OUT_FILE_NAME, (1024, 768)).into_drawing_area();

    root.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .caption("Heatmap Example", ("sans-serif", 80))
        .margin(5)
        .top_x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0i32..15i32, 15i32..0i32).unwrap();

    chart
        .configure_mesh()
        .x_labels(15)
        .y_labels(15)
        .max_light_lines(4)
        .x_label_offset(35)
        .y_label_offset(25)
        .disable_x_mesh()
        .disable_y_mesh()
        .label_style(("sans-serif", 20))
        .draw().unwrap();

    let mut matrix = [[0; 15]; 15];

    for i in 0..15 {
        matrix[i][i] = i + 4;
    }

    chart.draw_series(
        matrix
            .iter()
            .zip(0..)
            .flat_map(|(row, y)| row.iter().zip(0..).map(move |(value, x)| (x, y, value)))
            .map(|(x, y, value)| {
                Rectangle::new(
                    [(x, y), (x + 1, y + 1)],
                    HSLColor(
                        240.0 / 360.0 - 240.0 / 360.0 * (*value as f64 / 20.0),
                        0.7,
                        0.1 + 0.4 * *value as f64 / 20.0,
                    )
                    .filled(),
                )
            }),
    ).unwrap();

    // To avoid the IO failure being ignored silently, we manually call the present function
    root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
    println!("Result has been saved to {}", OUT_FILE_NAME);
}
