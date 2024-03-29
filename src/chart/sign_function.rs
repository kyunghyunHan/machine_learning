use plotters::prelude::*;

const OUT_FILE_NAME: &str = "./src/chart/sign_function.png";

pub fn main() {
    // Create a drawing area
    let root_area = BitMapBackend::new(OUT_FILE_NAME, (824, 768)).into_drawing_area();
    root_area.fill(&WHITE).unwrap();
    let root_area: DrawingArea<BitMapBackend<'_>, plotters::coord::Shift> = root_area.titled("Sign Function", ("sans-serif", 60)).unwrap();

    // Create a chart for the step function
    let mut sign_function = ChartBuilder::on(&root_area)
        .margin(5)
        .set_all_label_area_size(50)
        .build_cartesian_2d(-4f32..4f32, (-1f32..1f32).step(0.5)) // Adjust Y-axis range for better visualization
        .unwrap();

    sign_function
        .configure_mesh()
        // .disable_mesh()
        .x_label_formatter(&|v| format!("{:.1}", v))
        .y_label_formatter(&|v| format!("{:.1}", v))
        .draw()
        .unwrap();


    sign_function
    .draw_series(LineSeries::new(
        vec![(0f32, -1f32), (0f32, 1f32)],
        &BLUE,
    ))
    .unwrap();
   sign_function
    .draw_series(LineSeries::new(
        vec![(-4f32, -1f32), (0f32, -1f32)],
        &BLUE,
    ))
    .unwrap();
sign_function.draw_series(LineSeries::new(
    (0f32..4.2f32).step(0.2).values().map(|x| (x, 1f32)),
    &BLUE,
)).unwrap();
}
