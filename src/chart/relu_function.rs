use plotters::prelude::*;

const OUT_FILE_NAME: &str = "./src/chart/relu_function.png";

pub fn main() {
    // Create a drawing area
    let root_area = BitMapBackend::new(OUT_FILE_NAME, (824, 768)).into_drawing_area();
    root_area.fill(&WHITE).unwrap();
    let root_area: DrawingArea<BitMapBackend<'_>, plotters::coord::Shift> = root_area.titled("ReLU Function", ("sans-serif", 60)).unwrap();

    // Create a chart for the step function
    let mut relu_function = ChartBuilder::on(&root_area)
        .margin(5)
        .set_all_label_area_size(50)
        .build_cartesian_2d(-4f32..5f32, (0f32..5f32).step(1.0)) // Adjust Y-axis range for better visualization
        .unwrap();

    relu_function
        .configure_mesh()
        .disable_mesh()
        .x_label_formatter(&|v| format!("{:.1}", v))
        .y_label_formatter(&|v| format!("{:.1}", v))
        .draw()
        .unwrap();

   
relu_function.draw_series(LineSeries::new(
    (-4f32..6f32).step(1.0).values().map(|x|  (x,f32::max(0f32, x))
   ), 
    &BLUE,
))
.unwrap();
}
