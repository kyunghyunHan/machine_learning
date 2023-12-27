use plotters::prelude::*;

const OUT_FILE_NAME: &str = "./src/chart/sigmoid_function.png";

pub fn main() {
    // Create a drawing area
    let root_area = BitMapBackend::new(OUT_FILE_NAME, (824, 768)).into_drawing_area();
    root_area.fill(&WHITE).unwrap();
    let root_area: DrawingArea<BitMapBackend<'_>, plotters::coord::Shift> = root_area.titled("SigmoidFunction", ("sans-serif", 60)).unwrap();

    // Create a chart for the step function
    let mut sigmoid_function = ChartBuilder::on(&root_area)
        .margin(5)
        .set_all_label_area_size(50)
        .build_cartesian_2d((-6f32..6.1f32).step(6f32), (0f32..1f32).step(0.5)) 
        .unwrap();

         sigmoid_function
        .configure_mesh().disable_mesh()
        .x_label_formatter(&|v| format!("{:.1}", v))
        .y_label_formatter(&|v| format!("{:.1}", v))
        .draw()
        .unwrap();

    

        sigmoid_function.draw_series(LineSeries::new(
            (-6f32..7f32).step(0.1).values().map(|x| (x, 1.0 / (1.0 + (-x).exp()))), 
            &BLUE,
        ))
    .unwrap();

}
