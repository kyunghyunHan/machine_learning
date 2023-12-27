use plotters::prelude::*;

const OUT_FILE_NAME: &str = "./src/chart/tanh_function.png";

pub fn main() {
    // Create a drawing area
    let root_area = BitMapBackend::new(OUT_FILE_NAME, (824, 768)).into_drawing_area();
    root_area.fill(&WHITE).unwrap();
    let root_area: DrawingArea<BitMapBackend<'_>, plotters::coord::Shift> = root_area.titled("TanhFunction", ("sans-serif", 60)).unwrap();

    // Create a chart for the step function
    let mut tanh_function = ChartBuilder::on(&root_area)
        .margin(5)
        .set_all_label_area_size(50)
        .build_cartesian_2d((-6f32..6.1f32).step(6f32), (-1f32..1f32).step(0.5)) 
        .unwrap();

         tanh_function
        .configure_mesh().disable_mesh()
        .x_label_formatter(&|v| format!("{:.1}", v))
        .y_label_formatter(&|v| format!("{:.1}", v))
        .draw()
        .unwrap();

    

        tanh_function.draw_series(LineSeries::new(
            (-6f32..7f32).step(0.1).values().map(|x|(x,(x.exp() - (-x).exp()) / (x.exp() + (-x).exp())) ), 
            &BLUE,
        ))
    .unwrap();

}
