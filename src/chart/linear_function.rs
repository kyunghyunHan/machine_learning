use plotters::prelude::*;


const OUT_FILE_NAME: &str = "./src/chart/liner_function.png";
pub fn main(){
    let root_area = BitMapBackend::new(OUT_FILE_NAME, (824, 768)).into_drawing_area();
    root_area.fill(&WHITE).unwrap();//배경
    let root_area: DrawingArea<BitMapBackend<'_>, plotters::coord::Shift> = root_area.titled("LinearFunction", ("sans-serif", 60)).unwrap();
    let x_axis = (-3.4f32..3.4).step(0.1);
    let mut linear_function = ChartBuilder::on(&root_area)
    .margin(5)
    .set_all_label_area_size(50)
    .build_cartesian_2d(-8f32..8f32, -8f32..8f32).unwrap();

    linear_function.configure_mesh()
   
    .x_label_formatter(&|v| format!("{:.1}", v))
    .y_label_formatter(&|v| format!("{:.1}", v))
    .draw().unwrap();

linear_function.draw_series(LineSeries::new(
    (-8f32..11f32).step(1.0).values().map(|x| (x, x)),
    &BLUE,
))

.unwrap();
linear_function
.draw_series(LineSeries::new(
    vec![(0f32, -8f32), (0f32, 8f32)],
    &BLACK,
))
.unwrap();

// Draw horizontal line (X-axis)
linear_function
.draw_series(LineSeries::new(
    vec![(-8f32, 0f32), (8f32, 0f32)],
    &BLACK,
))
.unwrap();
     
}