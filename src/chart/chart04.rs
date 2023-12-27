use plotters::prelude::*;


const OUT_FILE_NAME: &str = "./src/chart/chart04.png";
pub fn main(){
    let root_area = BitMapBackend::new(OUT_FILE_NAME, (1024, 768)).into_drawing_area();
    root_area.fill(&WHITE).unwrap();//배경
    let root_area: DrawingArea<BitMapBackend<'_>, plotters::coord::Shift> = root_area.titled("Image Title", ("sans-serif", 60)).unwrap();
    let x_axis = (-3.4f32..3.4).step(0.1);
    let mut cc = ChartBuilder::on(&root_area)
    .margin(5)
    .set_all_label_area_size(50)
    .build_cartesian_2d(-3.4f32..3.4, -1.2f32..1.2f32).unwrap();

    cc.configure_mesh()
    .disable_mesh()
    .x_label_formatter(&|v| format!("{:.1}", v))
    .y_label_formatter(&|v| format!("{:.1}", v))
    .draw().unwrap();
}