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
    .build_cartesian_2d(-10.0f32..10.0, -10.0f32..10.0).unwrap();

    cc.configure_mesh()
   
    .x_label_formatter(&|v| format!("{:.1}", v))
    .y_label_formatter(&|v| format!("{:.1}", v))
    .draw().unwrap();
cc.draw_series(PointSeries::of_element(
    (-3.0f32..2.1f32).step(1.0).values().map(|x| (x, x.signum())),
    5,
    ShapeStyle::from(&RED).filled(),
    &|coord, size, style| {
        EmptyElement::at(coord)
            + Circle::new((0, 0), size, style)
            + Text::new(format!("{:?}", coord), (0, 15), ("sans-serif", 15))
    },
)).unwrap();

     
}