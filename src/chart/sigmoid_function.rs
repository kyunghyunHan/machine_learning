use plotters::prelude::*;
const OUT_FILE_NAME:&str= "./src/chart/sigmoid_function.png";
pub fn main(){
    let root_area = BitMapBackend::new(OUT_FILE_NAME, (824, 768)).into_drawing_area();
    root_area.fill(&WHITE).unwrap();
    let root_area: DrawingArea<BitMapBackend<'_>, plotters::coord::Shift> = root_area.titled("StepFunction", ("sans-serif", 60)).unwrap();

}