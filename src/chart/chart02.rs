use plotters::prelude::*;

const OUT_FILE_NAME: &str = "./src/chart/chart02.png";
pub fn main() {
    let root_area = BitMapBackend::new(OUT_FILE_NAME, (600, 400))
    .into_drawing_area();
    root_area.fill(&WHITE).unwrap();

    let mut ctx = ChartBuilder::on(&root_area)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .caption("Bar Demo", ("sans-serif", 40))
        .build_cartesian_2d((0..10).into_segmented(), 0..50)
        .unwrap();
    
    ctx.configure_mesh().draw().unwrap();

    let data = [25,30];
    let data1 = [30,40];

    ctx.draw_series((0..2).zip(data.iter()).map(|(x, y)| {
        let x0 = SegmentValue::Exact(x);
        let x1 = SegmentValue::Exact(x + 1);
        let mut bar = Rectangle::new([(x0, 0), (x1, *y)], RED.filled());
        bar.set_margin(0, 0, 0,0 );
        bar
    }))
    .unwrap();
ctx.draw_series((0..2).zip(data.iter()).map(|(x, y)| {
    let x0 = SegmentValue::Exact(x);
    let x1 = SegmentValue::Exact(x + 1);
    let mut bar = Rectangle::new([(x0, 0), (x1, *y)], RED.filled());
    bar.set_margin(0, 0, 0,0 );
    bar
}))
.unwrap();

    ctx.draw_series((2..4).zip(data1.iter()).map(|(x, y)| {
    let x0 = SegmentValue::Exact(x);
    let x1 = SegmentValue::Exact(x + 1);
    let mut bar = Rectangle::new([(x0, 0), (x1, *y)], BLUE.filled());
    bar.set_margin(0, 0, 0, 0);
    bar
    }))
    .unwrap();

}