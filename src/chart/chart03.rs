use plotters::prelude::*;
const OUT_FILE_NAME: &str = "./src/chart/chart03.png";
pub fn main(){
    //1024 x 768 pexel setting
    let root_area = BitMapBackend::new(OUT_FILE_NAME, (1024, 768)).into_drawing_area();
    root_area.fill(&WHITE).unwrap();//배경
    //title
    
    let root_area: DrawingArea<BitMapBackend<'_>, plotters::coord::Shift> = root_area.titled("Image Title", ("sans-serif", 60)).unwrap();
    //수직방향으로 나누기 처음영역이 512
    let (upper, lower) = root_area.split_vertically(200);
    //범위설정
    let x_axis = (-3.4f32..3.4).step(0.1);
    //ChartBuilder를 사용하여 새로운 차트를 생성하는 부분
    let mut cc = ChartBuilder::on(&upper)
    .margin(5)
    .set_all_label_area_size(50)
    .caption("Sine and Cosine", ("sans-serif", 40))
    .build_cartesian_2d(-3.4f32..3.4, -1.2f32..1.2f32).unwrap();
//set_all_label_area_size:그래프 크기 설정
//caption:제목설정
//build_cartesian_2d:카테시안 좌표계 설정- 범위


//configure_mesh:눈금 및 그리드 설정
//x_labels:x축에 사용되는 눈금의 개수 설정
//.disable_mesh() // 메시 또는 그리드 라인 비활성화

cc.configure_mesh()
    .disable_mesh()
    .x_label_formatter(&|v| format!("{:.1}", v))
    .y_label_formatter(&|v| format!("{:.1}", v))
    .draw().unwrap();
//Sine
cc.draw_series(LineSeries::new(x_axis.values().map(|x| (x, x.sin())), &RED)).unwrap()
    .label("Sine")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));
//Cosin
cc.draw_series(LineSeries::new(
    x_axis.values().map(|x| (x, x.cos())),
    &BLUE,
)).unwrap()
.label("Cosine")
.legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

cc.configure_series_labels().border_style(BLACK).draw().unwrap();

cc.draw_series(PointSeries::of_element(
    (-3.0f32..2.1f32).step(1.0).values().map(|x| (x, x.sin())),
    5,
    ShapeStyle::from(&RED).filled(),
    &|coord, size, style| {
        EmptyElement::at(coord)
            + Circle::new((0, 0), size, style)
            + Text::new(format!("{:?}", coord), (0, 15), ("sans-serif", 15))
    },
)).unwrap();
/*=========================================================== */
//1행 2열로 나누기
let drawing_areas = lower.split_evenly((1, 3));

for (drawing_area, idx) in drawing_areas.iter().zip(1..) {
    let mut cc = ChartBuilder::on(drawing_area)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .margin_right(20)
        .caption(format!("y = x^{}", 1 + 2 * idx), ("sans-serif", 40))
        .build_cartesian_2d(-1f32..1f32, -1f32..1f32).unwrap();
    cc.configure_mesh()
        .x_labels(5)
        .y_labels(3)
        .max_light_lines(4)
        .draw().unwrap();

    cc.draw_series(LineSeries::new(
        (-1f32..1f32)
            .step(0.01)
            .values()
            .map(|x| (x, x.powf(idx as f32 * 2.0 + 1.0))),
        &BLUE,
    )).unwrap();
}

// To avoid the IO failure being ignored silently, we manually call the present function
root_area.present().unwrap();
println!("Result has been saved to {}", OUT_FILE_NAME);
}