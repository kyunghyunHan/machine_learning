use plotters::prelude::*;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Sample data (Replace this with your data)
    let mut train_df = HashMap::new();
    train_df.insert("Yes", 30);
    train_df.insert("No", 70);

    // Convert HashMap data into a vector of tuples
    let data: Vec<(&str, i32)> = train_df.into_iter().collect();

    // Create a drawing area that is 800 pixels in width and 600 pixels in height
    let root = BitMapBackend::new("plot.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    // Create a pie chart
    let mut  pie = ChartBuilder::on(&root)
        .caption("Target distribution", ("Arial", 30))
        .build_ranged(0.0..1.0, 0.0..1.0)?;

    // Draw the pie chart using the provided data
    pie.draw_series(
        data.iter()
            .map(|(label, value)| (*label, *value))
            .map(|(label, value)| {
                return (label, value);
            })
            .into_iter()
            .map(|(label, value)| {
                return (      , value, &BLUE);
            }),
    )?;
    
    // Save the chart to a file named "plot.png"
    root.present()?;

    Ok(())
}
