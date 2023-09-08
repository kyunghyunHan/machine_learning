extern crate serde;
mod ReadSort;
mod Polars;
mod SmartCore;
mod SimpleNeuralNetwork;
fn main() {
    println!("Hello, world!!!!");
    // ReadSort::read_csv::main();
    // Polars::polars::main();
    // SmartCore::smart_core::main();
    SimpleNeuralNetwork::simple_neural_networks::main();
}
