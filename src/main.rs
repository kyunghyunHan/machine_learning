extern crate serde;
mod ReadSort;
mod Polars;
mod SmartCore;
mod SimpleNeuralNetwork;
mod CustomNnet;
mod ConvNnet;
fn main() {
    println!("Hello, world!!!!");
    ConvNnet::conv_nnet::main();
    // CustomNnet::custom_nnet::main();
    // ReadSort::read_csv::main();
    // Polars::polars::main();
    // SmartCore::smart_core::main();
    // SimpleNeuralNetwork::simple_neural_networks::main();
}
