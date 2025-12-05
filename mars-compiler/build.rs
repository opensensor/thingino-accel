//! Build script for Mars compiler

fn main() {
    // No special build steps needed - using the `onnx` crate for protobuf parsing
    println!("cargo:rerun-if-changed=build.rs");
}
