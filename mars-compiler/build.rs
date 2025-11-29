//! Build script to compile ONNX protobuf definitions

fn main() {
    // We'll use a pre-generated onnx.rs file instead of compiling proto at build time
    // This avoids needing protoc installed
    println!("cargo:rerun-if-changed=build.rs");
}

