//! MGK Decompiler - Convert Ingenic Magik .mgk models to ONNX format
//!
//! MGK files are ELF shared libraries containing compiled neural network layers
//! for Ingenic's Neural Network Accelerator (NNA).

mod binary_parser;
mod elf_parser;
mod layer_decoder;
mod rodata_parser;
mod types;
mod weight_extractor;

use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "mgk-decompiler")]
#[command(about = "Decompile Ingenic Magik .mgk models to ONNX format")]
#[command(version)]
struct Args {
    /// Input .mgk file path
    #[arg(short, long)]
    input: PathBuf,

    /// Output JSON file for intermediate representation
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Verbose output
    #[arg(short, long, default_value_t = false)]
    verbose: bool,

    /// Dump raw layer parameters
    #[arg(long, default_value_t = false)]
    dump_params: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("MGK Decompiler v{}", env!("CARGO_PKG_VERSION"));
    println!("Input: {}", args.input.display());

    // Parse the ELF file
    let mgk = elf_parser::parse_mgk_file(&args.input)
        .with_context(|| format!("Failed to parse MGK file: {}", args.input.display()))?;

    println!("\n=== ELF Structure ===");
    println!("Architecture: {:?}", mgk.architecture);
    println!("Entry point: 0x{:08x}", mgk.entry_point);
    println!("Sections: {}", mgk.sections.len());

    // Print section info
    if args.verbose {
        println!("\n=== Sections ===");
        for section in &mgk.sections {
            println!(
                "  {:20} addr=0x{:08x} size=0x{:06x} ({:6} bytes)",
                section.name, section.address, section.size, section.size
            );
        }
    }

    // Print symbols related to layers
    println!("\n=== Layer Symbols ===");
    let layer_symbols: Vec<_> = mgk.symbols.iter()
        .filter(|s| s.demangled.contains("Layer") || s.demangled.contains("layer"))
        .collect();

    for sym in layer_symbols.iter().take(20) {
        println!("  0x{:08x}: {}", sym.address, sym.demangled);
    }
    if layer_symbols.len() > 20 {
        println!("  ... and {} more", layer_symbols.len() - 20);
    }

    // Decode layers
    println!("\n=== Detected Layers ===");
    let layers = layer_decoder::detect_layers(&mgk)?;
    for layer in &layers {
        println!("  {:3}: {:20} (op_type={})",
            layer.id, layer.layer_type, layer.op_type);
    }

    // Extract complete model metadata from rodata
    println!("\n=== Model Metadata (from .rodata) ===");
    match rodata_parser::extract_model_metadata(&mgk) {
        Ok(metadata) => {
            // Show inputs
            println!("\n  Inputs:");
            for input in &metadata.inputs {
                println!("    - {}", input);
            }

            // Show outputs
            println!("\n  Outputs:");
            for output in &metadata.outputs {
                println!("    - {}", output);
            }

            // Show layers from rodata
            println!("\n  Layers (from strings):");
            for layer in metadata.layers.iter().take(30) {
                println!("    0x{:06x}: {} (id={:?}, type={})",
                    layer.offset, layer.name, layer.layer_id, layer.layer_type);
            }
            if metadata.layers.len() > 30 {
                println!("    ... and {} more layers", metadata.layers.len() - 30);
            }

            // Show operation paths
            println!("\n  Operation Paths:");
            for op in &metadata.op_paths {
                println!("    0x{:06x}: {}/{} params={:?}",
                    op.offset, op.op_type, op.kernel_name, op.params);
            }

            // Show tensors
            println!("\n  Tensors:");
            for tensor in metadata.tensors.iter().take(20) {
                println!("    0x{:06x}: {} (format={:?}, dtype={:?})",
                    tensor.offset, tensor.name, tensor.data_format, tensor.data_type);
            }
            if metadata.tensors.len() > 20 {
                println!("    ... and {} more tensors", metadata.tensors.len() - 20);
            }
        }
        Err(e) => {
            println!("  Error extracting metadata: {}", e);
        }
    }

    // Extract layer graph
    println!("\n=== Layer Graph ===");
    match rodata_parser::extract_layer_graph(&mgk) {
        Ok(graph) => {
            for node in graph.iter().take(30) {
                println!("  {} (type={}, id={:?})",
                    node.name, node.layer_type, node.layer_id);
                if !node.outputs.is_empty() {
                    println!("    outputs: {:?}", node.outputs);
                }
            }
            if graph.len() > 30 {
                println!("  ... and {} more nodes", graph.len() - 30);
            }
        }
        Err(e) => {
            println!("  Error extracting layer graph: {}", e);
        }
    }

    // Extract weight data
    println!("\n=== Weight Data ===");
    let mut file = std::fs::File::open(&args.input)?;
    match weight_extractor::extract_weight_info(&mut file) {
        Ok(header) => {
            println!("  Appended data offset: 0x{:x}", header.data_offset);
            println!("  Appended data size: {} bytes ({:.1} KB)",
                header.data_size, header.data_size as f64 / 1024.0);
            println!("  Weight data offset: 0x{:x} (relative: 0x{:x})",
                header.data_offset + header.weights_offset, header.weights_offset);
            println!("  Weight data size: {} bytes ({:.1} KB)",
                header.weights_size, header.weights_size as f64 / 1024.0);

            println!("\n  Header entries:");
            for entry in &header.header_entries {
                println!("    0x{:04x}: {} / {}", entry.offset, entry.value1, entry.value2);
            }

            // Extract and analyze weights
            if let Ok(weights) = weight_extractor::extract_weights(&mut file, &header) {
                let stats = weight_extractor::analyze_weights(&weights);
                println!("\n  Weight statistics:");
                println!("    Total bytes: {}", stats.total_bytes);
                println!("    Value range: {:?} to {:?}", stats.min_value, stats.max_value);
                println!("    Mean value: {:.2}", stats.mean);
                println!("    Zero count: {} ({:.1}%)", stats.zero_count, stats.zero_percentage);
            }
        }
        Err(e) => {
            println!("  Error extracting weight info: {}", e);
        }
    }

    // Output JSON if requested
    if let Some(output_path) = args.output {
        // Get metadata for JSON output
        let metadata = rodata_parser::extract_model_metadata(&mgk).ok();
        let layer_graph = rodata_parser::extract_layer_graph(&mgk).ok();

        // Get weight info for JSON output
        let mut file = std::fs::File::open(&args.input)?;
        let (weight_info, weight_stats) = match weight_extractor::extract_weight_info(&mut file) {
            Ok(header) => {
                let stats = weight_extractor::extract_weights(&mut file, &header)
                    .map(|w| weight_extractor::analyze_weights(&w))
                    .ok();
                (Some(header), stats)
            }
            Err(_) => (None, None),
        };

        let output = types::DecompilerOutput {
            model_name: args.input.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string(),
            layers,
            symbols: mgk.symbols,
            metadata,
            layer_graph,
            weight_info,
            weight_stats,
        };

        let json = serde_json::to_string_pretty(&output)?;
        std::fs::write(&output_path, json)?;
        println!("\nOutput written to: {}", output_path.display());
    }

    Ok(())
}
