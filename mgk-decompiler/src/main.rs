//! MGK Decompiler - Convert Ingenic Magik .mgk models to ONNX format
//!
//! MGK files are ELF shared libraries containing compiled neural network layers
//! for Ingenic's Neural Network Accelerator (NNA).

mod aec_onnx_export;
mod binary_parser;
mod elf_parser;
mod layer_config;
mod layer_decoder;
mod onnx_export;
mod rodata_parser;
mod types;
mod weight_extractor;
mod yolo_onnx_export;

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

    /// Show summary only (compact output)
    #[arg(long, default_value_t = false)]
    summary: bool,

    /// Analyze weight structure (1024-byte blocks)
    #[arg(long, default_value_t = false)]
    analyze_weights: bool,

    /// Extract weights to directory
    #[arg(long)]
    extract_weights: Option<PathBuf>,

    /// Export to ONNX file (generic structure)
    #[arg(long)]
    onnx: Option<PathBuf>,

    /// Export AEC model to ONNX with embedded weights
    #[arg(long)]
    aec_onnx: Option<PathBuf>,

    /// Export YOLO model to ONNX with embedded weights
    #[arg(long)]
    yolo_onnx: Option<PathBuf>,

    /// Reference ONNX model for YOLO export (provides graph structure)
    #[arg(long)]
    yolo_reference: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Parse the ELF file
    let mgk = elf_parser::parse_mgk_file(&args.input)
        .with_context(|| format!("Failed to parse MGK file: {}", args.input.display()))?;

    // Get model name
    let model_name = args.input.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");

    // Handle --summary mode (compact output like Python version)
    if args.summary {
        return print_summary(&args, &mgk, model_name);
    }

    println!("MGK Decompiler v{}", env!("CARGO_PKG_VERSION"));
    println!("Input: {}", args.input.display());

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
    if !args.analyze_weights {
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

                // Analyze weight blocks if requested
                if args.analyze_weights {
                    println!("\n=== Weight Block Analysis ===");
                    let blocks = weight_extractor::analyze_weight_blocks(&weights);
                    let dense_count = blocks.iter().filter(|b| b.is_dense).count();
                    println!("  Total blocks: {} (1024 bytes each)", blocks.len());
                    println!("  Dense blocks: {} ({:.1}%)",
                        dense_count,
                        dense_count as f64 / blocks.len() as f64 * 100.0);

                    println!("\n  Dense block details:");
                    for block in blocks.iter().filter(|b| b.is_dense).take(20) {
                        println!("    Block {:4}: offset=0x{:06x}, nonzero={}, std={:.1}, range=[{}, {}]",
                            block.block_index, block.offset, block.nonzero_count,
                            block.std_dev, block.min_val, block.max_val);
                    }
                    if dense_count > 20 {
                        println!("    ... and {} more dense blocks", dense_count - 20);
                    }

                    // Detect boundaries
                    let boundaries = weight_extractor::detect_weight_boundaries(&weights);
                    println!("\n  Weight boundaries detected: {}", boundaries.len());
                    for (i, boundary) in boundaries.iter().take(10).enumerate() {
                        println!("    Boundary {}: 0x{:06x}", i, boundary);
                    }
                }
            }
        }
        Err(e) => {
            println!("  Error extracting weight info: {}", e);
        }
    }

    // Show known layer weight mappings
    if !args.analyze_weights {
        println!("\n=== Known Layer Weight Mappings ===");
        let layer_mappings = weight_extractor::get_known_layer_mappings();
        for mapping in &layer_mappings {
            println!("  0x{:05x}: {} ({} bytes)",
                mapping.offset, mapping.layer_name, mapping.size);
        }
    }

    // Extract weights to directory if requested
    if let Some(ref extract_dir) = args.extract_weights {
        extract_weights_to_dir(&args.input, extract_dir)?;
    }

    // Export to ONNX if requested
    if let Some(ref onnx_path) = args.onnx {
        println!("\n=== Exporting to ONNX ===");
        println!("Output: {}", onnx_path.display());

        // Get metadata for export
        let metadata = rodata_parser::extract_model_metadata(&mgk)
            .unwrap_or_default();

        // Get weight data
        let mut file = std::fs::File::open(&args.input)?;
        let weight_data = match weight_extractor::extract_weight_info(&mut file) {
            Ok(header) => weight_extractor::extract_weights(&mut file, &header).unwrap_or_default(),
            Err(_) => Vec::new(),
        };

        match onnx_export::export_to_onnx(
            &model_name,
            &metadata,
            &weight_data,
            onnx_path,
        ) {
            Ok(()) => println!("ONNX model saved successfully!"),
            Err(e) => println!("ONNX export failed: {}", e),
        }
    }

    // Export AEC model with embedded weights
    if let Some(ref aec_onnx_path) = args.aec_onnx {
        println!("\n=== Exporting AEC Model to ONNX (with weights) ===");
        println!("Output: {}", aec_onnx_path.display());

        // Extract scales from metadata file if available
        let metadata_dir = args.input.parent().unwrap_or(std::path::Path::new("."));
        let metadata_path = metadata_dir.join("extracted_aec_full").join("metadata.json");

        let scales = aec_onnx_export::extract_scales_from_metadata(&metadata_path)
            .unwrap_or_else(|_| {
                // Fall back to extracting from rodata
                let metadata = rodata_parser::extract_model_metadata(&mgk).unwrap_or_default();
                metadata.scale_groups.iter()
                    .flat_map(|g| g.scales.iter().copied())
                    .filter(|s| *s > 0.005 && *s < 0.1)
                    .collect()
            });

        // Extract weights from MGK
        let mut file = std::fs::File::open(&args.input)?;
        let header = weight_extractor::extract_weight_info(&mut file)?;
        let weight_data = weight_extractor::extract_weights(&mut file, &header)?;
        let weights: Vec<i8> = weight_data.iter().map(|&b| b as i8).collect();

        println!("  Weights: {} bytes", weights.len());
        println!("  Scales: {} values", scales.len());

        // Create and export AEC model
        let exporter = aec_onnx_export::AecOnnxExporter::new(
            model_name.to_string(),
            weights,
            scales,
        );

        match exporter.export(aec_onnx_path) {
            Ok(()) => println!("AEC ONNX model saved successfully!"),
            Err(e) => println!("AEC ONNX export failed: {}", e),
        }
    }

    // Export YOLO model with embedded weights
    if let Some(ref yolo_onnx_path) = args.yolo_onnx {
        println!("\n=== Exporting YOLOv5s Model to ONNX (with weights) ===");
        println!("Output: {}", yolo_onnx_path.display());

        let reference = args.yolo_reference.as_deref();

        match yolo_onnx_export::export_yolov5s_onnx(&args.input, yolo_onnx_path, reference) {
            Ok(()) => println!("YOLOv5s ONNX model saved successfully!"),
            Err(e) => println!("YOLOv5s ONNX export failed: {}", e),
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

        // Extract model config if metadata is available
        let model_config = metadata.as_ref().map(|m| {
            layer_config::extract_layer_configs(m, weight_info.as_ref())
        });

        let layer_mappings = weight_extractor::get_known_layer_mappings();

        let output = types::DecompilerOutput {
            model_name: model_name.to_string(),
            layers,
            symbols: mgk.symbols,
            metadata,
            layer_graph,
            weight_info,
            weight_stats,
            model_config,
            layer_weight_mappings: Some(layer_mappings),
        };

        let json = serde_json::to_string_pretty(&output)?;
        std::fs::write(&output_path, json)?;
        println!("\nOutput written to: {}", output_path.display());
    }

    Ok(())
}

/// Print compact summary (like Python version)
fn print_summary(args: &Args, mgk: &types::MgkFile, model_name: &str) -> Result<()> {
    println!("Model: {}", model_name);

    // Get metadata
    let metadata = rodata_parser::extract_model_metadata(mgk)?;

    println!("Layers: {}", metadata.layers.len());
    println!("Scales: {}", metadata.scale_groups.iter().map(|g| g.scales.len()).sum::<usize>());

    // Get weight info
    let mut file = std::fs::File::open(&args.input)?;
    if let Ok(header) = weight_extractor::extract_weight_info(&mut file) {
        println!("Weight region: {} bytes", header.weights_size);
    }

    // Show tensor formats and data types
    if !metadata.tensor_formats.is_empty() {
        println!("Tensor formats: {}", metadata.tensor_formats.join(", "));
    }
    if !metadata.data_types.is_empty() {
        println!("Data types: {}", metadata.data_types.join(", "));
    }

    // Count layer types
    println!("\nLayer types:");
    let mut type_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for layer in &metadata.layers {
        *type_counts.entry(layer.layer_type.clone()).or_insert(0) += 1;
    }
    let mut counts: Vec<_> = type_counts.into_iter().collect();
    counts.sort_by(|a, b| b.1.cmp(&a.1));
    for (layer_type, count) in counts {
        println!("  {}: {}", layer_type, count);
    }

    // Count fused layers
    let fused_count = metadata.layers.iter().filter(|l| l.is_fused == Some(true)).count();
    if fused_count > 0 {
        println!("\nFused layers: {}", fused_count);
    }

    Ok(())
}

/// Extract weights to a directory
fn extract_weights_to_dir(input: &PathBuf, output_dir: &PathBuf) -> Result<()> {
    use std::io::Write;

    // Create output directory
    std::fs::create_dir_all(output_dir)?;

    let mut file = std::fs::File::open(input)?;
    let header = weight_extractor::extract_weight_info(&mut file)?;
    let weights = weight_extractor::extract_weights(&mut file, &header)?;

    // Analyze blocks and extract dense regions
    let blocks = weight_extractor::analyze_weight_blocks(&weights);
    let boundaries = weight_extractor::detect_weight_boundaries(&weights);

    println!("\n=== Extracting Weights ===");
    println!("Output directory: {}", output_dir.display());
    println!("Total weight bytes: {}", weights.len());
    println!("Dense blocks: {}", blocks.iter().filter(|b| b.is_dense).count());

    // Extract each region between boundaries
    for (i, window) in boundaries.windows(2).enumerate() {
        let start = window[0] as usize;
        let end = window[1] as usize;
        let region = &weights[start..end];

        // Check if this region is mostly dense
        let region_blocks: Vec<_> = blocks.iter()
            .filter(|b| b.offset >= start as u64 && b.offset < end as u64)
            .collect();
        let dense_ratio = region_blocks.iter().filter(|b| b.is_dense).count() as f64
            / region_blocks.len().max(1) as f64;

        if dense_ratio > 0.5 {
            let filename = format!("weights_{:03}_0x{:06x}.bin", i, start);
            let path = output_dir.join(&filename);
            let mut out_file = std::fs::File::create(&path)?;
            out_file.write_all(region)?;
            println!("  Extracted: {} ({} bytes, {:.0}% dense)",
                filename, region.len(), dense_ratio * 100.0);
        }
    }

    // Also extract the last region
    if let Some(&last_boundary) = boundaries.last() {
        let start = last_boundary as usize;
        if start < weights.len() {
            let region = &weights[start..];
            let filename = format!("weights_{:03}_0x{:06x}.bin", boundaries.len() - 1, start);
            let path = output_dir.join(&filename);
            let mut out_file = std::fs::File::create(&path)?;
            out_file.write_all(region)?;
            println!("  Extracted: {} ({} bytes)", filename, region.len());
        }
    }

    println!("\nWeight extraction complete.");
    Ok(())
}
