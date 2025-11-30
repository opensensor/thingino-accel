//! Mars Compiler - ONNX to Mars format converter
//!
//! Compiles ONNX models to .mars format for Ingenic T41 NNA

mod mars_format;
mod onnx_parser;

use anyhow::{Context, Result};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use mars_format::*;
use onnx_parser::{OnnxModel, OnnxNode, TensorDataType};

/// Convert IEEE 754 half-precision float (16-bit) to single-precision float (32-bit)
fn half_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        // Subnormal or zero
        if mant == 0 {
            return if sign == 1 { -0.0 } else { 0.0 };
        }
        // Subnormal: normalize
        let f = (mant as f32) / 1024.0 * (2.0_f32).powi(-14);
        return if sign == 1 { -f } else { f };
    } else if exp == 31 {
        // Inf or NaN
        if mant == 0 {
            return if sign == 1 { f32::NEG_INFINITY } else { f32::INFINITY };
        }
        return f32::NAN;
    }

    // Normal number
    let exp32 = exp + 127 - 15;  // Adjust exponent bias: fp16 bias=15, fp32 bias=127
    let mant32 = mant << 13;     // Shift mantissa: fp16 has 10 bits, fp32 has 23 bits
    let bits32 = (sign << 31) | (exp32 << 23) | mant32;
    f32::from_bits(bits32)
}

#[derive(Parser, Debug)]
#[command(name = "mars")]
#[command(about = "Compile ONNX models to Mars format for Ingenic T41 NNA")]
#[command(version)]
struct Args {
    /// Input ONNX model file
    #[arg(short, long)]
    input: PathBuf,

    /// Output .mars file
    #[arg(short, long)]
    output: PathBuf,

    /// Keep weights as float32 (don't quantize to int8)
    #[arg(short, long)]
    float32: bool,

    /// Use NHWC format for features (channels-last, faster gather)
    /// Default is NCHW (channels-first, ONNX native)
    #[arg(long)]
    nhwc: bool,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

/// Maps ONNX operator to Mars layer type
fn map_onnx_op_to_mars(op_type: &str) -> Option<LayerType> {
    match op_type {
        "Conv" => Some(LayerType::Conv2d),
        "MaxPool" => Some(LayerType::MaxPool),
        "AveragePool" | "GlobalAveragePool" => Some(LayerType::AvgPool),
        "Relu" => Some(LayerType::Relu),
        "LeakyRelu" => Some(LayerType::LeakyRelu),
        "Sigmoid" => Some(LayerType::Sigmoid),
        "Mul" => Some(LayerType::Mul),
        "Add" => Some(LayerType::Add),
        "Concat" => Some(LayerType::Concat),
        "Resize" | "Upsample" => Some(LayerType::Upsample),
        "Reshape" => Some(LayerType::Reshape),
        "Transpose" => Some(LayerType::Transpose),
        "Softmax" => Some(LayerType::Softmax),
        // BatchNorm should be folded into Conv, but keep as separate layer for now
        "BatchNormalization" => Some(LayerType::BatchNorm),
        // Skip these ops - they get folded or are handled differently
        // Pow is used in SiLU (x * sigmoid(x)) but we handle via Sigmoid+Mul
        // QuantizeLinear/DequantizeLinear are QDQ ops - we extract scales but skip the ops
        "Constant" | "Shape" | "Gather" | "Slice" | "Split" | "Sub" | "Div" | "Unsqueeze" | "Pow"
            | "QuantizeLinear" | "DequantizeLinear" => None,
        _ => {
            eprintln!("Warning: Unknown op type: {}", op_type);
            None
        }
    }
}

/// Compiler state for ONNX to Mars conversion
struct MarsCompiler {
    onnx: OnnxModel,
    tensors: Vec<MarsTensor>,
    layers: Vec<MarsLayer>,
    weights_data: Vec<u8>,
    tensor_map: HashMap<String, u32>,  // ONNX tensor name -> Mars tensor ID
    qdq_scales: HashMap<String, f32>,  // QDQ scale name -> scale value
    has_qdq: bool,  // Model has QDQ (QuantizeLinear/DequantizeLinear) nodes
    quantize: bool,
    use_nhwc: bool,  // Use NHWC format for features (faster gather on device)
    verbose: bool,
}

impl MarsCompiler {
    fn new(onnx: OnnxModel, quantize: bool, use_nhwc: bool, verbose: bool) -> Self {
        Self {
            onnx,
            tensors: Vec::new(),
            layers: Vec::new(),
            weights_data: Vec::new(),
            tensor_map: HashMap::new(),
            qdq_scales: HashMap::new(),
            has_qdq: false,
            quantize,
            use_nhwc,
            verbose,
        }
    }

    /// Parse QDQ (QuantizeLinear/DequantizeLinear) scales from the model
    /// These come from calibration and provide proper per-tensor scales
    fn parse_qdq_scales(&mut self) -> Result<()> {
        // Check if model has QDQ nodes
        let qdq_count = self.onnx.nodes.iter()
            .filter(|n| n.op_type == "QuantizeLinear" || n.op_type == "DequantizeLinear")
            .count();

        if qdq_count == 0 {
            if self.verbose {
                println!("No QDQ nodes found - using heuristic quantization");
            }
            return Ok(());
        }

        self.has_qdq = true;
        if self.verbose {
            println!("Found {} QDQ nodes - using calibrated scales", qdq_count);
        }

        // Extract scales from initializers (they have names ending in "_scale")
        for (name, tensor) in &self.onnx.initializers {
            if name.ends_with("_scale") {
                // Scale is a scalar float32 - may be in raw_data or float_data
                let scale = if !tensor.data.is_empty() {
                    if tensor.data.len() >= 4 {
                        f32::from_le_bytes([tensor.data[0], tensor.data[1],
                                           tensor.data[2], tensor.data[3]])
                    } else if tensor.data.len() >= 2 {
                        // Float16 scale
                        let bits = u16::from_le_bytes([tensor.data[0], tensor.data[1]]);
                        half_to_f32(bits)
                    } else {
                        continue;
                    }
                } else if !tensor.float_data.is_empty() {
                    // Scale stored in float_data field
                    tensor.float_data[0]
                } else {
                    continue;
                };

                // Map scale to the tensor it quantizes
                // E.g., "images_scale" -> "images", "model.0.conv.weight_scale" -> "model.0.conv.weight"
                let tensor_name = name.trim_end_matches("_scale");
                self.qdq_scales.insert(tensor_name.to_string(), scale);

                if self.verbose && self.qdq_scales.len() <= 10 {
                    println!("  QDQ scale: {} = {}", tensor_name, scale);
                }
            }
        }

        // Also parse QuantizeLinear nodes to find scale mappings for shared scales
        // E.g., MaxPool output might use the same scale as a previous Conv output
        for node in &self.onnx.nodes {
            if node.op_type == "QuantizeLinear" {
                // QuantizeLinear has inputs: [tensor, scale, zero_point]
                if node.inputs.len() >= 2 {
                    let input_tensor = &node.inputs[0];
                    let scale_name = &node.inputs[1];

                    // If the scale is in qdq_scales (by its base name), map this tensor to it
                    let scale_base = scale_name.trim_end_matches("_scale");
                    if let Some(&scale) = self.qdq_scales.get(scale_base) {
                        // Map the input tensor to this scale
                        if !self.qdq_scales.contains_key(input_tensor) {
                            self.qdq_scales.insert(input_tensor.clone(), scale);
                        }
                    }
                }
            }
        }

        if self.verbose {
            println!("Loaded {} QDQ scales (including shared)", self.qdq_scales.len());
        }

        Ok(())
    }

    /// Get QDQ scale for a tensor name, trying various name patterns
    fn get_qdq_scale(&self, name: &str) -> Option<f32> {
        // Direct lookup
        if let Some(&scale) = self.qdq_scales.get(name) {
            return Some(scale);
        }

        // Try with common suffixes stripped
        // Conv output patterns: "/model.0/conv/Conv_output_0" or "model.0.conv.output"

        // For DequantizeLinear outputs: "images_DequantizeLinear_Output" -> "images"
        // or "model.0.conv.weight_DequantizeLinear_Output" -> "model.0.conv.weight"
        if name.ends_with("_DequantizeLinear_Output") {
            let base = name.trim_end_matches("_DequantizeLinear_Output");
            if let Some(&scale) = self.qdq_scales.get(base) {
                return Some(scale);
            }
        }

        // For QuantizeLinear outputs: "images_QuantizeLinear_Output" -> "images"
        if name.ends_with("_QuantizeLinear_Output") {
            let base = name.trim_end_matches("_QuantizeLinear_Output");
            if let Some(&scale) = self.qdq_scales.get(base) {
                return Some(scale);
            }
        }

        // For QuantizeLinear inputs: "output0_QuantizeLinear_Input" -> "output0"
        if name.ends_with("_QuantizeLinear_Input") {
            let base = name.trim_end_matches("_QuantizeLinear_Input");
            if let Some(&scale) = self.qdq_scales.get(base) {
                return Some(scale);
            }
        }

        // For quantized weights: check if there's a scale for the base name
        if name.ends_with("_quantized") {
            let base = name.trim_end_matches("_quantized");
            if let Some(&scale) = self.qdq_scales.get(base) {
                return Some(scale);
            }
        }

        None
    }

    /// Compile the ONNX model to Mars format
    fn compile(&mut self) -> Result<()> {
        if self.verbose {
            self.onnx.print_summary();
            println!("\nCompiling to Mars format...\n");
        }

        // Step 0: Parse QDQ scales if present
        self.parse_qdq_scales()?;

        // Step 1: Create input tensors
        self.create_input_tensors()?;
        
        // Step 2: Process each ONNX node
        let pb = if self.verbose {
            let pb = ProgressBar::new(self.onnx.nodes.len() as u64);
            pb.set_style(ProgressStyle::default_bar()
                .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} {msg}")
                .unwrap()
                .progress_chars("#>-"));
            Some(pb)
        } else {
            None
        };
        
        // Clone nodes to avoid borrow issues
        let nodes: Vec<_> = self.onnx.nodes.iter().cloned().collect();
        for (idx, node) in nodes.iter().enumerate() {
            if let Some(ref pb) = pb {
                pb.set_position(idx as u64);
                pb.set_message(format!("{}: {}", node.op_type, node.name));
            }
            self.process_node(node)?;
        }
        
        if let Some(pb) = pb {
            pb.finish_with_message("Done");
        }

        // Step 3: Propagate scales through the graph
        // Some tensors may not have scales set if they were created before their producer layer
        self.propagate_scales();

        // Step 4: Mark output tensors
        self.mark_outputs()?;

        Ok(())
    }

    /// Propagate scales through the graph for tensors that don't have explicit QDQ scales
    fn propagate_scales(&mut self) {
        // Run multiple iterations to handle chains of layers
        for _iter in 0..5 {
            let mut any_updated = false;

            for layer_idx in 0..self.layers.len() {
                let layer = &self.layers[layer_idx];
                let layer_type = layer.layer_type.clone();
                let output_id = layer.output_tensor_ids[0];
                let num_inputs = layer.num_inputs as usize;
                let input_ids: Vec<u32> = (0..num_inputs)
                    .map(|i| layer.input_tensor_ids[i])
                    .collect();

                let out_scale = self.tensors.get(output_id as usize).map(|t| t.scale).unwrap_or(1.0);

                // Skip if output already has a non-default scale
                if (out_scale - 1.0).abs() > 0.0001 {
                    continue;
                }

                // Compute the scale based on layer type
                let new_scale = match layer_type {
                    // These layer types preserve input scale
                    LayerType::Reshape | LayerType::Transpose | LayerType::Softmax
                    | LayerType::MaxPool | LayerType::AvgPool | LayerType::Upsample => {
                        let in_scale = self.tensors.get(input_ids[0] as usize)
                            .map(|t| t.scale).unwrap_or(1.0);
                        if (in_scale - 1.0).abs() > 0.0001 {
                            Some(in_scale)
                        } else {
                            None
                        }
                    },
                    // Concat uses max of input scales
                    LayerType::Concat => {
                        let mut max_scale = 0.0f32;
                        for &tid in &input_ids {
                            let s = self.tensors.get(tid as usize).map(|t| t.scale).unwrap_or(1.0);
                            if (s - 1.0).abs() > 0.0001 {
                                max_scale = max_scale.max(s);
                            }
                        }
                        if max_scale > 0.0001 {
                            Some(max_scale)
                        } else {
                            None
                        }
                    },
                    // Add preserves input scale (same scale for both inputs ideally)
                    LayerType::Add => {
                        let s1 = self.tensors.get(input_ids[0] as usize).map(|t| t.scale).unwrap_or(1.0);
                        let s2 = if num_inputs > 1 {
                            self.tensors.get(input_ids[1] as usize).map(|t| t.scale).unwrap_or(1.0)
                        } else { 1.0 };
                        let max_s = s1.max(s2);
                        if (max_s - 1.0).abs() > 0.0001 {
                            Some(max_s)
                        } else {
                            None
                        }
                    },
                    // Mul output scale is product of input scales
                    LayerType::Mul => {
                        let s1 = self.tensors.get(input_ids[0] as usize).map(|t| t.scale).unwrap_or(1.0);
                        let s2 = if num_inputs > 1 {
                            self.tensors.get(input_ids[1] as usize).map(|t| t.scale).unwrap_or(1.0)
                        } else { 1.0 };
                        if (s1 - 1.0).abs() > 0.0001 && (s2 - 1.0).abs() > 0.0001 {
                            Some(s1 * s2)
                        } else if (s1 - 1.0).abs() > 0.0001 {
                            Some(s1)
                        } else if (s2 - 1.0).abs() > 0.0001 {
                            Some(s2)
                        } else {
                            None
                        }
                    },
                    _ => None,
                };

                if let Some(scale) = new_scale {
                    if let Some(tensor) = self.tensors.get_mut(output_id as usize) {
                        tensor.scale = scale;
                        any_updated = true;
                    }
                }
            }

            if !any_updated {
                break;
            }
        }
    }
    
    fn create_input_tensors(&mut self) -> Result<()> {
        for input in &self.onnx.inputs {
            let id = self.tensors.len() as u32;

            let mut tensor = MarsTensor::new(id, &input.name);
            tensor.ndims = input.dims.len() as u32;

            if self.use_nhwc && input.dims.len() == 4 {
                // Convert shape from NCHW to NHWC: [N,C,H,W] -> [N,H,W,C]
                tensor.shape[0] = input.dims[0].max(1) as i32;  // N
                tensor.shape[1] = input.dims[2].max(1) as i32;  // H
                tensor.shape[2] = input.dims[3].max(1) as i32;  // W
                tensor.shape[3] = input.dims[1].max(1) as i32;  // C
                tensor.format = DataFormat::Nhwc;
            } else {
                // Keep NCHW format (ONNX native)
                for (i, &dim) in input.dims.iter().enumerate() {
                    if i < MARS_MAX_DIMS {
                        tensor.shape[i] = dim.max(1) as i32;
                    }
                }
                tensor.format = DataFormat::Nchw;
            }
            tensor.dtype = if self.quantize { DataType::Int8 } else { DataType::Float32 };

            // Set input scale for INT8
            if self.quantize {
                if let Some(scale) = self.get_qdq_scale(&input.name) {
                    // Use QDQ calibrated scale
                    tensor.scale = scale;
                    if self.verbose {
                        println!("  Using QDQ input scale: {}", scale);
                    }
                } else {
                    // Fallback: assume input is normalized to [0, 1]
                    // Scale = 1/255 for typical image input
                    tensor.scale = 1.0 / 255.0;
                }
            }

            if self.verbose {
                let fmt = if self.use_nhwc { "NHWC" } else { "NCHW" };
                let ndims = tensor.ndims as usize;
                println!("Input tensor {}: {} {:?} ({})", id, input.name,
                    &tensor.shape[..ndims], fmt);
            }

            self.tensor_map.insert(input.name.clone(), id);
            self.tensors.push(tensor);
        }
        Ok(())
    }
    
    fn mark_outputs(&mut self) -> Result<()> {
        for output in &self.onnx.outputs {
            if let Some(&id) = self.tensor_map.get(&output.name) {
                if self.verbose {
                    println!("Output tensor {}: {}", id, output.name);
                }
            }
        }
        Ok(())
    }

    fn process_node(&mut self, node: &OnnxNode) -> Result<()> {
        let layer_type = match map_onnx_op_to_mars(&node.op_type) {
            Some(lt) => lt,
            None => return Ok(()),  // Skip unsupported ops
        };

        match layer_type {
            LayerType::Conv2d => self.process_conv(node)?,
            LayerType::MaxPool | LayerType::AvgPool => self.process_pool(node, layer_type)?,
            LayerType::Relu | LayerType::Sigmoid | LayerType::LeakyRelu => self.process_activation(node, layer_type)?,
            LayerType::Add | LayerType::Mul => self.process_elementwise(node, layer_type)?,
            LayerType::Concat => self.process_concat(node)?,
            LayerType::Upsample => self.process_upsample(node)?,
            LayerType::BatchNorm => self.process_batchnorm(node)?,
            LayerType::Reshape => self.process_reshape(node)?,
            LayerType::Transpose => self.process_transpose(node)?,
            LayerType::Softmax => self.process_softmax(node)?,
            _ => {
                if self.verbose {
                    println!("Skipping layer type: {:?}", layer_type);
                }
            }
        }

        Ok(())
    }

    /// Create or get a feature tensor
    fn get_or_create_tensor(&mut self, name: &str) -> u32 {
        if let Some(&id) = self.tensor_map.get(name) {
            return id;
        }

        let id = self.tensors.len() as u32;
        let mut tensor = MarsTensor::new(id, name);
        tensor.dtype = if self.quantize { DataType::Int8 } else { DataType::Float32 };

        // Use NHWC or NCHW based on compiler settings
        tensor.format = if self.use_nhwc { DataFormat::Nhwc } else { DataFormat::Nchw };

        // Try to get shape from ONNX shape_info (ONNX uses NCHW)
        // For QDQ models, tensor names may have suffixes like _DequantizeLinear_Output
        let shape_name = if self.onnx.shape_info.contains_key(name) {
            name.to_string()
        } else if name.ends_with("_DequantizeLinear_Output") {
            name.trim_end_matches("_DequantizeLinear_Output").to_string()
        } else if name.ends_with("_QuantizeLinear_Output") {
            name.trim_end_matches("_QuantizeLinear_Output").to_string()
        } else if name.ends_with("_QuantizeLinear_Input") {
            name.trim_end_matches("_QuantizeLinear_Input").to_string()
        } else {
            name.to_string()
        };

        if let Some(dims) = self.onnx.shape_info.get(&shape_name) {
            tensor.ndims = dims.len() as u32;
            if self.use_nhwc && dims.len() == 4 {
                // Convert NCHW -> NHWC shape
                tensor.shape[0] = dims[0].max(1) as i32;  // N
                tensor.shape[1] = dims[2].max(1) as i32;  // H
                tensor.shape[2] = dims[3].max(1) as i32;  // W
                tensor.shape[3] = dims[1].max(1) as i32;  // C
            } else {
                for (i, &dim) in dims.iter().enumerate() {
                    if i < MARS_MAX_DIMS {
                        tensor.shape[i] = dim.max(1) as i32;
                    }
                }
            }
        }

        // Set QDQ scale if available (for intermediate tensors in QDQ models)
        if self.quantize {
            if let Some(scale) = self.get_qdq_scale(name) {
                tensor.scale = scale;
            }
        }

        self.tensor_map.insert(name.to_string(), id);
        self.tensors.push(tensor);
        id
    }

    /// Create a weight tensor (uses NMHWSOIB2 format for NNA)
    fn create_weight_tensor(&mut self, name: &str, out_ch: usize, in_ch: usize, kh: usize, kw: usize) -> u32 {
        let id = self.tensors.len() as u32;
        let mut tensor = MarsTensor::new(id, name);
        tensor.dtype = if self.quantize { DataType::Int8 } else { DataType::Float32 };
        // Use NNA native format for weights: NMHWSOIB2
        tensor.format = DataFormat::Nmhwsoib2;
        tensor.ndims = 4;
        tensor.shape[0] = out_ch as i32;
        tensor.shape[1] = in_ch as i32;
        tensor.shape[2] = kh as i32;
        tensor.shape[3] = kw as i32;

        self.tensor_map.insert(name.to_string(), id);
        self.tensors.push(tensor);
        id
    }

    /// Update tensor scale (for quantized tensors)
    fn set_tensor_scale(&mut self, tensor_id: u32, scale: f32) {
        if let Some(tensor) = self.tensors.get_mut(tensor_id as usize) {
            if self.verbose && (scale - 1.0).abs() < 0.001 {
                eprintln!("  [WARN] Setting tensor {} ({}) scale to 1.0 (default)",
                         tensor_id, tensor.name);
            }
            tensor.scale = scale;
        }
    }

    /// Get tensor scale
    fn get_tensor_scale(&self, tensor_id: u32) -> f32 {
        self.tensors.get(tensor_id as usize).map(|t| t.scale).unwrap_or(1.0)
    }

    /// Update tensor shape (for output tensors computed from layer params)
    fn update_tensor_shape(&mut self, tensor_id: u32, shape: &[i32]) {
        if let Some(tensor) = self.tensors.get_mut(tensor_id as usize) {
            if tensor.ndims == 0 || tensor.shape[0] == 0 {
                tensor.ndims = shape.len() as u32;
                for (i, &dim) in shape.iter().enumerate() {
                    if i < MARS_MAX_DIMS {
                        tensor.shape[i] = dim;
                    }
                }
            }
        }
    }

    /// Get tensor shape
    fn get_tensor_shape(&self, tensor_id: u32) -> [i32; 4] {
        if let Some(tensor) = self.tensors.get(tensor_id as usize) {
            [tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3]]
        } else {
            [0, 0, 0, 0]
        }
    }

    fn add_weights(&mut self, data: &[u8]) -> (u64, u64) {
        let offset = self.weights_data.len() as u64;
        self.weights_data.extend_from_slice(data);
        // Align to 4 bytes
        while self.weights_data.len() % 4 != 0 {
            self.weights_data.push(0);
        }
        (offset, data.len() as u64)
    }

    fn quantize_weights(&self, data: &[u8], dtype: TensorDataType) -> (Vec<u8>, f32) {
        if !self.quantize {
            return (data.to_vec(), 1.0);
        }

        // Convert to f32 based on source data type
        let floats: Vec<f32> = match dtype {
            TensorDataType::Float => {
                // Float32: 4 bytes per element
                data.chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect()
            }
            TensorDataType::Float16 => {
                // Float16: 2 bytes per element, need to convert to f32
                data.chunks_exact(2)
                    .map(|b| {
                        let bits = u16::from_le_bytes([b[0], b[1]]);
                        half_to_f32(bits)
                    })
                    .collect()
            }
            TensorDataType::Int8 => {
                // Already INT8 - just return as-is with scale 1.0
                // This shouldn't normally happen but handle gracefully
                return (data.to_vec(), 1.0 / 127.0);
            }
            _ => {
                // Unknown type - try as float32
                eprintln!("  Warning: Unknown dtype {:?}, trying as float32", dtype);
                data.chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect()
            }
        };

        // Debug: print first few float values
        if self.verbose && !floats.is_empty() {
            let sample: Vec<f32> = floats.iter().take(8).copied().collect();
            eprintln!("    First 8 floats: {:?}", sample);
        }

        // Find scale
        let max_abs = floats.iter().map(|f| f.abs()).fold(0.0f32, f32::max);
        let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };

        if self.verbose {
            eprintln!("    max_abs={} scale={}", max_abs, scale);
        }

        // Quantize to int8
        let quantized: Vec<u8> = floats.iter()
            .map(|&f| ((f / scale).round().clamp(-127.0, 127.0) as i8) as u8)
            .collect();

        (quantized, scale)
    }

    /// Convert raw bytes to f32 vector
    fn bytes_to_f32(data: &[u8]) -> Vec<f32> {
        data.chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    }

    fn process_conv(&mut self, node: &OnnxNode) -> Result<()> {
        let layer_id = self.layers.len() as u32;

        // Get input tensor
        let input_name = node.inputs.get(0).context("Conv missing input")?;
        let input_id = self.get_or_create_tensor(input_name);

        // Get weight tensor - handle QDQ models where weight input is DequantizeLinear output
        let weight_input_name = node.inputs.get(1).context("Conv missing weight")?;

        // For QDQ models: weight input is "model.0.conv.weight_DequantizeLinear_Output"
        // We need to find the quantized weight "model.0.conv.weight_quantized"
        let (weight_name, weight_tensor, qdq_weight_scale) = if self.has_qdq {
            // Try to find quantized weight by stripping "_DequantizeLinear_Output" suffix
            let base_name = weight_input_name.trim_end_matches("_DequantizeLinear_Output");
            let quant_name = format!("{}_quantized", base_name);

            if let Some(quant_tensor) = self.onnx.initializers.get(&quant_name) {
                // Found quantized weights - also get the scale
                let scale = self.get_qdq_scale(base_name);
                if self.verbose && layer_id < 3 {
                    println!("  Found QDQ quantized weights: {} (scale={:?})", quant_name, scale);
                }
                (quant_name, quant_tensor.clone(), scale)
            } else if let Some(tensor) = self.onnx.initializers.get(weight_input_name) {
                // Fallback to direct lookup
                (weight_input_name.clone(), tensor.clone(), None)
            } else {
                anyhow::bail!("Conv weight not found: {}", weight_input_name);
            }
        } else {
            // Non-QDQ model: direct lookup
            let tensor = self.onnx.initializers.get(weight_input_name)
                .context("Conv weight not found in initializers")?;
            (weight_input_name.clone(), tensor.clone(), None)
        };

        // Debug: print weight tensor info for first few convs
        if self.verbose && layer_id < 3 {
            eprintln!("  Conv[{}] weights: dtype={:?} dims={:?} bytes={}",
                     layer_id, weight_tensor.data_type, weight_tensor.dims, weight_tensor.data.len());
            let first_bytes: Vec<u8> = weight_tensor.data.iter().take(16).copied().collect();
            eprintln!("    First 16 bytes: {:?}", first_bytes);
        }

        // Get kernel shape from weights [O, I, H, W]
        let out_ch = weight_tensor.dims.get(0).copied().unwrap_or(1) as u32;
        let in_ch = weight_tensor.dims.get(1).copied().unwrap_or(1) as u32;
        let kh = weight_tensor.dims.get(2).copied().unwrap_or(3) as u32;
        let kw = weight_tensor.dims.get(3).copied().unwrap_or(3) as u32;

        // Process weights - pack based on quantization and feature format
        let (weight_data, scale, weight_format) = if self.quantize {
            // Check if weights are already INT8 (from QDQ model)
            if weight_tensor.data_type == TensorDataType::Int8 {
                // Already quantized - use directly with QDQ scale
                let scale = qdq_weight_scale.unwrap_or(1.0 / 127.0);
                let weights = weight_tensor.data.clone();
                if self.use_nhwc {
                    let ohwi_weights = convert_oihw_to_ohwi(&weights,
                        out_ch as usize, in_ch as usize, kh as usize, kw as usize);
                    (ohwi_weights, scale, DataFormat::Ohwi)
                } else {
                    (weights, scale, DataFormat::Oihw)
                }
            } else {
                // Float weights - quantize them
                let (quant_weights, scale) = self.quantize_weights(&weight_tensor.data, weight_tensor.data_type);
                if self.use_nhwc {
                    let ohwi_weights = convert_oihw_to_ohwi(&quant_weights,
                        out_ch as usize, in_ch as usize, kh as usize, kw as usize);
                    (ohwi_weights, scale, DataFormat::Ohwi)
                } else {
                    (quant_weights, scale, DataFormat::Oihw)
                }
            }
        } else {
            // Float32: Store as-is in OIHW format (standard ONNX layout)
            (weight_tensor.data.clone(), 1.0, DataFormat::Oihw)
        };
        let (weight_offset, weight_size) = self.add_weights(&weight_data);

        // Create weight tensor
        let weight_id = self.tensors.len() as u32;
        let mut w_tensor = MarsTensor::new(weight_id, &weight_name);
        w_tensor.dtype = if self.quantize { DataType::Int8 } else { DataType::Float32 };
        w_tensor.format = weight_format;
        w_tensor.ndims = 4;
        w_tensor.shape[0] = out_ch as i32;
        w_tensor.shape[1] = in_ch as i32;
        w_tensor.shape[2] = kh as i32;
        w_tensor.shape[3] = kw as i32;
        w_tensor.scale = scale;
        w_tensor.data_offset = weight_offset;
        w_tensor.data_size = weight_size;
        self.tensor_map.insert(weight_name.clone(), weight_id);
        self.tensors.push(w_tensor);

        // Handle bias if present
        let bias_id = if let Some(bias_name) = node.inputs.get(2) {
            if let Some(bias_tensor) = self.onnx.initializers.get(bias_name) {
                // Clone bias data to avoid borrow conflict
                let bias_data = bias_tensor.data.clone();
                // Bias stored as float32 in both modes
                let (bias_offset, bias_size) = self.add_weights(&bias_data);

                let bid = self.tensors.len() as u32;
                let mut b_tensor = MarsTensor::new(bid, bias_name);
                b_tensor.dtype = DataType::Float32;  // Bias always float32
                b_tensor.ndims = 1;
                b_tensor.shape[0] = out_ch as i32;
                b_tensor.data_offset = bias_offset;
                b_tensor.data_size = bias_size;
                self.tensors.push(b_tensor);
                bid
            } else {
                0xFFFFFFFF
            }
        } else {
            0xFFFFFFFF
        };

        // Create output tensor
        let output_name = node.outputs.get(0).context("Conv missing output")?;
        let output_id = self.get_or_create_tensor(output_name);

        // Parse attributes
        let strides = node.get_ints("strides").map(|v| v.as_slice()).unwrap_or(&[1, 1]);
        let pads = node.get_ints("pads").map(|v| v.as_slice()).unwrap_or(&[0, 0, 0, 0]);
        let dilations = node.get_ints("dilations").map(|v| v.as_slice()).unwrap_or(&[1, 1]);
        let group = node.get_int("group").unwrap_or(1) as u32;

        // Calculate output shape: out_h = (in_h + pad_top + pad_bottom - dilation*(kh-1) - 1) / stride + 1
        let input_shape = self.get_tensor_shape(input_id);
        let sh = strides.get(0).copied().unwrap_or(1) as i32;
        let sw = strides.get(1).copied().unwrap_or(1) as i32;
        let dh = dilations.get(0).copied().unwrap_or(1) as i32;
        let dw = dilations.get(1).copied().unwrap_or(1) as i32;
        let pad_t = pads.get(0).copied().unwrap_or(0) as i32;
        let pad_l = pads.get(1).copied().unwrap_or(0) as i32;
        let pad_b = pads.get(2).copied().unwrap_or(0) as i32;
        let pad_r = pads.get(3).copied().unwrap_or(0) as i32;

        // Get spatial dimensions based on data format
        let (in_h, in_w) = if self.use_nhwc {
            // NHWC: shape = [N, H, W, C]
            (input_shape[1], input_shape[2])
        } else {
            // NCHW: shape = [N, C, H, W]
            (input_shape[2], input_shape[3])
        };
        let out_h = (in_h + pad_t + pad_b - dh * (kh as i32 - 1) - 1) / sh + 1;
        let out_w = (in_w + pad_l + pad_r - dw * (kw as i32 - 1) - 1) / sw + 1;

        // Output shape based on format
        if self.use_nhwc {
            // NHWC: [N, H, W, C]
            self.update_tensor_shape(output_id, &[input_shape[0], out_h, out_w, out_ch as i32]);
        } else {
            // NCHW: [N, C, H, W]
            self.update_tensor_shape(output_id, &[input_shape[0], out_ch as i32, out_h, out_w]);
        }

        // Propagate quantization scales for INT8
        // The combined_scale in runtime = (in_scale * w_scale) / out_scale
        if self.quantize {
            // Try to get QDQ calibrated output scale first
            if let Some(out_scale) = self.get_qdq_scale(output_name) {
                // Use calibrated scale from QDQ model
                self.set_tensor_scale(output_id, out_scale);
                if self.verbose && layer_id < 5 {
                    let in_scale = self.get_tensor_scale(input_id);
                    let combined = (in_scale * scale) / out_scale;
                    println!("  Conv[{}] QDQ scales: in={:.6} w={:.6} out={:.6} -> combined={:.6}",
                             layer_id, in_scale, scale, out_scale, combined);
                }
            } else {
                // Fallback: conservative heuristic to prevent overflow
                // combined_scale = 1 / fan_in
                let in_scale = self.get_tensor_scale(input_id);
                let fan_in = (in_ch * kh * kw) as f32;
                let out_scale = in_scale * scale * fan_in;
                self.set_tensor_scale(output_id, out_scale);
                if self.verbose && layer_id < 5 {
                    println!("  Conv[{}] heuristic scales: in={:.6} w={:.6} out={:.6} (fan_in={})",
                             layer_id, in_scale, scale, out_scale, fan_in);
                }
            }
        }

        // Determine layer type (depthwise vs regular conv)
        let layer_type = if group > 1 && group == in_ch && group == out_ch {
            LayerType::DepthwiseConv2d
        } else {
            LayerType::Conv2d
        };

        // Create layer
        let mut layer = MarsLayer::new(layer_id, layer_type);
        layer.num_inputs = 1;
        layer.num_outputs = 1;
        layer.input_tensor_ids[0] = input_id;
        layer.output_tensor_ids[0] = output_id;

        layer.params = LayerParams::Conv(ConvParams {
            kernel_h: kh,
            kernel_w: kw,
            stride_h: strides.get(0).copied().unwrap_or(1) as u32,
            stride_w: strides.get(1).copied().unwrap_or(1) as u32,
            dilation_h: dilations.get(0).copied().unwrap_or(1) as u32,
            dilation_w: dilations.get(1).copied().unwrap_or(1) as u32,
            padding: if pads.iter().all(|&p| p == 0) { Padding::Valid } else { Padding::Explicit },
            pad_top: pads.get(0).copied().unwrap_or(0) as u32,
            pad_left: pads.get(1).copied().unwrap_or(0) as u32,
            pad_bottom: pads.get(2).copied().unwrap_or(0) as u32,
            pad_right: pads.get(3).copied().unwrap_or(0) as u32,
            groups: group,
            activation: Activation::None,
            weight_tensor_id: weight_id,
            bias_tensor_id: bias_id,
        });

        self.layers.push(layer);

        if self.verbose {
            println!("Conv {}: in_ch={} out_ch={} k={}x{} s={:?}",
                layer_id, in_ch, out_ch, kh, kw, strides);
        }

        Ok(())
    }

    fn process_pool(&mut self, node: &OnnxNode, layer_type: LayerType) -> Result<()> {
        let layer_id = self.layers.len() as u32;

        let input_name = node.inputs.get(0).context("Pool missing input")?;
        let input_id = self.get_or_create_tensor(input_name);

        let output_name = node.outputs.get(0).context("Pool missing output")?;
        let output_id = self.get_or_create_tensor(output_name);

        let kernel = node.get_ints("kernel_shape").map(|v| v.as_slice()).unwrap_or(&[2, 2]);
        let strides = node.get_ints("strides").map(|v| v.as_slice()).unwrap_or(&[2, 2]);
        let pads = node.get_ints("pads").map(|v| v.as_slice()).unwrap_or(&[0, 0, 0, 0]);

        // Calculate output shape
        let input_shape = self.get_tensor_shape(input_id);
        let kh = kernel.get(0).copied().unwrap_or(2) as i32;
        let kw = kernel.get(1).copied().unwrap_or(2) as i32;
        let sh = strides.get(0).copied().unwrap_or(2) as i32;
        let sw = strides.get(1).copied().unwrap_or(2) as i32;
        let pad_t = pads.get(0).copied().unwrap_or(0) as i32;
        let pad_b = pads.get(2).copied().unwrap_or(0) as i32;
        let pad_l = pads.get(1).copied().unwrap_or(0) as i32;
        let pad_r = pads.get(3).copied().unwrap_or(0) as i32;

        let out_h = (input_shape[2] + pad_t + pad_b - kh) / sh + 1;
        let out_w = (input_shape[3] + pad_l + pad_r - kw) / sw + 1;
        self.update_tensor_shape(output_id, &[input_shape[0], input_shape[1], out_h, out_w]);

        // Pool preserves scale (max/avg don't change value range significantly)
        if self.quantize {
            let in_scale = self.get_tensor_scale(input_id);
            self.set_tensor_scale(output_id, in_scale);
        }

        let mut layer = MarsLayer::new(layer_id, layer_type);
        layer.num_inputs = 1;
        layer.num_outputs = 1;
        layer.input_tensor_ids[0] = input_id;
        layer.output_tensor_ids[0] = output_id;

        layer.params = LayerParams::Pool(PoolParams {
            kernel_h: kh as u32,
            kernel_w: kw as u32,
            stride_h: sh as u32,
            stride_w: sw as u32,
            padding: if pads.iter().all(|&p| p == 0) { Padding::Valid } else { Padding::Explicit },
            pad_top: pad_t as u32,
            pad_bottom: pad_b as u32,
            pad_left: pad_l as u32,
            pad_right: pad_r as u32,
        });

        self.layers.push(layer);
        Ok(())
    }

    fn process_activation(&mut self, node: &OnnxNode, layer_type: LayerType) -> Result<()> {
        let layer_id = self.layers.len() as u32;

        let input_name = node.inputs.get(0).context("Activation missing input")?;
        let input_id = self.get_or_create_tensor(input_name);

        let output_name = node.outputs.get(0).context("Activation missing output")?;
        let output_id = self.get_or_create_tensor(output_name);

        // Activation layers keep the same shape
        let input_shape = self.get_tensor_shape(input_id);
        self.update_tensor_shape(output_id, &input_shape);

        // Propagate scale for activations
        // Sigmoid/Softmax: output is [0,1], so scale = 1/127 maps to full range
        // ReLU: output range <= input range (same scale)
        // SiLU: output range <= input range (same scale)
        if self.quantize {
            let in_scale = self.get_tensor_scale(input_id);
            let out_scale = match layer_type {
                LayerType::Sigmoid => 1.0 / 127.0,  // [0, 1] range
                _ => in_scale,  // ReLU, SiLU, etc. preserve range
            };
            self.set_tensor_scale(output_id, out_scale);
        }

        let mut layer = MarsLayer::new(layer_id, layer_type);
        layer.num_inputs = 1;
        layer.num_outputs = 1;
        layer.input_tensor_ids[0] = input_id;
        layer.output_tensor_ids[0] = output_id;

        self.layers.push(layer);
        Ok(())
    }

    /// Process BatchNormalization: y = (x - mean) / sqrt(var + eps) * gamma + beta
    fn process_batchnorm(&mut self, node: &OnnxNode) -> Result<()> {
        let layer_id = self.layers.len() as u32;

        let input_name = node.inputs.get(0).context("BatchNorm missing input")?;
        let input_id = self.get_or_create_tensor(input_name);

        let output_name = node.outputs.get(0).context("BatchNorm missing output")?;
        let output_id = self.get_or_create_tensor(output_name);

        // BatchNorm keeps the same shape
        let input_shape = self.get_tensor_shape(input_id);
        self.update_tensor_shape(output_id, &input_shape);

        // Get number of channels from input shape (NCHW format: shape[1])
        let num_channels = if input_shape.len() >= 2 { input_shape[1] as usize } else { 1 };

        // Get BatchNorm parameters: scale (gamma), B (beta), mean, var
        let scale_name = node.inputs.get(1);
        let bias_name = node.inputs.get(2);
        let mean_name = node.inputs.get(3);
        let var_name = node.inputs.get(4);

        // Get epsilon attribute
        let epsilon = node.get_float("epsilon").unwrap_or(1e-5);

        // Fuse BN params into scale and bias: y = x * (gamma / sqrt(var + eps)) + (beta - mean * gamma / sqrt(var + eps))
        let mut fused_scale = vec![1.0f32; num_channels];
        let mut fused_bias = vec![0.0f32; num_channels];

        // Get gamma (scale)
        let gamma: Vec<f32> = if let Some(name) = scale_name {
            if let Some(tensor) = self.onnx.initializers.get(name) {
                Self::bytes_to_f32(&tensor.data)
            } else {
                vec![1.0f32; num_channels]
            }
        } else {
            vec![1.0f32; num_channels]
        };

        // Get beta (bias)
        let beta: Vec<f32> = if let Some(name) = bias_name {
            if let Some(tensor) = self.onnx.initializers.get(name) {
                Self::bytes_to_f32(&tensor.data)
            } else {
                vec![0.0f32; num_channels]
            }
        } else {
            vec![0.0f32; num_channels]
        };

        // Get mean
        let mean: Vec<f32> = if let Some(name) = mean_name {
            if let Some(tensor) = self.onnx.initializers.get(name) {
                Self::bytes_to_f32(&tensor.data)
            } else {
                vec![0.0f32; num_channels]
            }
        } else {
            vec![0.0f32; num_channels]
        };

        // Get var
        let var: Vec<f32> = if let Some(name) = var_name {
            if let Some(tensor) = self.onnx.initializers.get(name) {
                Self::bytes_to_f32(&tensor.data)
            } else {
                vec![1.0f32; num_channels]
            }
        } else {
            vec![1.0f32; num_channels]
        };

        // Compute fused scale and bias
        for i in 0..num_channels.min(gamma.len()).min(var.len()) {
            let inv_std = 1.0 / (var[i] + epsilon).sqrt();
            fused_scale[i] = gamma[i] * inv_std;
            fused_bias[i] = beta.get(i).copied().unwrap_or(0.0)
                          - mean.get(i).copied().unwrap_or(0.0) * fused_scale[i];
        }

        // Store fused scale as weight tensor
        let scale_bytes: Vec<u8> = fused_scale.iter()
            .flat_map(|&f| f.to_le_bytes())
            .collect();
        let (scale_offset, scale_size) = self.add_weights(&scale_bytes);

        let scale_tensor_id = self.tensors.len() as u32;
        let mut scale_tensor = MarsTensor::new(scale_tensor_id, &format!("{}_scale", node.name));
        scale_tensor.dtype = DataType::Float32;
        scale_tensor.ndims = 1;
        scale_tensor.shape[0] = num_channels as i32;
        scale_tensor.data_offset = scale_offset;
        scale_tensor.data_size = scale_size;
        self.tensors.push(scale_tensor);

        // Store fused bias as weight tensor
        let bias_bytes: Vec<u8> = fused_bias.iter()
            .flat_map(|&f| f.to_le_bytes())
            .collect();
        let (bias_offset, bias_size) = self.add_weights(&bias_bytes);

        let bias_tensor_id = self.tensors.len() as u32;
        let mut bias_tensor = MarsTensor::new(bias_tensor_id, &format!("{}_bias", node.name));
        bias_tensor.dtype = DataType::Float32;
        bias_tensor.ndims = 1;
        bias_tensor.shape[0] = num_channels as i32;
        bias_tensor.data_offset = bias_offset;
        bias_tensor.data_size = bias_size;
        self.tensors.push(bias_tensor);

        // BatchNorm can change scale based on fused_scale values
        // Use max fused_scale to estimate output range change
        if self.quantize {
            let in_scale = self.get_tensor_scale(input_id);
            let max_fused = fused_scale.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
            self.set_tensor_scale(output_id, in_scale * max_fused.max(0.1));
        }

        let mut layer = MarsLayer::new(layer_id, LayerType::BatchNorm);
        layer.num_inputs = 3;  // input, scale, bias
        layer.num_outputs = 1;
        layer.input_tensor_ids[0] = input_id;
        layer.input_tensor_ids[1] = scale_tensor_id;
        layer.input_tensor_ids[2] = bias_tensor_id;
        layer.output_tensor_ids[0] = output_id;

        self.layers.push(layer);
        Ok(())
    }

    fn process_elementwise(&mut self, node: &OnnxNode, layer_type: LayerType) -> Result<()> {
        let layer_id = self.layers.len() as u32;

        // Elementwise ops can have 2 inputs
        let input_a = node.inputs.get(0).context("Elementwise missing input A")?;
        let input_b = node.inputs.get(1).context("Elementwise missing input B")?;

        let input_a_id = self.get_or_create_tensor(input_a);
        let input_b_id = self.get_or_create_tensor(input_b);

        let output_name = node.outputs.get(0).context("Elementwise missing output")?;
        let output_id = self.get_or_create_tensor(output_name);

        // Elementwise ops preserve shape (use first input's shape)
        let input_shape = self.get_tensor_shape(input_a_id);
        self.update_tensor_shape(output_id, &input_shape);

        // Propagate scale for elementwise ops
        if self.quantize {
            let scale_a = self.get_tensor_scale(input_a_id);
            let scale_b = self.get_tensor_scale(input_b_id);
            let out_scale = match layer_type {
                LayerType::Add => scale_a.max(scale_b),  // Use max for add
                LayerType::Mul => {
                    // For Mul: if one input has default scale (1.0), use the other
                    // This handles constant multipliers (strides, grid offsets, etc.)
                    if (scale_a - 1.0).abs() < 0.001 { scale_b }
                    else if (scale_b - 1.0).abs() < 0.001 { scale_a }
                    else { scale_a.min(scale_b) }  // Both non-default: use smaller
                },
                _ => scale_a,
            };
            self.set_tensor_scale(output_id, out_scale);
        }

        let mut layer = MarsLayer::new(layer_id, layer_type);
        layer.num_inputs = 2;
        layer.num_outputs = 1;
        layer.input_tensor_ids[0] = input_a_id;
        layer.input_tensor_ids[1] = input_b_id;
        layer.output_tensor_ids[0] = output_id;

        self.layers.push(layer);
        Ok(())
    }

    fn process_concat(&mut self, node: &OnnxNode) -> Result<()> {
        let layer_id = self.layers.len() as u32;

        // Handle negative axis (common in ONNX: -1 means last dim)
        let raw_axis = node.get_int("axis").unwrap_or(1);
        let mut axis = if raw_axis < 0 { (4 + raw_axis) as u32 } else { raw_axis as u32 };
        axis = axis.min(3);  // Clamp to valid range

        // Adjust axis for NHWC format: NCHW axis 1 (C) -> NHWC axis 3
        // NCHW [N,C,H,W] vs NHWC [N,H,W,C]
        if self.use_nhwc && axis > 0 {
            axis = match axis {
                1 => 3,  // C dimension: NCHW[1] -> NHWC[3]
                2 => 1,  // H dimension: NCHW[2] -> NHWC[1]
                3 => 2,  // W dimension: NCHW[3] -> NHWC[2]
                _ => axis,
            };
        }

        let mut layer = MarsLayer::new(layer_id, LayerType::Concat);
        layer.num_inputs = node.inputs.len().min(4) as u32;
        layer.num_outputs = 1;

        // Get input shapes for concat calculation
        let mut total_axis_size = 0i32;
        let mut base_shape = [0i32; 4];

        for (i, input_name) in node.inputs.iter().take(4).enumerate() {
            let tid = self.get_or_create_tensor(input_name);
            layer.input_tensor_ids[i] = tid;
            let shape = self.get_tensor_shape(tid);
            if i == 0 {
                base_shape = shape;
            }
            if (axis as usize) < 4 {
                total_axis_size += shape[axis as usize];
            }
        }

        let output_name = node.outputs.get(0).context("Concat missing output")?;
        let output_id = self.get_or_create_tensor(output_name);
        layer.output_tensor_ids[0] = output_id;

        // Output shape: same as first input, but axis dimension is sum of all inputs
        let mut out_shape = base_shape;
        if (axis as usize) < 4 {
            out_shape[axis as usize] = total_axis_size;
        }
        self.update_tensor_shape(output_id, &out_shape);

        // Propagate scale - use max of input scales (preserve dynamic range)
        // But only if the output tensor doesn't already have a QDQ-calibrated scale
        if self.quantize {
            let existing_scale = self.get_tensor_scale(output_id);
            // Only override if the existing scale is the default 1.0
            if (existing_scale - 1.0).abs() < 0.0001 {
                let mut max_scale = 0.0f32;
                for i in 0..layer.num_inputs as usize {
                    max_scale = max_scale.max(self.get_tensor_scale(layer.input_tensor_ids[i]));
                }
                if max_scale > 0.0001 {
                    self.set_tensor_scale(output_id, max_scale);
                }
            }
        }

        layer.params = LayerParams::Concat(ConcatParams {
            axis,
            num_inputs: layer.num_inputs,
        });

        self.layers.push(layer);
        Ok(())
    }

    fn process_upsample(&mut self, node: &OnnxNode) -> Result<()> {
        let layer_id = self.layers.len() as u32;

        let input_name = node.inputs.get(0).context("Upsample missing input")?;
        let input_id = self.get_or_create_tensor(input_name);

        let output_name = node.outputs.get(0).context("Upsample missing output")?;
        let output_id = self.get_or_create_tensor(output_name);

        // Get scale factors from scales input or sizes input
        let (scale_h, scale_w) = if let Some(scales_name) = node.inputs.get(2) {
            if let Some(scales) = self.onnx.initializers.get(scales_name) {
                let floats: Vec<f32> = scales.data.chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                (floats.get(2).copied().unwrap_or(2.0) as u32,
                 floats.get(3).copied().unwrap_or(2.0) as u32)
            } else {
                (2, 2)
            }
        } else {
            (2, 2)
        };

        // Calculate output shape based on format
        let input_shape = self.get_tensor_shape(input_id);
        if self.use_nhwc {
            // NHWC: [N, H, W, C]
            let out_h = input_shape[1] * scale_h as i32;
            let out_w = input_shape[2] * scale_w as i32;
            self.update_tensor_shape(output_id, &[input_shape[0], out_h, out_w, input_shape[3]]);
        } else {
            // NCHW: [N, C, H, W]
            let out_h = input_shape[2] * scale_h as i32;
            let out_w = input_shape[3] * scale_w as i32;
            self.update_tensor_shape(output_id, &[input_shape[0], input_shape[1], out_h, out_w]);
        }

        let mode = node.get_string("mode").unwrap_or("nearest");
        let mode_val = if mode == "bilinear" || mode == "linear" { 1 } else { 0 };

        // Propagate scale - upsample doesn't change value range
        if self.quantize {
            let in_scale = self.get_tensor_scale(input_id);
            self.set_tensor_scale(output_id, in_scale);
        }

        let mut layer = MarsLayer::new(layer_id, LayerType::Upsample);
        layer.num_inputs = 1;
        layer.num_outputs = 1;
        layer.input_tensor_ids[0] = input_id;
        layer.output_tensor_ids[0] = output_id;

        layer.params = LayerParams::Upsample(UpsampleParams {
            scale_h,
            scale_w,
            mode: mode_val,
        });

        self.layers.push(layer);
        Ok(())
    }

    fn process_reshape(&mut self, node: &OnnxNode) -> Result<()> {
        let layer_id = self.layers.len() as u32;

        let input_name = node.inputs.get(0).context("Reshape missing input")?;
        let input_id = self.get_or_create_tensor(input_name);

        let output_name = node.outputs.get(0).context("Reshape missing output")?;
        let output_id = self.get_or_create_tensor(output_name);

        // Get target shape from the second input (constant)
        let mut target_shape = [0i32; 6];
        let mut ndims = 4u32;
        if let Some(shape_name) = node.inputs.get(1) {
            if let Some(shape_data) = self.onnx.initializers.get(shape_name) {
                // Shape is stored as int64
                let dims: Vec<i64> = shape_data.data.chunks_exact(8)
                    .map(|b| i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
                    .collect();
                ndims = dims.len().min(6) as u32;
                for (i, &d) in dims.iter().take(6).enumerate() {
                    target_shape[i] = d as i32;
                }
            }
        }

        // Update output tensor shape
        let mut out_shape = [1i32; 4];
        for i in 0..4.min(ndims as usize) {
            out_shape[i] = target_shape[i];
        }
        self.update_tensor_shape(output_id, &out_shape);

        // Reshape preserves scale
        if self.quantize {
            let in_scale = self.get_tensor_scale(input_id);
            self.set_tensor_scale(output_id, in_scale);
        }

        let mut layer = MarsLayer::new(layer_id, LayerType::Reshape);
        layer.num_inputs = 1;
        layer.num_outputs = 1;
        layer.input_tensor_ids[0] = input_id;
        layer.output_tensor_ids[0] = output_id;

        // Store target shape in params (use generic params)
        layer.params = LayerParams::Reshape(ReshapeParams {
            target_shape,
            ndims,
        });

        self.layers.push(layer);
        Ok(())
    }

    fn process_transpose(&mut self, node: &OnnxNode) -> Result<()> {
        let layer_id = self.layers.len() as u32;

        let input_name = node.inputs.get(0).context("Transpose missing input")?;
        let input_id = self.get_or_create_tensor(input_name);

        let output_name = node.outputs.get(0).context("Transpose missing output")?;
        let output_id = self.get_or_create_tensor(output_name);

        // Get perm from attributes
        let default_perm: Vec<i64> = vec![0, 1, 2, 3];
        let perm = node.get_ints("perm").unwrap_or(&default_perm);
        let mut perm_arr = [0u32; 6];
        for (i, &p) in perm.iter().take(6).enumerate() {
            perm_arr[i] = p as u32;
        }

        // Calculate output shape based on perm
        let input_shape = self.get_tensor_shape(input_id);
        let mut out_shape = [1i32; 4];
        for i in 0..4.min(perm.len()) {
            let src_idx = perm[i] as usize;
            if src_idx < 4 {
                out_shape[i] = input_shape[src_idx];
            }
        }
        self.update_tensor_shape(output_id, &out_shape);

        // Transpose preserves scale
        if self.quantize {
            let in_scale = self.get_tensor_scale(input_id);
            self.set_tensor_scale(output_id, in_scale);
        }

        let mut layer = MarsLayer::new(layer_id, LayerType::Transpose);
        layer.num_inputs = 1;
        layer.num_outputs = 1;
        layer.input_tensor_ids[0] = input_id;
        layer.output_tensor_ids[0] = output_id;

        layer.params = LayerParams::Transpose(TransposeParams {
            perm: perm_arr,
            ndims: perm.len() as u32,
        });

        self.layers.push(layer);
        Ok(())
    }

    fn process_softmax(&mut self, node: &OnnxNode) -> Result<()> {
        let layer_id = self.layers.len() as u32;

        let input_name = node.inputs.get(0).context("Softmax missing input")?;
        let input_id = self.get_or_create_tensor(input_name);

        let output_name = node.outputs.get(0).context("Softmax missing output")?;
        let output_id = self.get_or_create_tensor(output_name);

        // Softmax preserves shape
        let input_shape = self.get_tensor_shape(input_id);
        self.update_tensor_shape(output_id, &input_shape);

        // Softmax output is [0, 1], so scale = 1/127
        if self.quantize {
            self.set_tensor_scale(output_id, 1.0 / 127.0);
        }

        let axis = node.get_int("axis").unwrap_or(-1);
        let axis = if axis < 0 { (4 + axis) as u32 } else { axis as u32 };

        let mut layer = MarsLayer::new(layer_id, LayerType::Softmax);
        layer.num_inputs = 1;
        layer.num_outputs = 1;
        layer.input_tensor_ids[0] = input_id;
        layer.output_tensor_ids[0] = output_id;

        layer.params = LayerParams::Softmax(SoftmaxParams { axis });

        self.layers.push(layer);
        Ok(())
    }

    /// Write the compiled model to a .mars file
    fn write<W: Write>(&self, w: &mut W) -> Result<()> {
        // Calculate offsets
        let tensors_offset = HEADER_SIZE;
        let layers_offset = tensors_offset + self.tensors.len() * TENSOR_SIZE;
        let weights_offset = layers_offset + self.layers.len() * LAYER_SIZE;

        // Write header
        let mut header = MarsHeader::new();
        header.num_layers = self.layers.len() as u32;
        header.num_tensors = self.tensors.len() as u32;
        header.num_inputs = self.onnx.inputs.len() as u32;
        header.num_outputs = self.onnx.outputs.len() as u32;
        header.weights_offset = weights_offset as u64;
        header.weights_size = self.weights_data.len() as u64;

        // Set input/output tensor IDs
        for (i, input) in self.onnx.inputs.iter().take(4).enumerate() {
            if let Some(&id) = self.tensor_map.get(&input.name) {
                header.input_tensor_ids[i] = id;
            }
        }
        for (i, output) in self.onnx.outputs.iter().take(4).enumerate() {
            if let Some(&id) = self.tensor_map.get(&output.name) {
                header.output_tensor_ids[i] = id;
                if self.verbose {
                    println!("Output {}: {} -> tensor_id {}", i, output.name, id);
                }
            } else {
                // Try to find the tensor with a suffix (QDQ models add _QuantizeLinear_Input)
                let alt_name = format!("{}_QuantizeLinear_Input", output.name);
                if let Some(&id) = self.tensor_map.get(&alt_name) {
                    header.output_tensor_ids[i] = id;
                    if self.verbose {
                        let scale = self.tensors.get(id as usize).map(|t| t.scale).unwrap_or(1.0);
                        println!("Output {}: {} (via {}) -> tensor_id {} (scale={})",
                                 i, output.name, alt_name, id, scale);
                    }
                } else {
                    eprintln!("Warning: Output tensor {} not found in tensor_map", output.name);
                }
            }
        }

        header.write(w)?;

        // Write tensors
        for tensor in &self.tensors {
            tensor.write(w)?;
        }

        // Write layers
        for layer in &self.layers {
            layer.write(w)?;
        }

        // Write weights
        w.write_all(&self.weights_data)?;

        Ok(())
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("Mars Compiler v{}", env!("CARGO_PKG_VERSION"));
    println!("Input:  {}", args.input.display());
    println!("Output: {}", args.output.display());
    println!();

    // Load ONNX model
    println!("Loading ONNX model...");
    let onnx = OnnxModel::load(&args.input)
        .context("Failed to load ONNX model")?;

    // Compile to Mars
    // If --float32 is specified, don't quantize (keep float32 weights)
    let quantize = !args.float32;
    let use_nhwc = args.nhwc;
    println!("Quantization: {}", if quantize { "INT8" } else { "FLOAT32 (no quantization)" });
    println!("Feature format: {}", if use_nhwc { "NHWC (channels-last)" } else { "NCHW (channels-first)" });
    let mut compiler = MarsCompiler::new(onnx, quantize, use_nhwc, args.verbose);
    compiler.compile()?;

    // Write output
    println!("\nWriting Mars model...");
    let file = File::create(&args.output)
        .context("Failed to create output file")?;
    let mut writer = BufWriter::new(file);
    compiler.write(&mut writer)?;
    writer.flush()?;

    // Summary
    let file_size = std::fs::metadata(&args.output)?.len();
    println!("\nCompilation complete!");
    println!("  Layers:      {}", compiler.layers.len());
    println!("  Tensors:     {}", compiler.tensors.len());
    println!("  Weights:     {} bytes", compiler.weights_data.len());
    println!("  Output size: {} bytes ({:.2} KB)", file_size, file_size as f64 / 1024.0);

    Ok(())
}
