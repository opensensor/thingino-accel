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
use onnx_parser::{OnnxModel, OnnxNode};

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
    
    /// Quantize weights to int8
    #[arg(short, long, default_value = "true")]
    quantize: bool,
    
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
        "Sigmoid" => Some(LayerType::Sigmoid),
        "Mul" => Some(LayerType::Mul),
        "Add" => Some(LayerType::Add),
        "Concat" => Some(LayerType::Concat),
        "Resize" | "Upsample" => Some(LayerType::Upsample),
        "Reshape" => Some(LayerType::Reshape),
        "Transpose" => Some(LayerType::Transpose),
        "Softmax" => Some(LayerType::Softmax),
        // Skip these ops - they get folded or are handled differently
        "Constant" | "Shape" | "Gather" | "Slice" | "Split" | "Sub" | "Div" => None,
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
    quantize: bool,
    verbose: bool,
}

impl MarsCompiler {
    fn new(onnx: OnnxModel, quantize: bool, verbose: bool) -> Self {
        Self {
            onnx,
            tensors: Vec::new(),
            layers: Vec::new(),
            weights_data: Vec::new(),
            tensor_map: HashMap::new(),
            quantize,
            verbose,
        }
    }
    
    /// Compile the ONNX model to Mars format
    fn compile(&mut self) -> Result<()> {
        if self.verbose {
            self.onnx.print_summary();
            println!("\nCompiling to Mars format...\n");
        }
        
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
        
        // Step 3: Mark output tensors
        self.mark_outputs()?;
        
        Ok(())
    }
    
    fn create_input_tensors(&mut self) -> Result<()> {
        for input in &self.onnx.inputs {
            let id = self.tensors.len() as u32;
            
            let mut tensor = MarsTensor::new(id, &input.name);
            tensor.ndims = input.dims.len() as u32;
            for (i, &dim) in input.dims.iter().enumerate() {
                if i < MARS_MAX_DIMS {
                    tensor.shape[i] = dim.max(1) as i32;
                }
            }
            // Input is NCHW from ONNX, convert to NHWC for NNA
            tensor.format = DataFormat::Nhwc;
            tensor.dtype = if self.quantize { DataType::Int8 } else { DataType::Float32 };
            
            self.tensor_map.insert(input.name.clone(), id);
            self.tensors.push(tensor);
            
            if self.verbose {
                println!("Input tensor {}: {} {:?}", id, input.name, input.dims);
            }
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
            LayerType::Relu | LayerType::Sigmoid => self.process_activation(node, layer_type)?,
            LayerType::Add | LayerType::Mul => self.process_elementwise(node, layer_type)?,
            LayerType::Concat => self.process_concat(node)?,
            LayerType::Upsample => self.process_upsample(node)?,
            _ => {
                if self.verbose {
                    println!("Skipping layer type: {:?}", layer_type);
                }
            }
        }

        Ok(())
    }

    /// Create or get a feature tensor (uses NDHWC32 format for NNA)
    fn get_or_create_tensor(&mut self, name: &str) -> u32 {
        if let Some(&id) = self.tensor_map.get(name) {
            return id;
        }

        let id = self.tensors.len() as u32;
        let mut tensor = MarsTensor::new(id, name);
        tensor.dtype = if self.quantize { DataType::Int8 } else { DataType::Float32 };
        // Use NNA native format for features: NDHWC32 (32-channel groups)
        tensor.format = DataFormat::Ndhwc32;

        // Try to get shape from ONNX shape_info
        if let Some(dims) = self.onnx.shape_info.get(name) {
            tensor.ndims = dims.len() as u32;
            for (i, &dim) in dims.iter().enumerate() {
                if i < MARS_MAX_DIMS {
                    tensor.shape[i] = dim.max(1) as i32;
                }
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

    fn quantize_weights(&self, float_data: &[u8]) -> (Vec<u8>, f32) {
        if !self.quantize {
            return (float_data.to_vec(), 1.0);
        }

        // Convert bytes to f32
        let floats: Vec<f32> = float_data.chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        // Find scale
        let max_abs = floats.iter().map(|f| f.abs()).fold(0.0f32, f32::max);
        let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };

        // Quantize to int8
        let quantized: Vec<u8> = floats.iter()
            .map(|&f| ((f / scale).round().clamp(-127.0, 127.0) as i8) as u8)
            .collect();

        (quantized, scale)
    }

    fn process_conv(&mut self, node: &OnnxNode) -> Result<()> {
        let layer_id = self.layers.len() as u32;

        // Get input tensor
        let input_name = node.inputs.get(0).context("Conv missing input")?;
        let input_id = self.get_or_create_tensor(input_name);

        // Get weight tensor from initializers
        let weight_name = node.inputs.get(1).context("Conv missing weight")?;
        let weight_tensor = self.onnx.initializers.get(weight_name)
            .context("Conv weight not found in initializers")?;

        // Get kernel shape from weights [O, I, H, W]
        let out_ch = weight_tensor.dims.get(0).copied().unwrap_or(1) as u32;
        let in_ch = weight_tensor.dims.get(1).copied().unwrap_or(1) as u32;
        let kh = weight_tensor.dims.get(2).copied().unwrap_or(3) as u32;
        let kw = weight_tensor.dims.get(3).copied().unwrap_or(3) as u32;

        // Quantize and pack weights in NMHWSOIB2 format (NNA native)
        let (quant_weights, scale) = self.quantize_weights(&weight_tensor.data);

        // Convert quantized weights from OIHW to NMHWSOIB2 packed format
        let int8_weights: Vec<i8> = quant_weights.iter().map(|&b| b as i8).collect();
        let packed_weights = pack_weights_nmhwsoib2(
            &int8_weights,
            out_ch as usize,
            in_ch as usize,
            kh as usize,
            kw as usize,
        );
        let (weight_offset, weight_size) = self.add_weights(&packed_weights);

        // Create weight tensor with NMHWSOIB2 format
        let weight_id = self.tensors.len() as u32;
        let mut w_tensor = MarsTensor::new(weight_id, weight_name);
        w_tensor.dtype = if self.quantize { DataType::Int8 } else { DataType::Float32 };
        w_tensor.format = DataFormat::Nmhwsoib2;  // NNA native packed format
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
                let (quant_bias, _) = self.quantize_weights(&bias_tensor.data);
                let (bias_offset, bias_size) = self.add_weights(&quant_bias);

                let bid = self.tensors.len() as u32;
                let mut b_tensor = MarsTensor::new(bid, bias_name);
                b_tensor.dtype = if self.quantize { DataType::Int8 } else { DataType::Float32 };
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

        // Input is NCHW in ONNX, shape[1] = C, shape[2] = H, shape[3] = W
        let in_h = input_shape[2];
        let in_w = input_shape[3];
        let out_h = (in_h + pad_t + pad_b - dh * (kh as i32 - 1) - 1) / sh + 1;
        let out_w = (in_w + pad_l + pad_r - dw * (kw as i32 - 1) - 1) / sw + 1;

        // Output shape: [N, out_ch, out_h, out_w] in NCHW
        self.update_tensor_shape(output_id, &[input_shape[0], out_ch as i32, out_h, out_w]);

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

        let mut layer = MarsLayer::new(layer_id, layer_type);
        layer.num_inputs = 1;
        layer.num_outputs = 1;
        layer.input_tensor_ids[0] = input_id;
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

        let axis = node.get_int("axis").unwrap_or(1) as u32;

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
            total_axis_size += shape[axis as usize];
        }

        let output_name = node.outputs.get(0).context("Concat missing output")?;
        let output_id = self.get_or_create_tensor(output_name);
        layer.output_tensor_ids[0] = output_id;

        // Output shape: same as first input, but axis dimension is sum of all inputs
        let mut out_shape = base_shape;
        out_shape[axis as usize] = total_axis_size;
        self.update_tensor_shape(output_id, &out_shape);

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

        // Calculate output shape
        let input_shape = self.get_tensor_shape(input_id);
        let out_h = input_shape[2] * scale_h as i32;
        let out_w = input_shape[3] * scale_w as i32;
        self.update_tensor_shape(output_id, &[input_shape[0], input_shape[1], out_h, out_w]);

        let mode = node.get_string("mode").unwrap_or("nearest");
        let mode_val = if mode == "bilinear" || mode == "linear" { 1 } else { 0 };

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
    let mut compiler = MarsCompiler::new(onnx, args.quantize, args.verbose);
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
