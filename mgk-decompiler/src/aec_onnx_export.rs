//! AEC-Specific ONNX Export Module
//!
//! Exports the AEC MGK model to a runnable ONNX format with proper
//! reshape operations and embedded weights.

use anyhow::{Result, Context};
use prost::Message;
use std::fs::File;
use std::io::Write;
use std::path::Path;

use crate::onnx_export::{
    AttributeProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto,
    StringStringEntryProto, TensorDataType, TensorProto, make_value_info,
};

/// ONNX IR version
const ONNX_IR_VERSION: i64 = 9;
/// ONNX opset version  
const ONNX_OPSET_VERSION: i64 = 13;

/// AEC model parameters derived from log.txt analysis
#[derive(Clone, Debug)]
pub struct AecModelParams {
    pub n_freq: i64,        // 256 frequency bins
    pub n_frames: i64,      // 8 input frames
    pub n_channels: i64,    // 32 internal channels
    pub hidden_size: i64,   // 32 GRU hidden size
}

impl Default for AecModelParams {
    fn default() -> Self {
        Self {
            n_freq: 256,
            n_frames: 8,
            n_channels: 32,
            hidden_size: 32,
        }
    }
}

/// Quantization scales for weight dequantization
#[derive(Clone, Debug)]
pub struct LayerScale {
    pub name: String,
    pub scale: f32,
}

/// Extract quantization scales from metadata JSON
pub fn extract_scales_from_metadata(metadata_path: &Path) -> Result<Vec<f32>> {
    let content = std::fs::read_to_string(metadata_path)
        .context("Failed to read metadata.json")?;
    
    let json: serde_json::Value = serde_json::from_str(&content)
        .context("Failed to parse metadata.json")?;
    
    let mut scales = Vec::new();
    if let Some(arr) = json.get("quantization_scales").and_then(|v| v.as_array()) {
        for entry in arr {
            if let Some(val) = entry.get("value").and_then(|v| v.as_f64()) {
                let scale = val as f32;
                // Filter to reasonable scale values for layer weights
                if scale > 0.005 && scale < 0.1 {
                    scales.push(scale);
                }
            }
        }
    }
    
    // Deduplicate consecutive scales
    let mut unique_scales = Vec::new();
    let mut prev: Option<f32> = None;
    for s in scales {
        if prev.is_none() || (s - prev.unwrap()).abs() > 0.0001 {
            unique_scales.push(s);
            prev = Some(s);
        }
    }
    
    Ok(unique_scales)
}

/// Load raw INT8 weights from weights.bin
pub fn load_raw_weights(weights_path: &Path) -> Result<Vec<i8>> {
    let data = std::fs::read(weights_path)
        .context("Failed to read weights.bin")?;
    
    // Reinterpret bytes as i8
    let weights: Vec<i8> = data.iter().map(|&b| b as i8).collect();
    Ok(weights)
}

/// Create a shape constant tensor for Reshape operations
fn make_shape_tensor(name: &str, dims: &[i64]) -> TensorProto {
    TensorProto {
        name: name.to_string(),
        dims: vec![dims.len() as i64],
        data_type: TensorDataType::Int64 as i32,
        int64_data: dims.to_vec(),
        raw_data: Vec::new(),
        float_data: Vec::new(),
        int32_data: Vec::new(),
    }
}

/// Create an ONNX node
fn make_node(name: &str, op_type: &str, inputs: Vec<String>, outputs: Vec<String>) -> NodeProto {
    NodeProto {
        name: name.to_string(),
        op_type: op_type.to_string(),
        input: inputs,
        output: outputs,
        domain: String::new(),
        attribute: Vec::new(),
        doc_string: String::new(),
    }
}

/// Add attributes to a node
fn add_attrs(mut node: NodeProto, attrs: Vec<AttributeProto>) -> NodeProto {
    node.attribute = attrs;
    node
}

/// Dequantize INT8 weights to FP32 using scale
fn dequantize_weights(weights: &[i8], scale: f32) -> Vec<f32> {
    weights.iter().map(|&w| w as f32 * scale).collect()
}

/// Create FP32 weight initializer from dequantized data
fn make_fp32_initializer(name: &str, dims: &[i64], data: &[f32]) -> TensorProto {
    let raw_data: Vec<u8> = data.iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();

    TensorProto {
        name: name.to_string(),
        dims: dims.to_vec(),
        data_type: TensorDataType::Float as i32,
        raw_data,
        float_data: Vec::new(),
        int32_data: Vec::new(),
        int64_data: Vec::new(),
    }
}

/// AEC ONNX Exporter
///
/// Builds an ONNX model matching the AEC architecture:
/// - Input: [B, n_freq, n_frames] = [B, 256, 8]
/// - Encoder: Conv1D layers with downsampling
/// - GRU layers for temporal processing
/// - Decoder: ConvTranspose1D for upsampling
/// - Output: [B, n_freq, 2] = [B, 256, 2] sigmoid mask
pub struct AecOnnxExporter {
    params: AecModelParams,
    weights: Vec<i8>,
    scales: Vec<f32>,
    model_name: String,
}

impl AecOnnxExporter {
    pub fn new(model_name: String, weights: Vec<i8>, scales: Vec<f32>) -> Self {
        Self {
            params: AecModelParams::default(),
            weights,
            scales,
            model_name,
        }
    }

    /// Export to ONNX file
    pub fn export<P: AsRef<Path>>(&self, output_path: P) -> Result<()> {
        let model = self.build_model()?;

        let mut buf = Vec::new();
        model.encode(&mut buf).context("Failed to encode ONNX model")?;

        let mut file = File::create(output_path.as_ref())
            .context("Failed to create output file")?;
        file.write_all(&buf).context("Failed to write ONNX model")?;

        Ok(())
    }

    fn build_model(&self) -> Result<ModelProto> {
        let graph = self.build_aec_graph()?;

        Ok(ModelProto {
            ir_version: ONNX_IR_VERSION,
            opset_import: vec![OperatorSetIdProto {
                domain: String::new(),
                version: ONNX_OPSET_VERSION,
            }],
            producer_name: "mgk-decompiler".to_string(),
            producer_version: env!("CARGO_PKG_VERSION").to_string(),
            domain: String::new(),
            model_version: 1,
            doc_string: format!("AEC model decompiled from MGK: {}", self.model_name),
            graph: Some(graph),
            metadata_props: vec![
                StringStringEntryProto {
                    key: "source".to_string(),
                    value: "Ingenic Magik T41 NPU".to_string(),
                },
            ],
        })
    }

    fn get_scale(&self, idx: usize) -> f32 {
        self.scales.get(idx).copied().unwrap_or(1.0 / 127.0)
    }

    fn build_aec_graph(&self) -> Result<GraphProto> {
        let p = &self.params;
        let mut nodes = Vec::new();
        let mut initializers = Vec::new();
        let mut weight_offset = 0usize;
        let mut scale_idx = 0usize;

        // Input: [B, n_freq, n_frames] = [B, 256, 8]
        let input = make_value_info("input", TensorDataType::Float, &[-1, p.n_freq, p.n_frames]);

        // ============================================================
        // Layer 1: Expand frames to channels (8 -> 32)
        // Conv1d: input [B, 256, 8] -> [B, 256, 32] via [8, 32] kernel (actually transpose conv-like)
        // Actually it's [B, 8, 256] -> permute -> Conv -> [B, 32, 256]
        // ============================================================

        // Transpose input: [B, 256, 8] -> [B, 8, 256]
        let node = add_attrs(
            make_node("transpose_in", "Transpose", vec!["input".into()], vec!["transposed".into()]),
            vec![AttributeProto::ints("perm", vec![0, 2, 1])]
        );
        nodes.push(node);

        // expand Conv1d: [B, 8, 256] -> [B, 32, 256]
        // Weight shape: [out_ch, in_ch, kernel] = [32, 8, 1]
        let expand_size = (p.n_channels * p.n_frames) as usize;
        let expand_weights = self.extract_and_dequant(weight_offset, expand_size, &mut scale_idx);
        weight_offset += expand_size;

        initializers.push(make_fp32_initializer(
            "expand_weight", &[p.n_channels, p.n_frames, 1], &expand_weights
        ));
        initializers.push(make_fp32_initializer(
            "expand_bias", &[p.n_channels], &vec![0.0f32; p.n_channels as usize]
        ));

        let node = add_attrs(
            make_node("expand", "Conv",
                vec!["transposed".into(), "expand_weight".into(), "expand_bias".into()],
                vec!["expanded".into()]),
            vec![
                AttributeProto::ints("kernel_shape", vec![1]),
                AttributeProto::ints("pads", vec![0, 0]),
                AttributeProto::ints("strides", vec![1]),
            ]
        );
        nodes.push(node);

        // ReLU
        nodes.push(make_node("expand_relu", "Relu", vec!["expanded".into()], vec!["expand_out".into()]));

        // ============================================================
        // Layer 2: Downsample 256 -> 128
        // Conv1d: [B, 32, 256] -> [B, 32, 128] with kernel=2, stride=2
        // ============================================================
        let down1_size = (p.n_channels * p.n_channels * 2) as usize;
        let down1_weights = self.extract_and_dequant(weight_offset, down1_size, &mut scale_idx);
        weight_offset += down1_size;

        initializers.push(make_fp32_initializer(
            "down1_weight", &[p.n_channels, p.n_channels, 2], &down1_weights
        ));
        initializers.push(make_fp32_initializer(
            "down1_bias", &[p.n_channels], &vec![0.0f32; p.n_channels as usize]
        ));

        let node = add_attrs(
            make_node("down1", "Conv",
                vec!["expand_out".into(), "down1_weight".into(), "down1_bias".into()],
                vec!["down1_out".into()]),
            vec![
                AttributeProto::ints("kernel_shape", vec![2]),
                AttributeProto::ints("strides", vec![2]),
                AttributeProto::ints("pads", vec![0, 0]),
            ]
        );
        nodes.push(node);
        nodes.push(make_node("down1_relu", "Relu", vec!["down1_out".into()], vec!["down1_relu_out".into()]));

        // ============================================================
        // Layer 3: Conv at 128
        // ============================================================
        let conv1_size = (p.n_channels * p.n_channels) as usize;
        let conv1_weights = self.extract_and_dequant(weight_offset, conv1_size, &mut scale_idx);
        weight_offset += conv1_size;

        initializers.push(make_fp32_initializer(
            "conv1_weight", &[p.n_channels, p.n_channels, 1], &conv1_weights
        ));
        initializers.push(make_fp32_initializer(
            "conv1_bias", &[p.n_channels], &vec![0.0f32; p.n_channels as usize]
        ));

        let node = add_attrs(
            make_node("conv1", "Conv",
                vec!["down1_relu_out".into(), "conv1_weight".into(), "conv1_bias".into()],
                vec!["conv1_out".into()]),
            vec![
                AttributeProto::ints("kernel_shape", vec![1]),
                AttributeProto::ints("pads", vec![0, 0]),
            ]
        );
        nodes.push(node);
        nodes.push(make_node("conv1_relu", "Relu", vec!["conv1_out".into()], vec!["conv1_relu_out".into()]));

        // ============================================================
        // Layer 4: Downsample 128 -> 64
        // ============================================================
        let down2_size = (p.n_channels * p.n_channels * 2) as usize;
        let down2_weights = self.extract_and_dequant(weight_offset, down2_size, &mut scale_idx);
        weight_offset += down2_size;

        initializers.push(make_fp32_initializer(
            "down2_weight", &[p.n_channels, p.n_channels, 2], &down2_weights
        ));
        initializers.push(make_fp32_initializer(
            "down2_bias", &[p.n_channels], &vec![0.0f32; p.n_channels as usize]
        ));

        let node = add_attrs(
            make_node("down2", "Conv",
                vec!["conv1_relu_out".into(), "down2_weight".into(), "down2_bias".into()],
                vec!["down2_out".into()]),
            vec![
                AttributeProto::ints("kernel_shape", vec![2]),
                AttributeProto::ints("strides", vec![2]),
                AttributeProto::ints("pads", vec![0, 0]),
            ]
        );
        nodes.push(node);
        nodes.push(make_node("down2_relu", "Relu", vec!["down2_out".into()], vec!["down2_relu_out".into()]));

        // ============================================================
        // Feature layers (3 Conv1d layers at 64 freq bins)
        // ============================================================
        let mut prev_out = "down2_relu_out".to_string();
        for i in 0..3 {
            let feat_size = (p.n_channels * p.n_channels) as usize;
            let feat_weights = self.extract_and_dequant(weight_offset, feat_size, &mut scale_idx);
            weight_offset += feat_size;

            let w_name = format!("feat{}_weight", i);
            let b_name = format!("feat{}_bias", i);
            let out_name = format!("feat{}_out", i);
            let relu_name = format!("feat{}_relu", i);
            let relu_out = format!("feat{}_relu_out", i);

            initializers.push(make_fp32_initializer(&w_name, &[p.n_channels, p.n_channels, 1], &feat_weights));
            initializers.push(make_fp32_initializer(&b_name, &[p.n_channels], &vec![0.0f32; p.n_channels as usize]));

            let node = add_attrs(
                make_node(&format!("feat{}", i), "Conv",
                    vec![prev_out.clone(), w_name, b_name],
                    vec![out_name.clone()]),
                vec![
                    AttributeProto::ints("kernel_shape", vec![1]),
                    AttributeProto::ints("pads", vec![0, 0]),
                ]
            );
            nodes.push(node);
            nodes.push(make_node(&relu_name, "Relu", vec![out_name], vec![relu_out.clone()]));
            prev_out = relu_out;
        }

        // ============================================================
        // Reshape for GRU: [B, 32, 64] -> [B, 64, 32]
        // GRU expects [batch, seq_len, features]
        // ============================================================
        let node = add_attrs(
            make_node("pre_gru_transpose", "Transpose", vec![prev_out], vec!["gru_input".into()]),
            vec![AttributeProto::ints("perm", vec![0, 2, 1])]
        );
        nodes.push(node);

        // ============================================================
        // GRU1: [B, 64, 32] -> [B, 64, 32]
        // ONNX GRU: X=[B,seq,input], W=[num_dir,3*hidden,input], R=[num_dir,3*hidden,hidden]
        // ============================================================
        let h = p.hidden_size;
        let c = p.n_channels;

        // GRU weights: W_ih [3*H, C] and W_hh [3*H, H]
        let gru1_ih_size = (3 * h * c) as usize;
        let gru1_hh_size = (3 * h * h) as usize;
        let gru1_ih = self.extract_and_dequant(weight_offset, gru1_ih_size, &mut scale_idx);
        weight_offset += gru1_ih_size;
        let gru1_hh = self.extract_and_dequant(weight_offset, gru1_hh_size, &mut scale_idx);
        weight_offset += gru1_hh_size;

        initializers.push(make_fp32_initializer("gru1_W", &[1, 3*h, c], &gru1_ih));
        initializers.push(make_fp32_initializer("gru1_R", &[1, 3*h, h], &gru1_hh));
        initializers.push(make_fp32_initializer("gru1_B", &[1, 6*h], &vec![0.0f32; (6*h) as usize]));

        let node = add_attrs(
            make_node("gru1", "GRU",
                vec!["gru_input".into(), "gru1_W".into(), "gru1_R".into(), "gru1_B".into()],
                vec!["gru1_Y".into(), "gru1_Y_h".into()]),
            vec![
                AttributeProto::int("hidden_size", h),
                AttributeProto::string("direction", "forward"),
            ]
        );
        nodes.push(node);

        // GRU output is [seq, num_dir, batch, hidden] = [64, 1, B, 32]
        // Need to squeeze and reshape to [B, 64, 32]
        // Squeeze dim 1: [64, 1, B, 32] -> [64, B, 32]
        initializers.push(make_shape_tensor("squeeze_axes", &[1]));
        nodes.push(make_node("gru1_squeeze", "Squeeze",
            vec!["gru1_Y".into(), "squeeze_axes".into()],
            vec!["gru1_squeezed".into()]));

        // Transpose: [64, B, 32] -> [B, 64, 32]
        let node = add_attrs(
            make_node("gru1_transpose", "Transpose", vec!["gru1_squeezed".into()], vec!["gru1_out".into()]),
            vec![AttributeProto::ints("perm", vec![1, 0, 2])]
        );
        nodes.push(node);

        // ============================================================
        // GRU2 (bidirectional): [B, 64, 32] -> [B, 64, 64]
        // ============================================================
        // Forward weights
        let gru2_ih = self.extract_and_dequant(weight_offset, gru1_hh_size, &mut scale_idx);
        weight_offset += gru1_hh_size;
        let gru2_hh = self.extract_and_dequant(weight_offset, gru1_hh_size, &mut scale_idx);
        weight_offset += gru1_hh_size;
        // Reverse weights
        let gru2_ih_r = self.extract_and_dequant(weight_offset, gru1_hh_size, &mut scale_idx);
        weight_offset += gru1_hh_size;
        let gru2_hh_r = self.extract_and_dequant(weight_offset, gru1_hh_size, &mut scale_idx);
        weight_offset += gru1_hh_size;

        // Combine forward and reverse into [2, 3*H, H]
        let mut gru2_W = gru2_ih.clone();
        gru2_W.extend(&gru2_ih_r);
        let mut gru2_R = gru2_hh.clone();
        gru2_R.extend(&gru2_hh_r);

        initializers.push(make_fp32_initializer("gru2_W", &[2, 3*h, h], &gru2_W));
        initializers.push(make_fp32_initializer("gru2_R", &[2, 3*h, h], &gru2_R));
        initializers.push(make_fp32_initializer("gru2_B", &[2, 6*h], &vec![0.0f32; (12*h) as usize]));

        let node = add_attrs(
            make_node("gru2", "GRU",
                vec!["gru1_out".into(), "gru2_W".into(), "gru2_R".into(), "gru2_B".into()],
                vec!["gru2_Y".into(), "gru2_Y_h".into()]),
            vec![
                AttributeProto::int("hidden_size", h),
                AttributeProto::string("direction", "bidirectional"),
            ]
        );
        nodes.push(node);

        // GRU2 output: [64, 2, B, 32] -> need [B, 64, 64]
        // First reshape [64, 2, B, 32] -> [64, B, 64]
        initializers.push(make_shape_tensor("gru2_shape", &[64, -1, 64]));
        nodes.push(make_node("gru2_reshape", "Reshape",
            vec!["gru2_Y".into(), "gru2_shape".into()],
            vec!["gru2_reshaped".into()]));

        // Transpose: [64, B, 64] -> [B, 64, 64]
        let node = add_attrs(
            make_node("gru2_transpose", "Transpose", vec!["gru2_reshaped".into()], vec!["gru2_out".into()]),
            vec![AttributeProto::ints("perm", vec![1, 0, 2])]
        );
        nodes.push(node);

        // Transpose for Conv: [B, 64, 64] -> [B, 64, 64] (channels first)
        let node = add_attrs(
            make_node("decoder_transpose", "Transpose", vec!["gru2_out".into()], vec!["decoder_in".into()]),
            vec![AttributeProto::ints("perm", vec![0, 2, 1])]
        );
        nodes.push(node);

        // ============================================================
        // Decoder: Upsample back to 256 freq bins
        // ============================================================
        // up1: [B, 64, 64] -> [B, 32, 128] via ConvTranspose1d
        let up1_size = (p.n_channels * 2 * h * 2) as usize;
        let up1_weights = self.extract_and_dequant(weight_offset, up1_size, &mut scale_idx);
        weight_offset += up1_size;

        initializers.push(make_fp32_initializer("up1_weight", &[2*h, p.n_channels, 2], &up1_weights));
        initializers.push(make_fp32_initializer("up1_bias", &[p.n_channels], &vec![0.0f32; p.n_channels as usize]));

        let node = add_attrs(
            make_node("up1", "ConvTranspose",
                vec!["decoder_in".into(), "up1_weight".into(), "up1_bias".into()],
                vec!["up1_out".into()]),
            vec![
                AttributeProto::ints("kernel_shape", vec![2]),
                AttributeProto::ints("strides", vec![2]),
            ]
        );
        nodes.push(node);
        nodes.push(make_node("up1_relu", "Relu", vec!["up1_out".into()], vec!["up1_relu_out".into()]));

        // up2: [B, 32, 128] -> [B, 32, 256] via ConvTranspose1d
        let up2_size = (p.n_channels * p.n_channels * 2) as usize;
        let up2_weights = self.extract_and_dequant(weight_offset, up2_size, &mut scale_idx);
        weight_offset += up2_size;

        initializers.push(make_fp32_initializer("up2_weight", &[p.n_channels, p.n_channels, 2], &up2_weights));
        initializers.push(make_fp32_initializer("up2_bias", &[p.n_channels], &vec![0.0f32; p.n_channels as usize]));

        let node = add_attrs(
            make_node("up2", "ConvTranspose",
                vec!["up1_relu_out".into(), "up2_weight".into(), "up2_bias".into()],
                vec!["up2_out".into()]),
            vec![
                AttributeProto::ints("kernel_shape", vec![2]),
                AttributeProto::ints("strides", vec![2]),
            ]
        );
        nodes.push(node);
        nodes.push(make_node("up2_relu", "Relu", vec!["up2_out".into()], vec!["up2_relu_out".into()]));

        // ============================================================
        // Output: [B, 32, 256] -> [B, 2, 256] -> sigmoid
        // ============================================================
        let out_size = (2 * p.n_channels) as usize;
        let out_weights = self.extract_and_dequant(weight_offset, out_size, &mut scale_idx);
        let _ = weight_offset + out_size;

        initializers.push(make_fp32_initializer("out_weight", &[2, p.n_channels, 1], &out_weights));
        initializers.push(make_fp32_initializer("out_bias", &[2], &vec![0.0f32; 2]));

        let node = add_attrs(
            make_node("output_conv", "Conv",
                vec!["up2_relu_out".into(), "out_weight".into(), "out_bias".into()],
                vec!["pre_sigmoid".into()]),
            vec![
                AttributeProto::ints("kernel_shape", vec![1]),
                AttributeProto::ints("pads", vec![0, 0]),
            ]
        );
        nodes.push(node);

        // Sigmoid activation
        nodes.push(make_node("sigmoid", "Sigmoid", vec!["pre_sigmoid".into()], vec!["mask".into()]));

        // Final transpose: [B, 2, 256] -> [B, 256, 2]
        let node = add_attrs(
            make_node("output_transpose", "Transpose", vec!["mask".into()], vec!["output".into()]),
            vec![AttributeProto::ints("perm", vec![0, 2, 1])]
        );
        nodes.push(node);

        // Output: [B, 256, 2]
        let output = make_value_info("output", TensorDataType::Float, &[-1, p.n_freq, 2]);

        eprintln!("Built AEC ONNX graph:");
        eprintln!("  Nodes: {}", nodes.len());
        eprintln!("  Initializers: {}", initializers.len());
        eprintln!("  Weights used: {} bytes", weight_offset);

        Ok(GraphProto {
            name: self.model_name.clone(),
            node: nodes,
            input: vec![input],
            output: vec![output],
            initializer: initializers,
            doc_string: String::new(),
            value_info: Vec::new(),
        })
    }

    /// Extract weights from buffer and dequantize
    fn extract_and_dequant(&self, offset: usize, size: usize, scale_idx: &mut usize) -> Vec<f32> {
        let scale = self.get_scale(*scale_idx);
        *scale_idx += 1;

        if offset + size <= self.weights.len() {
            dequantize_weights(&self.weights[offset..offset+size], scale)
        } else {
            // Return zeros if not enough data
            vec![0.0f32; size]
        }
    }
}

