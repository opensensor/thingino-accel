//! Parser for .rodata section of MGK files
//!
//! The .rodata section contains:
//! - Tensor names (null-terminated strings)
//! - Data format strings (NHWC, NDHWC32, etc.)
//! - Data type strings (UINT8, FP32, etc.)
//! - Quantized weights and parameters
//! - Layer names and operation paths
//! - Quantization scales (FP32 values)

use crate::types::{MgkFile, Section};
use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Extracted tensor information from rodata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorInfo {
    pub name: String,
    pub offset: usize,
    pub data_format: Option<String>,
    pub data_type: Option<String>,
}

/// Extracted layer information from rodata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerInfo {
    pub name: String,
    pub layer_type: String,
    pub layer_id: Option<u32>,
    pub offset: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_fused: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fused_ops: Option<Vec<String>>,
}

/// Operation path info (e.g., "Gru/gru_ubit8/2/0/0/")
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpPathInfo {
    pub op_type: String,
    pub kernel_name: String,
    pub params: Vec<String>,
    pub offset: usize,
}

/// Quantization scale group (consecutive scales for a layer)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScaleGroup {
    pub start_offset: usize,
    pub scales: Vec<f32>,
    pub layer_hint: Option<String>,
}

/// Model metadata extracted from rodata
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub tensors: Vec<TensorInfo>,
    pub layers: Vec<LayerInfo>,
    pub op_paths: Vec<OpPathInfo>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub input_names: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub output_names: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub scale_groups: Vec<ScaleGroup>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub tensor_formats: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub data_types: Vec<String>,
}

/// Known data formats in MGK files
const DATA_FORMATS: &[&str] = &[
    "NHWC", "NCHW", "NDHWC32", "NDHWC", "NC", "N", "OIHW", "HWIO",
];

/// Known data types in MGK files
const DATA_TYPES: &[&str] = &[
    "UINT8", "INT8", "INT4", "FP32", "FP16", "INT32", "INT16",
];

/// Known layer type prefixes in rodata
const LAYER_TYPE_PREFIXES: &[(&str, &str)] = &[
    ("QuantizeGRU", "GRU"),
    ("QuantizeBatchNorm", "BatchNorm"),
    ("QuantizeFeature", "Feature"),
    ("QuantizeConv2DWrapper", "Conv2D_Fused"),
    ("QuantizeConv", "Conv"),
    ("QuantizeRelu", "Relu"),
    ("QuantizeAdd", "Add"),
    ("QuantizePool", "Pool"),
    ("QuantizeConcat", "Concat"),
    ("QuantizeConcatInference", "Concat"),
    ("QuantizeUpsample", "Upsample"),
    ("QuantizeSlice", "Slice"),
    ("QuantizeReshape", "Reshape"),
    ("QuantizePermute", "Permute"),
    ("QuantizeSigmoid", "Sigmoid"),
    ("QuantizeMul", "Mul"),
    ("QuantizeSoftmax", "Softmax"),
    ("QuantizeWeight", "Weight"),
];

/// Fusion indicators in layer names
const FUSION_INDICATORS: &[&str] = &[
    "QuantizeConv2DWrapper",
    "conv2d_tnpu",
    "QuantizeWeight",
    "fuse_",
    "_fused",
];

/// Extract tensor information from the rodata section
pub fn extract_tensor_info(mgk: &MgkFile) -> Result<Vec<TensorInfo>> {
    let rodata = mgk.sections.iter()
        .find(|s| s.name == ".rodata")
        .ok_or_else(|| anyhow::anyhow!("No .rodata section found"))?;
    
    let mut tensors = Vec::new();
    let data = &rodata.data;
    
    // Scan for tensor name patterns
    // Tensor names typically start with "onnx__" or "__" and are null-terminated
    let mut i = 0;
    while i < data.len() {
        if let Some(name) = try_extract_tensor_name(data, i) {
            let offset = i;
            
            // Look for associated format/type strings nearby
            let (data_format, data_type) = find_associated_metadata(data, i + name.len() + 1);
            
            tensors.push(TensorInfo {
                name,
                offset,
                data_format,
                data_type,
            });
            
            // Skip past this string
            i += tensors.last().unwrap().name.len() + 1;
        } else {
            i += 1;
        }
    }
    
    Ok(tensors)
}

/// Try to extract a tensor name starting at the given offset
fn try_extract_tensor_name(data: &[u8], offset: usize) -> Option<String> {
    // Check for common tensor name prefixes
    let prefixes = [
        "onnx__", "__FormatConvert", "__Reshape", "__ConvertTensor",
        "__Transpose", "__L", "input", "output", "hidden",
        "layer_", "Conv", "Relu", "Add", "Concat"
    ];

    for prefix in prefixes {
        if data[offset..].starts_with(prefix.as_bytes()) {
            // Extract null-terminated string
            if let Some(end) = data[offset..].iter().position(|&b| b == 0) {
                let s = String::from_utf8_lossy(&data[offset..offset + end]).to_string();
                // Validate it looks like a tensor name (alphanumeric + underscore + dash)
                if s.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '-') && s.len() > 2 {
                    return Some(s);
                }
            }
        }
    }

    None
}

/// Find data format and type strings near a tensor name
fn find_associated_metadata(data: &[u8], start: usize) -> (Option<String>, Option<String>) {
    let mut format = None;
    let mut dtype = None;
    
    // Search within 64 bytes after the tensor name
    let search_end = (start + 64).min(data.len());
    
    for i in start..search_end {
        // Check for data format
        if format.is_none() {
            for fmt in DATA_FORMATS {
                if data[i..].starts_with(fmt.as_bytes()) {
                    // Verify it's null-terminated or followed by non-alpha
                    let end = i + fmt.len();
                    if end < data.len() && (data[end] == 0 || !data[end].is_ascii_alphabetic()) {
                        format = Some(fmt.to_string());
                        break;
                    }
                }
            }
        }
        
        // Check for data type
        if dtype.is_none() {
            for dt in DATA_TYPES {
                if data[i..].starts_with(dt.as_bytes()) {
                    let end = i + dt.len();
                    if end < data.len() && (data[end] == 0 || !data[end].is_ascii_alphabetic()) {
                        dtype = Some(dt.to_string());
                        break;
                    }
                }
            }
        }
        
        if format.is_some() && dtype.is_some() {
            break;
        }
    }
    
    (format, dtype)
}

/// Get the rodata section
pub fn get_rodata_section(mgk: &MgkFile) -> Option<&Section> {
    mgk.sections.iter().find(|s| s.name == ".rodata")
}

/// Extract layer information from rodata using multiple patterns
/// Pattern 1 (AEC style): layer_N_Type (e.g., layer_37_QuantizeGRU)
/// Pattern 2 (YOLO style): NNN_Quantize (e.g., 414_Quantize)
/// Pattern 3 (PTQ fused): ptq_model_<op>_N_Quantize
/// Pattern 4 (Output): NNN_output_last_layer
pub fn extract_layer_info(mgk: &MgkFile) -> Result<Vec<LayerInfo>> {
    let rodata = mgk.sections.iter()
        .find(|s| s.name == ".rodata")
        .ok_or_else(|| anyhow::anyhow!("No .rodata section found"))?;

    let mut layers = Vec::new();
    let data = &rodata.data;

    let mut i = 0;
    while i < data.len() {
        // Pattern 1: layer_N_Type (AEC style)
        if data[i..].starts_with(b"layer_") {
            if let Some(end) = data[i..].iter().position(|&b| b == 0) {
                let name = String::from_utf8_lossy(&data[i..i + end]).to_string();
                let layer_id = parse_layer_id(&name);
                let layer_type = parse_layer_type(&name);
                let (is_fused, fused_ops) = detect_fusion(&name);

                layers.push(LayerInfo {
                    name: name.clone(),
                    layer_type,
                    layer_id,
                    offset: i,
                    is_fused: if is_fused { Some(true) } else { None },
                    fused_ops: if fused_ops.is_empty() { None } else { Some(fused_ops) },
                });

                i += end + 1;
                continue;
            }
        }

        // Pattern 2: NNN_Quantize (YOLO style) - starts with 3+ digits
        if i + 3 < data.len() && data[i].is_ascii_digit() && data[i + 1].is_ascii_digit() && data[i + 2].is_ascii_digit() {
            if let Some(end) = data[i..].iter().position(|&b| b == 0) {
                let name = String::from_utf8_lossy(&data[i..i + end]).to_string();
                if name.contains("Quantize") || name.contains("output_last_layer") {
                    let layer_id = parse_yolo_layer_id(&name);
                    let layer_type = parse_layer_type(&name);
                    let (is_fused, fused_ops) = detect_fusion(&name);

                    layers.push(LayerInfo {
                        name: name.clone(),
                        layer_type,
                        layer_id,
                        offset: i,
                        is_fused: if is_fused { Some(true) } else { None },
                        fused_ops: if fused_ops.is_empty() { None } else { Some(fused_ops) },
                    });

                    i += end + 1;
                    continue;
                }
            }
        }

        // Pattern 3: ptq_model_<op>_N_Quantize (PTQ fused)
        if data[i..].starts_with(b"ptq_model_") {
            if let Some(end) = data[i..].iter().position(|&b| b == 0) {
                let name = String::from_utf8_lossy(&data[i..i + end]).to_string();
                let layer_id = parse_ptq_layer_id(&name);
                let layer_type = parse_layer_type(&name);

                layers.push(LayerInfo {
                    name: name.clone(),
                    layer_type,
                    layer_id,
                    offset: i,
                    is_fused: Some(true),
                    fused_ops: Some(vec!["Conv".to_string(), "BN".to_string(), "ReLU".to_string()]),
                });

                i += end + 1;
                continue;
            }
        }

        // Pattern 4: onnx__QuantizeXXX_NNN (ONNX intermediate tensors)
        if data[i..].starts_with(b"onnx__Quantize") {
            if let Some(end) = data[i..].iter().position(|&b| b == 0) {
                let name = String::from_utf8_lossy(&data[i..i + end]).to_string();
                let layer_id = parse_onnx_layer_id(&name);
                let layer_type = parse_layer_type(&name);
                let (is_fused, fused_ops) = detect_fusion(&name);

                layers.push(LayerInfo {
                    name: name.clone(),
                    layer_type,
                    layer_id,
                    offset: i,
                    is_fused: if is_fused { Some(true) } else { None },
                    fused_ops: if fused_ops.is_empty() { None } else { Some(fused_ops) },
                });

                i += end + 1;
                continue;
            }
        }

        i += 1;
    }

    // Sort by layer_id
    layers.sort_by_key(|l| l.layer_id.unwrap_or(u32::MAX));

    Ok(layers)
}

/// Parse layer ID from layer name (e.g., "layer_80_..." -> 80)
fn parse_layer_id(name: &str) -> Option<u32> {
    let parts: Vec<&str> = name.split('_').collect();
    if parts.len() >= 2 {
        parts[1].parse().ok()
    } else {
        None
    }
}

/// Parse layer ID from YOLO-style name (e.g., "414_Quantize..." -> 414)
fn parse_yolo_layer_id(name: &str) -> Option<u32> {
    let parts: Vec<&str> = name.split('_').collect();
    if !parts.is_empty() {
        parts[0].parse().ok()
    } else {
        None
    }
}

/// Parse layer ID from PTQ-style name (e.g., "ptq_model_conv_5_Quantize" -> 5)
fn parse_ptq_layer_id(name: &str) -> Option<u32> {
    // Find the number before _Quantize
    let parts: Vec<&str> = name.split('_').collect();
    for (i, part) in parts.iter().enumerate() {
        if *part == "Quantize" && i > 0 {
            return parts[i - 1].parse().ok();
        }
    }
    None
}

/// Parse layer ID from ONNX-style name (e.g., "onnx__QuantizeConv_500" -> 500)
fn parse_onnx_layer_id(name: &str) -> Option<u32> {
    let parts: Vec<&str> = name.split('_').collect();
    if let Some(last) = parts.last() {
        last.parse().ok()
    } else {
        None
    }
}

/// Detect if a layer is fused (Conv+BN+ReLU)
fn detect_fusion(name: &str) -> (bool, Vec<String>) {
    let mut is_fused = false;
    let mut fused_ops = Vec::new();

    for indicator in FUSION_INDICATORS {
        if name.contains(indicator) {
            is_fused = true;
            break;
        }
    }

    if is_fused {
        // Detect which ops are fused
        if name.contains("Conv") || name.contains("conv") {
            fused_ops.push("Conv".to_string());
        }
        if name.contains("BatchNorm") || name.contains("bn") || name.contains("BN") {
            fused_ops.push("BN".to_string());
        }
        if name.contains("Relu") || name.contains("relu") || name.contains("ReLU") {
            fused_ops.push("ReLU".to_string());
        }
    }

    (is_fused, fused_ops)
}

/// Parse layer type from layer name
fn parse_layer_type(name: &str) -> String {
    for (pattern, layer_type) in LAYER_TYPE_PREFIXES {
        if name.contains(pattern) {
            return layer_type.to_string();
        }
    }

    // Try to extract type from name
    if name.contains("GRU") || name.contains("Gru") {
        return "GRU".to_string();
    }
    if name.contains("BatchNorm") {
        return "BatchNorm".to_string();
    }
    if name.contains("Feature") {
        return "Feature".to_string();
    }
    if name.contains("Conv") || name.contains("conv") {
        return "Conv".to_string();
    }
    if name.contains("Pool") || name.contains("pool") {
        return "Pool".to_string();
    }
    if name.contains("Add") && !name.contains("Addr") {
        return "Add".to_string();
    }
    if name.contains("Concat") || name.contains("concat") {
        return "Concat".to_string();
    }
    if name.contains("Upsample") || name.contains("upsample") {
        return "Upsample".to_string();
    }
    if name.contains("Reshape") || name.contains("reshape") {
        return "Reshape".to_string();
    }
    if name.contains("Sigmoid") || name.contains("sigmoid") {
        return "Sigmoid".to_string();
    }
    if name.contains("Relu") || name.contains("relu") || name.contains("ReLU") {
        return "ReLU".to_string();
    }
    if name.contains("output_last_layer") {
        return "Output".to_string();
    }

    // For YOLO-style NNN_Quantize names, mark as QuantizedLayer
    if name.ends_with("_Quantize") {
        return "QuantizedLayer".to_string();
    }

    "Unknown".to_string()
}

/// Extract operation path info (e.g., "Gru/gru_ubit8/2/0/0/")
pub fn extract_op_paths(mgk: &MgkFile) -> Result<Vec<OpPathInfo>> {
    let rodata = mgk.sections.iter()
        .find(|s| s.name == ".rodata")
        .ok_or_else(|| anyhow::anyhow!("No .rodata section found"))?;

    let mut op_paths = Vec::new();
    let data = &rodata.data;

    // Known op type prefixes
    let op_prefixes = [
        "FormatConvert/", "Normalize/", "Reshape/", "Gru/", "Permute/",
        "Concat/", "UpSample/", "Conv/", "Pool/", "Add/", "Slice/",
    ];

    let mut i = 0;
    while i < data.len() {
        for prefix in &op_prefixes {
            if data[i..].starts_with(prefix.as_bytes()) {
                if let Some(end) = data[i..].iter().position(|&b| b == 0) {
                    let path = String::from_utf8_lossy(&data[i..i + end]).to_string();

                    // Parse the path components
                    let parts: Vec<&str> = path.split('/').collect();
                    if parts.len() >= 2 {
                        let op_type = parts[0].to_string();
                        let kernel_name = parts[1].to_string();
                        let params: Vec<String> = parts[2..].iter()
                            .filter(|s| !s.is_empty())
                            .map(|s| s.to_string())
                            .collect();

                        op_paths.push(OpPathInfo {
                            op_type,
                            kernel_name,
                            params,
                            offset: i,
                        });
                    }

                    i += end + 1;
                    break;
                }
            }
        }
        i += 1;
    }

    Ok(op_paths)
}

/// Extract complete model metadata from rodata
pub fn extract_model_metadata(mgk: &MgkFile) -> Result<ModelMetadata> {
    let tensors = extract_tensor_info(mgk)?;
    let layers = extract_layer_info(mgk)?;
    let op_paths = extract_op_paths(mgk)?;
    let scale_groups = extract_scales(mgk).unwrap_or_default();
    let tensor_formats = extract_tensor_formats(mgk);
    let data_types = extract_data_types(mgk);

    // Identify inputs and outputs
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();

    for tensor in &tensors {
        if tensor.name.starts_with("input") || tensor.name == "hidden" {
            inputs.push(tensor.name.clone());
        }
        if tensor.name.contains("_last_layer") || tensor.name.contains("output") {
            outputs.push(tensor.name.clone());
        }
    }

    // Also check layers for outputs
    for layer in &layers {
        if layer.name.contains("output_last_layer") && !outputs.contains(&layer.name) {
            outputs.push(layer.name.clone());
        }
    }

    // Set input_names and output_names as copies for ONNX export
    let input_names = inputs.clone();
    let output_names = outputs.clone();

    Ok(ModelMetadata {
        tensors,
        layers,
        op_paths,
        inputs,
        outputs,
        input_names,
        output_names,
        scale_groups,
        tensor_formats,
        data_types,
    })
}

/// Extract strings from a section at specific offsets
pub fn extract_string_at(data: &[u8], offset: usize) -> Option<String> {
    if offset >= data.len() {
        return None;
    }

    if let Some(end) = data[offset..].iter().position(|&b| b == 0) {
        let s = String::from_utf8_lossy(&data[offset..offset + end]).to_string();
        if !s.is_empty() && s.is_ascii() {
            return Some(s);
        }
    }

    None
}

/// Scan for quantization parameters (often stored as sequences of floats)
pub fn find_quantization_params(data: &[u8], start: usize, count: usize) -> Vec<f32> {
    let mut params = Vec::new();
    let mut offset = start;

    while offset + 4 <= data.len() && params.len() < count {
        let bytes: [u8; 4] = [data[offset], data[offset + 1], data[offset + 2], data[offset + 3]];
        let val = f32::from_le_bytes(bytes);

        // Sanity check: quantization params are usually in reasonable range
        if val.is_finite() && val.abs() < 1000.0 {
            params.push(val);
        }
        offset += 4;
    }

    params
}

/// Extract quantization scales from rodata
/// Scales are FP32 values typically in range 0.001 to 10.0
pub fn extract_scales(mgk: &MgkFile) -> Result<Vec<ScaleGroup>> {
    let rodata = mgk.sections.iter()
        .find(|s| s.name == ".rodata")
        .ok_or_else(|| anyhow::anyhow!("No .rodata section found"))?;

    let data = &rodata.data;
    let mut scales: Vec<(usize, f32)> = Vec::new();

    // Scan for FP32 values that look like quantization scales
    let mut offset = 0;
    while offset + 4 <= data.len() {
        let bytes: [u8; 4] = [data[offset], data[offset + 1], data[offset + 2], data[offset + 3]];
        let val = f32::from_le_bytes(bytes);

        // Quantization scales are typically in range 0.001 to 10.0
        if val.is_finite() && val.abs() > 0.001 && val.abs() < 10.0 {
            // Additional check: avoid values that look like addresses or integers
            let as_int = i32::from_le_bytes(bytes);
            if as_int.abs() > 1000 {
                // Likely a valid scale
                scales.push((offset, val));
            }
        }
        offset += 4;
    }

    // Group consecutive scales (offset difference <= 16 bytes)
    let mut groups = Vec::new();
    let mut current_group: Vec<(usize, f32)> = Vec::new();

    for (off, val) in scales {
        if current_group.is_empty() {
            current_group.push((off, val));
        } else {
            let last_off = current_group.last().unwrap().0;
            if off - last_off <= 16 {
                current_group.push((off, val));
            } else {
                // Start new group
                if current_group.len() >= 2 {
                    groups.push(ScaleGroup {
                        start_offset: current_group[0].0,
                        scales: current_group.iter().map(|(_, v)| *v).collect(),
                        layer_hint: None,
                    });
                }
                current_group = vec![(off, val)];
            }
        }
    }

    // Don't forget the last group
    if current_group.len() >= 2 {
        groups.push(ScaleGroup {
            start_offset: current_group[0].0,
            scales: current_group.iter().map(|(_, v)| *v).collect(),
            layer_hint: None,
        });
    }

    Ok(groups)
}

/// Extract unique tensor formats found in rodata
pub fn extract_tensor_formats(mgk: &MgkFile) -> Vec<String> {
    let rodata = match mgk.sections.iter().find(|s| s.name == ".rodata") {
        Some(s) => s,
        None => return Vec::new(),
    };

    let data = &rodata.data;
    let mut formats = Vec::new();

    for fmt in DATA_FORMATS {
        let pattern = fmt.as_bytes();
        for i in 0..data.len().saturating_sub(pattern.len()) {
            if data[i..].starts_with(pattern) {
                let end = i + pattern.len();
                if end < data.len() && (data[end] == 0 || !data[end].is_ascii_alphabetic()) {
                    if !formats.contains(&fmt.to_string()) {
                        formats.push(fmt.to_string());
                    }
                    break;
                }
            }
        }
    }

    formats
}

/// Extract unique data types found in rodata
pub fn extract_data_types(mgk: &MgkFile) -> Vec<String> {
    let rodata = match mgk.sections.iter().find(|s| s.name == ".rodata") {
        Some(s) => s,
        None => return Vec::new(),
    };

    let data = &rodata.data;
    let mut types = Vec::new();

    for dt in DATA_TYPES {
        let pattern = dt.as_bytes();
        for i in 0..data.len().saturating_sub(pattern.len()) {
            if data[i..].starts_with(pattern) {
                let end = i + pattern.len();
                if end < data.len() && (data[end] == 0 || !data[end].is_ascii_alphabetic()) {
                    if !types.contains(&dt.to_string()) {
                        types.push(dt.to_string());
                    }
                    break;
                }
            }
        }
    }

    types
}

/// Layer graph node representing a layer and its connections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerGraphNode {
    pub name: String,
    pub layer_type: String,
    pub layer_id: Option<u32>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

/// Extract layer graph from tensor names
/// Tensor names follow patterns like:
/// - onnx__QuantizeConcatInference_500 (intermediate tensors)
/// - __FormatConvert__26 (format conversion outputs)
/// - layer_80_QuantizeBatchNorm (layer outputs)
/// - input, hidden, output (model I/O)
pub fn extract_layer_graph(mgk: &MgkFile) -> Result<Vec<LayerGraphNode>> {
    let metadata = extract_model_metadata(mgk)?;
    let mut nodes = Vec::new();

    // Group tensors by layer
    let mut layer_tensors: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();

    for tensor in &metadata.tensors {
        // Extract layer identifier from tensor name
        let layer_key = extract_layer_key(&tensor.name);
        layer_tensors.entry(layer_key).or_default().push(tensor.name.clone());
    }

    // Create nodes from layers
    for layer in &metadata.layers {
        let layer_key = format!("layer_{}", layer.layer_id.unwrap_or(0));
        let tensors = layer_tensors.get(&layer_key).cloned().unwrap_or_default();

        nodes.push(LayerGraphNode {
            name: layer.name.clone(),
            layer_type: layer.layer_type.clone(),
            layer_id: layer.layer_id,
            inputs: Vec::new(), // Would need more analysis to determine
            outputs: tensors,
        });
    }

    // Add nodes for format conversions
    for tensor in &metadata.tensors {
        if tensor.name.starts_with("__FormatConvert") {
            nodes.push(LayerGraphNode {
                name: tensor.name.clone(),
                layer_type: "FormatConvert".to_string(),
                layer_id: None,
                inputs: Vec::new(),
                outputs: vec![tensor.name.clone()],
            });
        }
    }

    Ok(nodes)
}

/// Extract layer key from tensor name
fn extract_layer_key(name: &str) -> String {
    if name.starts_with("layer_") {
        // Extract layer_XX from layer_XX_QuantizeXXX
        let parts: Vec<&str> = name.split('_').collect();
        if parts.len() >= 2 {
            return format!("{}_{}", parts[0], parts[1]);
        }
    }
    if name.starts_with("__FormatConvert") {
        return name.to_string();
    }
    if name.starts_with("onnx__") {
        // Extract the operation type
        let parts: Vec<&str> = name.split('_').collect();
        if parts.len() >= 3 {
            return format!("onnx_{}", parts[2]);
        }
    }
    "unknown".to_string()
}
