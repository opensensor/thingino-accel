//! Parser for .rodata section of MGK files
//!
//! The .rodata section contains:
//! - Tensor names (null-terminated strings)
//! - Data format strings (NHWC, NDHWC32, etc.)
//! - Data type strings (UINT8, FP32, etc.)
//! - Quantized weights and parameters
//! - Layer names and operation paths

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
}

/// Operation path info (e.g., "Gru/gru_ubit8/2/0/0/")
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpPathInfo {
    pub op_type: String,
    pub kernel_name: String,
    pub params: Vec<String>,
    pub offset: usize,
}

/// Model metadata extracted from rodata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub tensors: Vec<TensorInfo>,
    pub layers: Vec<LayerInfo>,
    pub op_paths: Vec<OpPathInfo>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
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
    ("QuantizeConv", "Conv"),
    ("QuantizeRelu", "Relu"),
    ("QuantizeAdd", "Add"),
    ("QuantizePool", "Pool"),
    ("QuantizeConcat", "Concat"),
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

/// Extract layer information from rodata
pub fn extract_layer_info(mgk: &MgkFile) -> Result<Vec<LayerInfo>> {
    let rodata = mgk.sections.iter()
        .find(|s| s.name == ".rodata")
        .ok_or_else(|| anyhow::anyhow!("No .rodata section found"))?;

    let mut layers = Vec::new();
    let data = &rodata.data;

    let mut i = 0;
    while i < data.len() {
        // Look for "layer_" prefix
        if data[i..].starts_with(b"layer_") {
            if let Some(end) = data[i..].iter().position(|&b| b == 0) {
                let name = String::from_utf8_lossy(&data[i..i + end]).to_string();

                // Parse layer ID from name (e.g., "layer_80_QuantizeBatchNorm")
                let layer_id = parse_layer_id(&name);
                let layer_type = parse_layer_type(&name);

                layers.push(LayerInfo {
                    name: name.clone(),
                    layer_type,
                    layer_id,
                    offset: i,
                });

                i += end + 1;
                continue;
            }
        }
        i += 1;
    }

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

    Ok(ModelMetadata {
        tensors,
        layers,
        op_paths,
        inputs,
        outputs,
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
