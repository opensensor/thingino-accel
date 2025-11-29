//! Layer configuration parser for MGK files
//!
//! Extracts detailed layer configurations by correlating:
//! - Layer names from rodata (layer_XX_QuantizeYYY)
//! - Tensor names and their formats/dtypes
//! - Operation paths (Op/kernel/params)
//! - Weight data mappings

use crate::rodata_parser::{ModelMetadata, TensorInfo, OpPathInfo};
use crate::weight_extractor::WeightHeader;
use serde::{Deserialize, Serialize};

/// Detailed layer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerConfig {
    pub name: String,
    pub layer_type: String,
    pub layer_id: Option<u32>,
    pub op_path: Option<String>,
    pub kernel_name: Option<String>,
    
    // Input/output tensor info
    pub input_tensors: Vec<TensorRef>,
    pub output_tensors: Vec<TensorRef>,
    
    // Quantization info
    pub quantization: Option<QuantizationInfo>,
    
    // Weight info (if applicable)
    pub weights: Option<WeightRef>,
}

/// Reference to a tensor with shape/format info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorRef {
    pub name: String,
    pub format: Option<String>,
    pub dtype: Option<String>,
}

/// Quantization parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationInfo {
    pub input_dtype: String,
    pub output_dtype: String,
    pub is_quantized: bool,
}

/// Reference to weight data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightRef {
    pub offset: u64,
    pub size: u64,
}

/// Model configuration with all layers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub layers: Vec<LayerConfig>,
    pub input_tensors: Vec<TensorRef>,
    pub output_tensors: Vec<TensorRef>,
    pub total_weight_bytes: u64,
}

/// Extract layer configurations from metadata
pub fn extract_layer_configs(
    metadata: &ModelMetadata,
    weight_header: Option<&WeightHeader>,
) -> ModelConfig {
    let mut configs = Vec::new();
    
    // Create a map of tensor names to their info
    let tensor_map: std::collections::HashMap<&str, &TensorInfo> = 
        metadata.tensors.iter().map(|t| (t.name.as_str(), t)).collect();
    
    // Create a map of op types to their paths
    let op_path_map: std::collections::HashMap<&str, &OpPathInfo> =
        metadata.op_paths.iter().map(|p| (p.op_type.as_str(), p)).collect();
    
    // Process each layer
    for layer in &metadata.layers {
        let mut config = LayerConfig {
            name: layer.name.clone(),
            layer_type: layer.layer_type.clone(),
            layer_id: layer.layer_id,
            op_path: None,
            kernel_name: None,
            input_tensors: Vec::new(),
            output_tensors: Vec::new(),
            quantization: None,
            weights: None,
        };
        
        // Find matching op path
        if let Some(op_path) = op_path_map.get(layer.layer_type.as_str()) {
            config.op_path = Some(format!("{}/{}", op_path.op_type, op_path.kernel_name));
            config.kernel_name = Some(op_path.kernel_name.clone());
        }
        
        // Determine quantization from layer name
        if layer.name.contains("Quantize") {
            config.quantization = Some(QuantizationInfo {
                input_dtype: "FP32".to_string(),
                output_dtype: if layer.name.contains("ubit8") { "UINT8" } else { "INT8" }.to_string(),
                is_quantized: true,
            });
        }
        
        // Find output tensor (usually matches layer name pattern)
        let output_name = &layer.name;
        config.output_tensors.push(TensorRef {
            name: output_name.clone(),
            format: tensor_map.get(output_name.as_str()).and_then(|t| t.data_format.clone()),
            dtype: tensor_map.get(output_name.as_str()).and_then(|t| t.data_type.clone()),
        });
        
        configs.push(config);
    }
    
    // Extract model inputs/outputs
    let input_tensors: Vec<TensorRef> = metadata.inputs.iter().map(|name| {
        TensorRef {
            name: name.clone(),
            format: tensor_map.get(name.as_str()).and_then(|t| t.data_format.clone()),
            dtype: tensor_map.get(name.as_str()).and_then(|t| t.data_type.clone()),
        }
    }).collect();
    
    let output_tensors: Vec<TensorRef> = metadata.outputs.iter().map(|name| {
        TensorRef {
            name: name.clone(),
            format: tensor_map.get(name.as_str()).and_then(|t| t.data_format.clone()),
            dtype: tensor_map.get(name.as_str()).and_then(|t| t.data_type.clone()),
        }
    }).collect();
    
    let total_weight_bytes = weight_header.map(|h| h.weights_size).unwrap_or(0);
    
    ModelConfig {
        layers: configs,
        input_tensors,
        output_tensors,
        total_weight_bytes,
    }
}

