//! Type definitions for MGK decompiler

use serde::{Deserialize, Serialize};

/// Represents an ELF section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Section {
    pub name: String,
    pub address: u64,
    pub offset: u64,
    pub size: u64,
    pub data: Vec<u8>,
}

/// Quantization parameters for a layer
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QuantizationParams {
    pub weight_bitwidth: u8,
    pub input_bitwidth: u8,
    pub output_bitwidth: u8,
    pub dequantize_scale: f32,
    pub offset: f32,
    pub threshold_min: f32,
    pub threshold_max: f32,
    pub fixpoint: bool,
}

impl QuantizationParams {
    pub fn new() -> Self {
        Self {
            weight_bitwidth: 8,
            input_bitwidth: 8,
            output_bitwidth: 8,
            dequantize_scale: 1.0,
            offset: 0.0,
            threshold_min: -128.0,
            threshold_max: 127.0,
            fixpoint: false,
        }
    }
}

/// Represents a symbol from the ELF file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Symbol {
    pub name: String,
    pub demangled: String,
    pub address: u64,
    pub size: u64,
    pub symbol_type: SymbolType,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum SymbolType {
    Function,
    Object,
    NoType,
    Section,
    File,
    Unknown,
}

/// Target architecture
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Architecture {
    Mips32El,
    Unknown,
}

/// Parsed MGK file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MgkFile {
    pub architecture: Architecture,
    pub entry_point: u64,
    pub sections: Vec<Section>,
    pub symbols: Vec<Symbol>,
}

/// Layer parameter record (18 bytes in binary format)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerParamRecord {
    pub op_type: i32,        // 4 bytes
    pub layer_id: u16,       // 2 bytes  
    pub param_index: u64,    // 8 bytes
    pub flags: u32,          // 4 bytes
}

/// Known layer types from symbol analysis
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum LayerType {
    Conv,
    Pool,
    Add,
    Concat,
    Reshape,
    Permute,
    Gru,
    Normalize,
    Upsample,
    Slice,
    FormatConvert,
    DeQuantize,
    GenerateBox,
    SqueezeUnsqueeze,
    Unknown,
}

impl LayerType {
    pub fn from_symbol(symbol: &str) -> Self {
        if symbol.contains("Conv") {
            LayerType::Conv
        } else if symbol.contains("Pool") {
            LayerType::Pool
        } else if symbol.contains("Add") && !symbol.contains("Addr") {
            LayerType::Add
        } else if symbol.contains("Concat") {
            LayerType::Concat
        } else if symbol.contains("Reshape") {
            LayerType::Reshape
        } else if symbol.contains("Permute") {
            LayerType::Permute
        } else if symbol.contains("Gru") {
            LayerType::Gru
        } else if symbol.contains("Normalize") {
            LayerType::Normalize
        } else if symbol.contains("Upsample") {
            LayerType::Upsample
        } else if symbol.contains("Slice") {
            LayerType::Slice
        } else if symbol.contains("FormatConvert") {
            LayerType::FormatConvert
        } else if symbol.contains("DeQuantize") {
            LayerType::DeQuantize
        } else if symbol.contains("GenerateBox") {
            LayerType::GenerateBox
        } else if symbol.contains("SqueezeUnsqueeze") {
            LayerType::SqueezeUnsqueeze
        } else {
            LayerType::Unknown
        }
    }
}

/// Detected layer in the model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedLayer {
    pub id: u16,
    pub layer_type: String,
    pub op_type: i32,
    pub param_index: u64,
    pub flags: u32,
    pub input_tensors: Vec<String>,
    pub output_tensors: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quant_params: Option<QuantizationParams>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weight_offset: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weight_size: Option<u64>,
    #[serde(default)]
    pub is_fused: bool,
}

/// Output format for JSON serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecompilerOutput {
    pub model_name: String,
    pub layers: Vec<DetectedLayer>,
    pub symbols: Vec<Symbol>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<crate::rodata_parser::ModelMetadata>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer_graph: Option<Vec<crate::rodata_parser::LayerGraphNode>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weight_info: Option<crate::weight_extractor::WeightHeader>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weight_stats: Option<crate::weight_extractor::WeightStats>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_config: Option<crate::layer_config::ModelConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer_weight_mappings: Option<Vec<crate::weight_extractor::LayerWeightMapping>>,
}
