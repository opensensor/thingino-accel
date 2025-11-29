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
}
