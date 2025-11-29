//! Layer decoder for MGK files
//!
//! Decodes layer information from the compiled MGK model by analyzing
//! symbols and the data.rel.ro section.

use crate::types::{DetectedLayer, LayerType, MgkFile, SymbolType};
use anyhow::Result;
use std::collections::HashMap;

/// Layer operation type constants (from reverse engineering)
#[allow(dead_code)]
mod op_types {
    pub const CONV: i32 = 1;
    pub const POOL: i32 = 2;
    pub const ADD: i32 = 3;
    pub const CONCAT: i32 = 4;
    pub const RESHAPE: i32 = 5;
    pub const PERMUTE: i32 = 6;
    pub const GRU: i32 = 7;
    pub const NORMALIZE: i32 = 8;
    pub const UPSAMPLE: i32 = 9;
    pub const SLICE: i32 = 10;
    pub const FORMAT_CONVERT: i32 = 11;
    pub const DEQUANTIZE: i32 = 12;
    pub const GENERATE_BOX: i32 = 13;
}

/// Detect layers from the MGK file structure
pub fn detect_layers(mgk: &MgkFile) -> Result<Vec<DetectedLayer>> {
    let mut layers = Vec::new();

    // Build a map of layer types from symbols
    let layer_type_map = build_layer_type_map(mgk);

    // Find param_init functions which indicate layer presence
    // These are named like: gru_param_init, conv2d_int8_param_init, etc.
    let param_init_symbols: Vec<_> = mgk.symbols.iter()
        .filter(|s| s.demangled.contains("param_init") && s.symbol_type == SymbolType::Function)
        .collect();

    // Each param_init function corresponds to a layer type
    for (idx, sym) in param_init_symbols.iter().enumerate() {
        let layer_type = detect_layer_type_from_param_init(&sym.demangled);

        layers.push(DetectedLayer {
            id: idx as u16,
            layer_type: format!("{:?}", layer_type),
            op_type: layer_type_to_op_type(layer_type),
            param_index: sym.address,
            flags: 0,
            input_tensors: Vec::new(),
            output_tensors: Vec::new(),
            quant_params: None,
            weight_offset: None,
            weight_size: None,
            is_fused: false,
        });
    }

    // If no param_init found, try to detect from LayerParam type symbols
    if layers.is_empty() {
        layers = detect_from_layer_params(mgk, &layer_type_map)?;
    }

    Ok(layers)
}

/// Build a map of layer types from symbols
fn build_layer_type_map(mgk: &MgkFile) -> HashMap<String, LayerType> {
    let mut map = HashMap::new();
    
    for sym in &mgk.symbols {
        if sym.demangled.contains("LayerParam") {
            let layer_type = LayerType::from_symbol(&sym.demangled);
            if layer_type != LayerType::Unknown {
                // Extract the layer name
                let name = extract_layer_name(&sym.demangled);
                map.insert(name, layer_type);
            }
        }
    }
    
    map
}

/// Extract layer name from a demangled symbol
fn extract_layer_name(symbol: &str) -> String {
    // Example: "magik::venus::layer::ConvLayerParam" -> "Conv"
    if let Some(idx) = symbol.rfind("::") {
        let name = &symbol[idx + 2..];
        name.replace("LayerParam", "")
            .replace("LayerRes", "")
            .replace("Param", "")
    } else {
        symbol.to_string()
    }
}

/// Detect layer type from a param_init function name
/// Examples:
///   - gru_param_init -> Gru
///   - conv2d_int8_param_init -> Conv
///   - maxpool_int8_param_init -> Pool
///   - avgpool_int8_param_init -> Pool
///   - add_int8_param_init -> Add
///   - concat_int8_param_init -> Concat
///   - reshape_param_init -> Reshape
///   - permute_param_init -> Permute
///   - upsample_int8_param_init -> Upsample
///   - slice_param_init -> Slice
///   - format_convert_param_init -> FormatConvert
///   - dequantize_param_init -> DeQuantize
///   - generate_box_param_init -> GenerateBox
///   - unsqueeze_int8_param_init -> SqueezeUnsqueeze
fn detect_layer_type_from_param_init(symbol: &str) -> LayerType {
    let lower = symbol.to_lowercase();

    if lower.contains("conv2d") || lower.contains("conv_") {
        LayerType::Conv
    } else if lower.contains("maxpool") || lower.contains("avgpool") || lower.contains("pool") {
        LayerType::Pool
    } else if lower.contains("add_int") || (lower.contains("add") && lower.contains("param_init")) {
        LayerType::Add
    } else if lower.contains("concat") {
        LayerType::Concat
    } else if lower.contains("reshape") {
        LayerType::Reshape
    } else if lower.contains("permute") {
        LayerType::Permute
    } else if lower.contains("gru") {
        LayerType::Gru
    } else if lower.contains("normalize") {
        LayerType::Normalize
    } else if lower.contains("upsample") {
        LayerType::Upsample
    } else if lower.contains("slice") {
        LayerType::Slice
    } else if lower.contains("format_convert") {
        LayerType::FormatConvert
    } else if lower.contains("dequantize") {
        LayerType::DeQuantize
    } else if lower.contains("generate_box") {
        LayerType::GenerateBox
    } else if lower.contains("squeeze") || lower.contains("unsqueeze") {
        LayerType::SqueezeUnsqueeze
    } else {
        LayerType::Unknown
    }
}

/// Detect layer type from a symbol name (for LayerParam symbols)
#[allow(dead_code)]
fn detect_layer_type_from_symbol(symbol: &str) -> LayerType {
    LayerType::from_symbol(symbol)
}

/// Convert LayerType to op_type integer
fn layer_type_to_op_type(layer_type: LayerType) -> i32 {
    match layer_type {
        LayerType::Conv => op_types::CONV,
        LayerType::Pool => op_types::POOL,
        LayerType::Add => op_types::ADD,
        LayerType::Concat => op_types::CONCAT,
        LayerType::Reshape => op_types::RESHAPE,
        LayerType::Permute => op_types::PERMUTE,
        LayerType::Gru => op_types::GRU,
        LayerType::Normalize => op_types::NORMALIZE,
        LayerType::Upsample => op_types::UPSAMPLE,
        LayerType::Slice => op_types::SLICE,
        LayerType::FormatConvert => op_types::FORMAT_CONVERT,
        LayerType::DeQuantize => op_types::DEQUANTIZE,
        LayerType::GenerateBox => op_types::GENERATE_BOX,
        LayerType::SqueezeUnsqueeze => 14,
        LayerType::Unknown => 0,
    }
}

/// Detect layers from LayerParam symbols in data.rel.ro
fn detect_from_layer_params(
    mgk: &MgkFile, 
    _layer_type_map: &HashMap<String, LayerType>
) -> Result<Vec<DetectedLayer>> {
    let mut layers = Vec::new();
    
    // Find unique layer type symbols
    let layer_param_symbols: Vec<_> = mgk.symbols.iter()
        .filter(|s| s.demangled.contains("LayerParam") && 
                   !s.demangled.contains("Sp_counted"))
        .collect();
    
    // Create layers from each unique LayerParam type
    let mut seen_types = std::collections::HashSet::new();
    for (idx, sym) in layer_param_symbols.iter().enumerate() {
        let layer_type = LayerType::from_symbol(&sym.demangled);
        let type_name = format!("{:?}", layer_type);
        
        if seen_types.insert(type_name.clone()) {
            layers.push(DetectedLayer {
                id: idx as u16,
                layer_type: type_name,
                op_type: layer_type_to_op_type(layer_type),
                param_index: sym.address,
                flags: 0,
                input_tensors: Vec::new(),
                output_tensors: Vec::new(),
                quant_params: None,
                weight_offset: None,
                weight_size: None,
                is_fused: false,
            });
        }
    }
    
    Ok(layers)
}

