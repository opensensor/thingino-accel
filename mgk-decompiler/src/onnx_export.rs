//! ONNX Export Module
//!
//! Exports MGK model structure to ONNX format.
//! Uses manual protobuf encoding to avoid requiring protoc.

use anyhow::{Result, Context};
use prost::Message;
use std::fs::File;
use std::io::Write;
use std::path::Path;

use crate::rodata_parser::{LayerInfo, ModelMetadata};

/// ONNX IR version (version 9 = opset 19)
const ONNX_IR_VERSION: i64 = 9;
/// ONNX opset version
const ONNX_OPSET_VERSION: i64 = 13;

// ============================================================================
// ONNX Protobuf Structures (manually defined to match onnx.proto3)
// ============================================================================

/// ONNX data types
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum TensorDataType {
    Undefined = 0,
    Float = 1,
    Uint8 = 2,
    Int8 = 3,
    Uint16 = 4,
    Int16 = 5,
    Int32 = 6,
    Int64 = 7,
    String = 8,
    Bool = 9,
    Float16 = 10,
    Double = 11,
    Uint32 = 12,
    Uint64 = 13,
}

/// ONNX AttributeProto
#[derive(Clone, Message)]
pub struct AttributeProto {
    #[prost(string, tag = "1")]
    pub name: String,
    #[prost(float, optional, tag = "2")]
    pub f: Option<f32>,
    #[prost(int64, optional, tag = "3")]
    pub i: Option<i64>,
    #[prost(bytes = "vec", optional, tag = "4")]
    pub s: Option<Vec<u8>>,
    #[prost(message, optional, tag = "5")]
    pub t: Option<TensorProto>,
    #[prost(float, repeated, tag = "7")]
    pub floats: Vec<f32>,
    #[prost(int64, repeated, tag = "8")]
    pub ints: Vec<i64>,
    #[prost(bytes = "vec", repeated, tag = "9")]
    pub strings: Vec<Vec<u8>>,
    #[prost(int32, tag = "20")]
    pub r#type: i32,
}

/// Attribute type enum
#[derive(Clone, Copy, Debug)]
#[repr(i32)]
pub enum AttributeType {
    Undefined = 0,
    Float = 1,
    Int = 2,
    String = 3,
    Tensor = 4,
    Floats = 6,
    Ints = 7,
    Strings = 8,
}

/// ONNX TensorProto
#[derive(Clone, Message)]
pub struct TensorProto {
    #[prost(int64, repeated, tag = "1")]
    pub dims: Vec<i64>,
    #[prost(int32, tag = "2")]
    pub data_type: i32,
    #[prost(string, tag = "8")]
    pub name: String,
    #[prost(bytes = "vec", tag = "9")]
    pub raw_data: Vec<u8>,
    #[prost(float, repeated, tag = "4")]
    pub float_data: Vec<f32>,
    #[prost(int32, repeated, tag = "5")]
    pub int32_data: Vec<i32>,
    #[prost(int64, repeated, tag = "7")]
    pub int64_data: Vec<i64>,
}

/// ONNX TensorShapeProto.Dimension
#[derive(Clone, Message)]
pub struct Dimension {
    #[prost(int64, optional, tag = "1")]
    pub dim_value: Option<i64>,
    #[prost(string, optional, tag = "2")]
    pub dim_param: Option<String>,
}

/// ONNX TensorShapeProto
#[derive(Clone, Message)]
pub struct TensorShapeProto {
    #[prost(message, repeated, tag = "1")]
    pub dim: Vec<Dimension>,
}

/// ONNX TypeProto.Tensor
#[derive(Clone, Message)]
pub struct TypeProtoTensor {
    #[prost(int32, tag = "1")]
    pub elem_type: i32,
    #[prost(message, optional, tag = "2")]
    pub shape: Option<TensorShapeProto>,
}

/// ONNX TypeProto (simplified - only tensor type)
#[derive(Clone, Message)]
pub struct TypeProto {
    #[prost(message, optional, tag = "1")]
    pub tensor_type: Option<TypeProtoTensor>,
}

/// ONNX ValueInfoProto
#[derive(Clone, Message)]
pub struct ValueInfoProto {
    #[prost(string, tag = "1")]
    pub name: String,
    #[prost(message, optional, tag = "2")]
    pub r#type: Option<TypeProto>,
    #[prost(string, tag = "3")]
    pub doc_string: String,
}

/// ONNX NodeProto
#[derive(Clone, Message)]
pub struct NodeProto {
    #[prost(string, repeated, tag = "1")]
    pub input: Vec<String>,
    #[prost(string, repeated, tag = "2")]
    pub output: Vec<String>,
    #[prost(string, tag = "3")]
    pub name: String,
    #[prost(string, tag = "4")]
    pub op_type: String,
    #[prost(string, tag = "7")]
    pub domain: String,
    #[prost(message, repeated, tag = "5")]
    pub attribute: Vec<AttributeProto>,
    #[prost(string, tag = "6")]
    pub doc_string: String,
}

/// ONNX GraphProto
#[derive(Clone, Message)]
pub struct GraphProto {
    #[prost(message, repeated, tag = "1")]
    pub node: Vec<NodeProto>,
    #[prost(string, tag = "2")]
    pub name: String,
    #[prost(message, repeated, tag = "5")]
    pub initializer: Vec<TensorProto>,
    #[prost(string, tag = "10")]
    pub doc_string: String,
    #[prost(message, repeated, tag = "11")]
    pub input: Vec<ValueInfoProto>,
    #[prost(message, repeated, tag = "12")]
    pub output: Vec<ValueInfoProto>,
    #[prost(message, repeated, tag = "13")]
    pub value_info: Vec<ValueInfoProto>,
}

/// ONNX OperatorSetIdProto
#[derive(Clone, Message)]
pub struct OperatorSetIdProto {
    #[prost(string, tag = "1")]
    pub domain: String,
    #[prost(int64, tag = "2")]
    pub version: i64,
}

/// ONNX StringStringEntryProto (for metadata)
#[derive(Clone, Message)]
pub struct StringStringEntryProto {
    #[prost(string, tag = "1")]
    pub key: String,
    #[prost(string, tag = "2")]
    pub value: String,
}

/// ONNX ModelProto - the root message
#[derive(Clone, Message)]
pub struct ModelProto {
    #[prost(int64, tag = "1")]
    pub ir_version: i64,
    #[prost(message, repeated, tag = "8")]
    pub opset_import: Vec<OperatorSetIdProto>,
    #[prost(string, tag = "2")]
    pub producer_name: String,
    #[prost(string, tag = "3")]
    pub producer_version: String,
    #[prost(string, tag = "4")]
    pub domain: String,
    #[prost(int64, tag = "5")]
    pub model_version: i64,
    #[prost(string, tag = "6")]
    pub doc_string: String,
    #[prost(message, optional, tag = "7")]
    pub graph: Option<GraphProto>,
    #[prost(message, repeated, tag = "14")]
    pub metadata_props: Vec<StringStringEntryProto>,
}

// ============================================================================
// Helper functions for creating ONNX structures
// ============================================================================

impl AttributeProto {
    pub fn int(name: &str, value: i64) -> Self {
        AttributeProto {
            name: name.to_string(),
            i: Some(value),
            r#type: AttributeType::Int as i32,
            ..Default::default()
        }
    }

    pub fn float(name: &str, value: f32) -> Self {
        AttributeProto {
            name: name.to_string(),
            f: Some(value),
            r#type: AttributeType::Float as i32,
            ..Default::default()
        }
    }

    pub fn ints(name: &str, values: Vec<i64>) -> Self {
        AttributeProto {
            name: name.to_string(),
            ints: values,
            r#type: AttributeType::Ints as i32,
            ..Default::default()
        }
    }

    pub fn floats(name: &str, values: Vec<f32>) -> Self {
        AttributeProto {
            name: name.to_string(),
            floats: values,
            r#type: AttributeType::Floats as i32,
            ..Default::default()
        }
    }

    pub fn string(name: &str, value: &str) -> Self {
        AttributeProto {
            name: name.to_string(),
            s: Some(value.as_bytes().to_vec()),
            r#type: AttributeType::String as i32,
            ..Default::default()
        }
    }
}

fn make_tensor_shape(dims: &[i64]) -> TensorShapeProto {
    TensorShapeProto {
        dim: dims.iter().map(|&d| Dimension {
            dim_value: Some(d),
            dim_param: None,
        }).collect(),
    }
}

pub fn make_value_info(name: &str, elem_type: TensorDataType, dims: &[i64]) -> ValueInfoProto {
    ValueInfoProto {
        name: name.to_string(),
        r#type: Some(TypeProto {
            tensor_type: Some(TypeProtoTensor {
                elem_type: elem_type as i32,
                shape: Some(make_tensor_shape(dims)),
            }),
        }),
        doc_string: String::new(),
    }
}

pub fn make_initializer(name: &str, data_type: TensorDataType, dims: &[i64], data: &[u8]) -> TensorProto {
    TensorProto {
        name: name.to_string(),
        data_type: data_type as i32,
        dims: dims.to_vec(),
        raw_data: data.to_vec(),
        ..Default::default()
    }
}

// ============================================================================
// MGK to ONNX Layer Mapping
// ============================================================================

/// Maps MGK layer type to ONNX operator
fn map_layer_to_onnx_op(layer_type: &str) -> &'static str {
    match layer_type {
        "Conv" | "Feature" => "Conv",
        "BatchNorm" => "BatchNormalization",
        "GRU" => "GRU",
        "Concat" => "Concat",
        "Pool" => "MaxPool",
        "Add" => "Add",
        "Upsample" => "Upsample",
        "Reshape" => "Reshape",
        "Permute" => "Transpose",
        "Slice" => "Slice",
        "Sigmoid" => "Sigmoid",
        "ReLU" => "Relu",
        "Output" | "DeQuantize" => "Identity",
        "QuantizedLayer" => "Conv",  // Most quantized layers are conv
        _ => "Identity",
    }
}

/// Build an ONNX node from a layer
/// Returns (node, output_name, additional_initializers)
fn build_onnx_node(
    layer: &LayerInfo,
    idx: usize,
    prev_output: &str,
    channels: i64,
) -> (NodeProto, String, Vec<TensorProto>) {
    let layer_type = &layer.layer_type;
    let op_type = map_layer_to_onnx_op(layer_type);
    let output_name = format!("layer_{}_out", idx);
    let mut initializers = Vec::new();

    let mut node = NodeProto {
        name: layer.name.clone(),
        op_type: op_type.to_string(),
        input: vec![prev_output.to_string()],
        output: vec![output_name.clone()],
        domain: String::new(),
        attribute: Vec::new(),
        doc_string: String::new(),
    };

    // Add operator-specific attributes and inputs
    match op_type {
        "Conv" => {
            // Default Conv attributes (3x3 kernel, padding=1, stride=1)
            node.attribute.push(AttributeProto::ints("kernel_shape", vec![3, 3]));
            node.attribute.push(AttributeProto::ints("pads", vec![1, 1, 1, 1]));
            node.attribute.push(AttributeProto::ints("strides", vec![1, 1]));
            node.attribute.push(AttributeProto::int("group", 1));

            // Add weight and bias inputs
            let weight_name = format!("{}_weight", layer.name);
            let bias_name = format!("{}_bias", layer.name);
            node.input.push(weight_name.clone());
            node.input.push(bias_name.clone());

            // Create placeholder weights (3x3 conv)
            let weight_data = vec![0u8; (channels * channels * 9) as usize * 4];
            initializers.push(make_initializer(
                &weight_name,
                TensorDataType::Float,
                &[channels, channels, 3, 3],
                &weight_data,
            ));

            // Create placeholder bias
            let bias_data = vec![0u8; channels as usize * 4];
            initializers.push(make_initializer(
                &bias_name,
                TensorDataType::Float,
                &[channels],
                &bias_data,
            ));
        }
        "MaxPool" => {
            node.attribute.push(AttributeProto::ints("kernel_shape", vec![2, 2]));
            node.attribute.push(AttributeProto::ints("strides", vec![2, 2]));
        }
        "BatchNormalization" => {
            node.attribute.push(AttributeProto::float("epsilon", 1e-5));
            node.attribute.push(AttributeProto::float("momentum", 0.9));

            // BatchNorm requires: X, scale, B, mean, var
            let scale_name = format!("{}_scale", layer.name);
            let bias_name = format!("{}_bias", layer.name);
            let mean_name = format!("{}_mean", layer.name);
            let var_name = format!("{}_var", layer.name);

            node.input.push(scale_name.clone());
            node.input.push(bias_name.clone());
            node.input.push(mean_name.clone());
            node.input.push(var_name.clone());

            // Create placeholder tensors (all ones for scale, zeros for others)
            let ones_data: Vec<u8> = (0..channels).flat_map(|_| 1.0f32.to_le_bytes()).collect();
            let zeros_data: Vec<u8> = vec![0u8; channels as usize * 4];

            initializers.push(make_initializer(&scale_name, TensorDataType::Float, &[channels], &ones_data));
            initializers.push(make_initializer(&bias_name, TensorDataType::Float, &[channels], &zeros_data));
            initializers.push(make_initializer(&mean_name, TensorDataType::Float, &[channels], &zeros_data));
            initializers.push(make_initializer(&var_name, TensorDataType::Float, &[channels], &ones_data));
        }
        "Concat" => {
            node.attribute.push(AttributeProto::int("axis", 1)); // Channel axis
        }
        "GRU" => {
            node.attribute.push(AttributeProto::string("direction", "forward"));
            node.attribute.push(AttributeProto::int("hidden_size", 64));

            // GRU requires: X, W, R, B (optional)
            let w_name = format!("{}_W", layer.name);
            let r_name = format!("{}_R", layer.name);

            node.input.push(w_name.clone());
            node.input.push(r_name.clone());

            // Placeholder GRU weights (3 gates * hidden_size)
            let hidden_size = 64i64;
            let input_size = channels;
            let w_data = vec![0u8; (3 * hidden_size * input_size) as usize * 4];
            let r_data = vec![0u8; (3 * hidden_size * hidden_size) as usize * 4];

            initializers.push(make_initializer(&w_name, TensorDataType::Float, &[1, 3 * hidden_size, input_size], &w_data));
            initializers.push(make_initializer(&r_name, TensorDataType::Float, &[1, 3 * hidden_size, hidden_size], &r_data));
        }
        "Upsample" => {
            node.attribute.push(AttributeProto::string("mode", "nearest"));
        }
        "Transpose" => {
            node.attribute.push(AttributeProto::ints("perm", vec![0, 2, 3, 1]));
        }
        _ => {}
    }

    (node, output_name, initializers)
}

/// ONNX Exporter for MGK models
pub struct OnnxExporter {
    model_name: String,
    metadata: ModelMetadata,
    #[allow(dead_code)]
    weight_data: Vec<u8>,
}

impl OnnxExporter {
    pub fn new(
        model_name: String,
        metadata: ModelMetadata,
        weight_data: Vec<u8>,
    ) -> Self {
        Self {
            model_name,
            metadata,
            weight_data,
        }
    }

    /// Export to ONNX format
    pub fn export<P: AsRef<Path>>(&self, output_path: P) -> Result<()> {
        let model = self.build_model()?;

        // Serialize to protobuf
        let mut buf = Vec::new();
        model.encode(&mut buf).context("Failed to encode ONNX model")?;

        // Write to file
        let mut file = File::create(output_path.as_ref())
            .context("Failed to create output file")?;
        file.write_all(&buf).context("Failed to write ONNX model")?;

        Ok(())
    }

    fn build_model(&self) -> Result<ModelProto> {
        let graph = self.build_graph()?;

        Ok(ModelProto {
            ir_version: ONNX_IR_VERSION,
            opset_import: vec![OperatorSetIdProto {
                domain: String::new(),  // Default ONNX domain
                version: ONNX_OPSET_VERSION,
            }],
            producer_name: "mgk-decompiler".to_string(),
            producer_version: env!("CARGO_PKG_VERSION").to_string(),
            domain: String::new(),
            model_version: 1,
            doc_string: format!("Decompiled from MGK model: {}", self.model_name),
            graph: Some(graph),
            metadata_props: vec![
                StringStringEntryProto {
                    key: "source".to_string(),
                    value: "Ingenic Magik T41/T31 NPU".to_string(),
                },
                StringStringEntryProto {
                    key: "original_model".to_string(),
                    value: self.model_name.clone(),
                },
            ],
        })
    }

    fn build_graph(&self) -> Result<GraphProto> {
        let mut nodes = Vec::new();
        let mut initializers = Vec::new();

        // Determine input shape from metadata
        let input_channels = 32i64;  // Default for T41 models
        let input_height = 64i64;
        let input_width = 64i64;

        // Create input
        let input_name = if !self.metadata.input_names.is_empty() {
            self.metadata.input_names[0].clone()
        } else {
            "input".to_string()
        };

        let input = make_value_info(
            &input_name,
            TensorDataType::Float,
            &[-1, input_channels, input_height, input_width],  // -1 for dynamic batch
        );

        // Build layer nodes from metadata.layers
        let mut prev_output = input_name.clone();
        let in_channels = input_channels;

        for (idx, layer) in self.metadata.layers.iter().enumerate() {
            let (node, output_name, layer_initializers) = build_onnx_node(layer, idx, &prev_output, in_channels);

            // Add node and its initializers
            nodes.push(node);
            initializers.extend(layer_initializers);

            prev_output = output_name;
        }

        // Create output
        let output_name = if !self.metadata.output_names.is_empty() {
            self.metadata.output_names[0].clone()
        } else {
            prev_output.clone()
        };

        // Add final identity to rename output if needed
        if output_name != prev_output {
            nodes.push(NodeProto {
                name: "output_rename".to_string(),
                op_type: "Identity".to_string(),
                input: vec![prev_output],
                output: vec![output_name.clone()],
                domain: String::new(),
                attribute: Vec::new(),
                doc_string: String::new(),
            });
        }

        let output = make_value_info(
            &output_name,
            TensorDataType::Float,
            &[-1, in_channels, input_height, input_width],
        );

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
}

/// High-level export function
pub fn export_to_onnx<P: AsRef<Path>>(
    model_name: &str,
    metadata: &ModelMetadata,
    weight_data: &[u8],
    output_path: P,
) -> Result<()> {
    let exporter = OnnxExporter::new(
        model_name.to_string(),
        metadata.clone(),
        weight_data.to_vec(),
    );

    exporter.export(output_path)
}
