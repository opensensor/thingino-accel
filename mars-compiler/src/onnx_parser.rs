//! ONNX Parser for Mars Compiler
//!
//! Parses ONNX protobuf models and extracts layer/tensor information
//! for conversion to Mars format.

use anyhow::{Context, Result};
use prost::Message;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

// ============================================================================
// ONNX Protobuf Structures (matching onnx.proto3)
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

impl TensorDataType {
    pub fn from_i32(v: i32) -> Self {
        match v {
            1 => Self::Float,
            2 => Self::Uint8,
            3 => Self::Int8,
            6 => Self::Int32,
            7 => Self::Int64,
            10 => Self::Float16,
            11 => Self::Double,
            _ => Self::Undefined,
        }
    }
    
    pub fn element_size(&self) -> usize {
        match self {
            Self::Float | Self::Int32 | Self::Uint32 => 4,
            Self::Int64 | Self::Uint64 | Self::Double => 8,
            Self::Int16 | Self::Uint16 | Self::Float16 => 2,
            Self::Int8 | Self::Uint8 | Self::Bool => 1,
            _ => 4,
        }
    }
}

/// Attribute type enum
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum AttributeType {
    Undefined = 0,
    Float = 1,
    Int = 2,
    String = 3,
    Tensor = 4,
    Graph = 5,
    Floats = 6,
    Ints = 7,
    Strings = 8,
    Tensors = 9,
    Graphs = 10,
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

/// ONNX TypeProto
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
}

// ============================================================================
// High-level parsed structures for Mars conversion
// ============================================================================

/// Parsed ONNX tensor info
#[derive(Debug, Clone)]
pub struct OnnxTensor {
    pub name: String,
    pub dims: Vec<i64>,
    pub data_type: TensorDataType,
    pub data: Vec<u8>,
    pub float_data: Vec<f32>,  // For scalars stored in float_data field
}

impl OnnxTensor {
    pub fn from_proto(proto: &TensorProto) -> Self {
        let data_type = TensorDataType::from_i32(proto.data_type);

        // Debug: warn if data_type is undefined
        if data_type == TensorDataType::Undefined && proto.data_type != 0 {
            eprintln!("Warning: Unknown ONNX data_type {} for tensor {}",
                     proto.data_type, proto.name);
        }

        // Get raw data - either from raw_data or typed arrays
        let data = if !proto.raw_data.is_empty() {
            proto.raw_data.clone()
        } else if !proto.float_data.is_empty() {
            proto.float_data.iter()
                .flat_map(|f| f.to_le_bytes())
                .collect()
        } else if !proto.int64_data.is_empty() {
            proto.int64_data.iter()
                .flat_map(|i| i.to_le_bytes())
                .collect()
        } else if !proto.int32_data.is_empty() {
            proto.int32_data.iter()
                .flat_map(|i| i.to_le_bytes())
                .collect()
        } else {
            Vec::new()
        };

        // Keep float_data for scalar scales (QDQ models store scales in float_data)
        let float_data = proto.float_data.clone();

        Self {
            name: proto.name.clone(),
            dims: proto.dims.clone(),
            data_type,
            data,
            float_data,
        }
    }

    pub fn num_elements(&self) -> usize {
        self.dims.iter().map(|&d| d as usize).product()
    }
}

/// Parsed ONNX node/operation
#[derive(Debug, Clone)]
pub struct OnnxNode {
    pub name: String,
    pub op_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attributes: HashMap<String, OnnxAttribute>,
}

/// ONNX attribute value
#[derive(Debug, Clone)]
pub enum OnnxAttribute {
    Int(i64),
    Float(f32),
    String(String),
    Ints(Vec<i64>),
    Floats(Vec<f32>),
    Tensor(OnnxTensor),
}

impl OnnxNode {
    pub fn from_proto(proto: &NodeProto) -> Self {
        let mut attributes = HashMap::new();

        for attr in &proto.attribute {
            let value = match attr.r#type {
                1 => attr.f.map(OnnxAttribute::Float),
                2 => attr.i.map(OnnxAttribute::Int),
                3 => attr.s.as_ref().map(|s| {
                    OnnxAttribute::String(String::from_utf8_lossy(s).to_string())
                }),
                4 => attr.t.as_ref().map(|t| {
                    OnnxAttribute::Tensor(OnnxTensor::from_proto(t))
                }),
                6 => Some(OnnxAttribute::Floats(attr.floats.clone())),
                7 => Some(OnnxAttribute::Ints(attr.ints.clone())),
                _ => None,
            };

            if let Some(v) = value {
                attributes.insert(attr.name.clone(), v);
            }
        }

        Self {
            name: proto.name.clone(),
            op_type: proto.op_type.clone(),
            inputs: proto.input.clone(),
            outputs: proto.output.clone(),
            attributes,
        }
    }

    pub fn get_int(&self, name: &str) -> Option<i64> {
        match self.attributes.get(name) {
            Some(OnnxAttribute::Int(v)) => Some(*v),
            _ => None,
        }
    }

    pub fn get_ints(&self, name: &str) -> Option<&Vec<i64>> {
        match self.attributes.get(name) {
            Some(OnnxAttribute::Ints(v)) => Some(v),
            _ => None,
        }
    }

    pub fn get_float(&self, name: &str) -> Option<f32> {
        match self.attributes.get(name) {
            Some(OnnxAttribute::Float(v)) => Some(*v),
            _ => None,
        }
    }

    pub fn get_string(&self, name: &str) -> Option<&str> {
        match self.attributes.get(name) {
            Some(OnnxAttribute::String(v)) => Some(v),
            _ => None,
        }
    }
}

/// Input/output shape info
#[derive(Debug, Clone)]
pub struct TensorShape {
    pub name: String,
    pub dims: Vec<i64>,
    pub data_type: TensorDataType,
}

impl TensorShape {
    pub fn from_value_info(vi: &ValueInfoProto) -> Option<Self> {
        let tensor_type = vi.r#type.as_ref()?.tensor_type.as_ref()?;
        let shape = tensor_type.shape.as_ref()?;

        let dims: Vec<i64> = shape.dim.iter()
            .map(|d| d.dim_value.unwrap_or(-1))
            .collect();

        Some(Self {
            name: vi.name.clone(),
            dims,
            data_type: TensorDataType::from_i32(tensor_type.elem_type),
        })
    }
}

/// Complete parsed ONNX model
#[derive(Debug)]
pub struct OnnxModel {
    pub name: String,
    pub producer: String,
    pub opset_version: i64,
    pub inputs: Vec<TensorShape>,
    pub outputs: Vec<TensorShape>,
    pub nodes: Vec<OnnxNode>,
    pub initializers: HashMap<String, OnnxTensor>,
    pub shape_info: HashMap<String, Vec<i64>>,  // name -> shape for all tensors
}

impl OnnxModel {
    /// Load an ONNX model from a file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let data = fs::read(path.as_ref())
            .context("Failed to read ONNX file")?;

        let proto = ModelProto::decode(data.as_slice())
            .context("Failed to parse ONNX protobuf")?;

        Self::from_proto(proto)
    }

    fn from_proto(proto: ModelProto) -> Result<Self> {
        let graph = proto.graph
            .ok_or_else(|| anyhow::anyhow!("ONNX model has no graph"))?;

        // Parse inputs (filter out initializers)
        let initializer_names: std::collections::HashSet<_> =
            graph.initializer.iter().map(|i| &i.name).collect();

        let inputs: Vec<TensorShape> = graph.input.iter()
            .filter(|vi| !initializer_names.contains(&vi.name))
            .filter_map(|vi| TensorShape::from_value_info(vi))
            .collect();

        // Parse outputs
        let outputs: Vec<TensorShape> = graph.output.iter()
            .filter_map(|vi| TensorShape::from_value_info(vi))
            .collect();

        // Parse nodes
        let nodes: Vec<OnnxNode> = graph.node.iter()
            .map(OnnxNode::from_proto)
            .collect();

        // Parse initializers (weights, biases, constants)
        let initializers: HashMap<String, OnnxTensor> = graph.initializer.iter()
            .map(|t| (t.name.clone(), OnnxTensor::from_proto(t)))
            .collect();

        // Get opset version
        let opset_version = proto.opset_import.first()
            .map(|op| op.version)
            .unwrap_or(11);

        // Build shape_info from all sources: inputs, outputs, value_info, initializers
        let mut shape_info: HashMap<String, Vec<i64>> = HashMap::new();

        // From inputs
        for vi in &graph.input {
            if let Some(shape) = TensorShape::from_value_info(vi) {
                shape_info.insert(shape.name.clone(), shape.dims.clone());
            }
        }

        // From outputs
        for vi in &graph.output {
            if let Some(shape) = TensorShape::from_value_info(vi) {
                shape_info.insert(shape.name.clone(), shape.dims.clone());
            }
        }

        // From value_info (intermediate tensors)
        for vi in &graph.value_info {
            if let Some(shape) = TensorShape::from_value_info(vi) {
                shape_info.insert(shape.name.clone(), shape.dims.clone());
            }
        }

        // From initializers
        for init in &graph.initializer {
            shape_info.insert(init.name.clone(), init.dims.clone());
        }

        Ok(Self {
            name: graph.name,
            producer: proto.producer_name,
            opset_version,
            inputs,
            outputs,
            nodes,
            initializers,
            shape_info,
        })
    }

    /// Print a summary of the model
    pub fn print_summary(&self) {
        println!("ONNX Model: {}", self.name);
        println!("Producer: {}", self.producer);
        println!("Opset: {}", self.opset_version);
        println!("\nInputs:");
        for input in &self.inputs {
            println!("  {} {:?} ({:?})", input.name, input.dims, input.data_type);
        }
        println!("\nOutputs:");
        for output in &self.outputs {
            println!("  {} {:?} ({:?})", output.name, output.dims, output.data_type);
        }
        println!("\nNodes: {}", self.nodes.len());

        // Count op types
        let mut op_counts: HashMap<&str, usize> = HashMap::new();
        for node in &self.nodes {
            *op_counts.entry(&node.op_type).or_insert(0) += 1;
        }
        let mut ops: Vec<_> = op_counts.into_iter().collect();
        ops.sort_by(|a, b| b.1.cmp(&a.1));
        for (op, count) in ops {
            println!("  {}: {}", op, count);
        }

        println!("\nInitializers: {}", self.initializers.len());
        let total_bytes: usize = self.initializers.values()
            .map(|t| t.data.len())
            .sum();
        println!("  Total weight bytes: {} ({:.2} MB)",
            total_bytes, total_bytes as f64 / 1024.0 / 1024.0);
    }
}

