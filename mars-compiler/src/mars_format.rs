//! Mars binary format definitions
//!
//! This matches the C structures in include/mars.h

use byteorder::{LittleEndian, WriteBytesExt};
use std::io::{self, Write};

pub const MARS_MAGIC: u32 = 0x5352414D; // "MARS" in little-endian
pub const MARS_VERSION: u32 = 1;
pub const MARS_MAX_DIMS: usize = 8;
pub const MARS_MAX_NAME: usize = 64;

// Header: 76 bytes
pub const HEADER_SIZE: usize = 76;
// Tensor: 124 bytes
pub const TENSOR_SIZE: usize = 124;
// Layer: 112 bytes
pub const LAYER_SIZE: usize = 112;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum DataType {
    Float32 = 0,
    Int32 = 1,
    Int16 = 2,
    Int8 = 3,
    Uint8 = 4,
    Uint4 = 5,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum DataFormat {
    Nhwc = 0,  // Height, Width, Channels
    Nchw = 1,  // Channels, Height, Width
    Ohwi = 2,  // Out_ch, Height, Width, In_ch (for weights)
    Oihw = 3,  // Out_ch, In_ch, Height, Width
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum LayerType {
    Conv2d = 0,
    DepthwiseConv2d = 1,
    MaxPool = 2,
    AvgPool = 3,
    GlobalAvgPool = 4,
    Relu = 5,
    Relu6 = 6,
    LeakyRelu = 7,
    Silu = 8,      // x * sigmoid(x)
    Sigmoid = 9,
    Concat = 10,
    Add = 11,
    Mul = 12,
    Upsample = 13,
    Reshape = 14,
    Transpose = 15,
    FullyConnected = 16,
    Softmax = 17,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum Activation {
    #[default]
    None = 0,
    Relu = 1,
    Relu6 = 2,
    LeakyRelu = 3,
    Silu = 4,
    Sigmoid = 5,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum Padding {
    #[default]
    Valid = 0,
    Same = 1,
    Explicit = 2,
}

/// Mars file header (76 bytes)
#[derive(Debug, Clone)]
pub struct MarsHeader {
    pub magic: u32,           // 4
    pub version: u32,         // 4
    pub num_layers: u32,      // 4
    pub num_tensors: u32,     // 4
    pub num_inputs: u32,      // 4
    pub num_outputs: u32,     // 4
    pub weights_offset: u32,  // 4
    pub weights_size: u32,    // 4
    pub input_tensor_ids: [u32; 4],  // 16
    pub output_tensor_ids: [u32; 4], // 16
    pub reserved: [u32; 3],   // 12
}

impl MarsHeader {
    pub fn new() -> Self {
        Self {
            magic: MARS_MAGIC,
            version: MARS_VERSION,
            num_layers: 0,
            num_tensors: 0,
            num_inputs: 0,
            num_outputs: 0,
            weights_offset: 0,
            weights_size: 0,
            input_tensor_ids: [0xFFFFFFFF; 4],
            output_tensor_ids: [0xFFFFFFFF; 4],
            reserved: [0; 3],
        }
    }

    pub fn write<W: Write>(&self, w: &mut W) -> io::Result<()> {
        w.write_u32::<LittleEndian>(self.magic)?;
        w.write_u32::<LittleEndian>(self.version)?;
        w.write_u32::<LittleEndian>(self.num_layers)?;
        w.write_u32::<LittleEndian>(self.num_tensors)?;
        w.write_u32::<LittleEndian>(self.num_inputs)?;
        w.write_u32::<LittleEndian>(self.num_outputs)?;
        w.write_u32::<LittleEndian>(self.weights_offset)?;
        w.write_u32::<LittleEndian>(self.weights_size)?;
        for id in &self.input_tensor_ids {
            w.write_u32::<LittleEndian>(*id)?;
        }
        for id in &self.output_tensor_ids {
            w.write_u32::<LittleEndian>(*id)?;
        }
        for r in &self.reserved {
            w.write_u32::<LittleEndian>(*r)?;
        }
        Ok(())
    }
}

/// Tensor descriptor (124 bytes)
#[derive(Debug, Clone)]
pub struct MarsTensor {
    pub id: u32,
    pub name: String,
    pub dtype: DataType,
    pub format: DataFormat,
    pub ndims: u32,
    pub shape: [u32; MARS_MAX_DIMS],
    pub scale: f32,
    pub zero_point: i32,
    pub data_offset: u32,
    pub data_size: u32,
}

impl MarsTensor {
    pub fn new(id: u32, name: &str) -> Self {
        Self {
            id,
            name: name.to_string(),
            dtype: DataType::Int8,
            format: DataFormat::Nhwc,
            ndims: 4,
            shape: [0; MARS_MAX_DIMS],
            scale: 1.0,
            zero_point: 0,
            data_offset: 0,
            data_size: 0,
        }
    }

    pub fn write<W: Write>(&self, w: &mut W) -> io::Result<()> {
        w.write_u32::<LittleEndian>(self.id)?;

        // Write name (64 bytes, null-padded)
        let mut name_bytes = [0u8; MARS_MAX_NAME];
        let name_len = self.name.len().min(MARS_MAX_NAME - 1);
        name_bytes[..name_len].copy_from_slice(&self.name.as_bytes()[..name_len]);
        w.write_all(&name_bytes)?;

        w.write_u32::<LittleEndian>(self.dtype as u32)?;
        w.write_u32::<LittleEndian>(self.format as u32)?;
        w.write_u32::<LittleEndian>(self.ndims)?;
        for dim in &self.shape {
            w.write_u32::<LittleEndian>(*dim)?;
        }
        w.write_f32::<LittleEndian>(self.scale)?;
        w.write_i32::<LittleEndian>(self.zero_point)?;
        w.write_u32::<LittleEndian>(self.data_offset)?;
        w.write_u32::<LittleEndian>(self.data_size)?;
        Ok(())
    }
}

/// Convolution parameters (48 bytes within params union)
#[derive(Debug, Clone, Default)]
pub struct ConvParams {
    pub kernel_h: u32,
    pub kernel_w: u32,
    pub stride_h: u32,
    pub stride_w: u32,
    pub dilation_h: u32,
    pub dilation_w: u32,
    pub padding: Padding,
    pub pad_top: u32,
    pub pad_bottom: u32,
    pub pad_left: u32,
    pub pad_right: u32,
    pub groups: u32,
    pub activation: Activation,
    pub weight_tensor_id: u32,
    pub bias_tensor_id: u32,  // 0xFFFFFFFF if no bias
}

impl ConvParams {
    pub fn write<W: Write>(&self, w: &mut W) -> io::Result<()> {
        w.write_u32::<LittleEndian>(self.kernel_h)?;
        w.write_u32::<LittleEndian>(self.kernel_w)?;
        w.write_u32::<LittleEndian>(self.stride_h)?;
        w.write_u32::<LittleEndian>(self.stride_w)?;
        w.write_u32::<LittleEndian>(self.dilation_h)?;
        w.write_u32::<LittleEndian>(self.dilation_w)?;
        w.write_u32::<LittleEndian>(self.padding as u32)?;
        w.write_u32::<LittleEndian>(self.pad_top)?;
        w.write_u32::<LittleEndian>(self.pad_bottom)?;
        w.write_u32::<LittleEndian>(self.pad_left)?;
        w.write_u32::<LittleEndian>(self.pad_right)?;
        w.write_u32::<LittleEndian>(self.groups)?;
        w.write_u32::<LittleEndian>(self.activation as u32)?;
        w.write_u32::<LittleEndian>(self.weight_tensor_id)?;
        w.write_u32::<LittleEndian>(self.bias_tensor_id)?;
        // Pad to 64 bytes
        w.write_all(&[0u8; 4])?;
        Ok(())
    }
}

/// Pool parameters
#[derive(Debug, Clone, Default)]
pub struct PoolParams {
    pub kernel_h: u32,
    pub kernel_w: u32,
    pub stride_h: u32,
    pub stride_w: u32,
    pub padding: Padding,
    pub pad_top: u32,
    pub pad_bottom: u32,
    pub pad_left: u32,
    pub pad_right: u32,
}

/// Concat parameters
#[derive(Debug, Clone, Default)]
pub struct ConcatParams {
    pub axis: u32,
    pub num_inputs: u32,
}

/// Upsample parameters
#[derive(Debug, Clone, Default)]
pub struct UpsampleParams {
    pub scale_h: u32,
    pub scale_w: u32,
    pub mode: u32,  // 0=nearest, 1=bilinear
}

/// Layer parameters union
#[derive(Debug, Clone)]
pub enum LayerParams {
    Conv(ConvParams),
    Pool(PoolParams),
    Concat(ConcatParams),
    Upsample(UpsampleParams),
    None,
}

/// Layer descriptor (112 bytes)
#[derive(Debug, Clone)]
pub struct MarsLayer {
    pub id: u32,
    pub layer_type: LayerType,
    pub num_inputs: u32,
    pub num_outputs: u32,
    pub input_tensor_ids: [u32; 4],
    pub output_tensor_ids: [u32; 4],
    pub params: LayerParams,
}

impl MarsLayer {
    pub fn new(id: u32, layer_type: LayerType) -> Self {
        Self {
            id,
            layer_type,
            num_inputs: 1,
            num_outputs: 1,
            input_tensor_ids: [0xFFFFFFFF; 4],
            output_tensor_ids: [0xFFFFFFFF; 4],
            params: LayerParams::None,
        }
    }

    pub fn write<W: Write>(&self, w: &mut W) -> io::Result<()> {
        w.write_u32::<LittleEndian>(self.id)?;
        w.write_u32::<LittleEndian>(self.layer_type as u32)?;
        w.write_u32::<LittleEndian>(self.num_inputs)?;
        w.write_u32::<LittleEndian>(self.num_outputs)?;
        for id in &self.input_tensor_ids {
            w.write_u32::<LittleEndian>(*id)?;
        }
        for id in &self.output_tensor_ids {
            w.write_u32::<LittleEndian>(*id)?;
        }
        // Write params (64 bytes)
        match &self.params {
            LayerParams::Conv(p) => p.write(w)?,
            _ => w.write_all(&[0u8; 64])?,
        }
        Ok(())
    }
}
