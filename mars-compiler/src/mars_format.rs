//! Mars binary format definitions
//!
//! This matches the C structures in include/mars.h exactly

use byteorder::{LittleEndian, WriteBytesExt};
use std::io::{self, Write};

pub const MARS_MAGIC: u32 = 0x5352414D; // "MARS" in little-endian
pub const MARS_VERSION_MAJOR: u16 = 1;
pub const MARS_VERSION_MINOR: u16 = 0;
pub const MARS_MAX_DIMS: usize = 6;
pub const MARS_MAX_NAME: usize = 60;  // 64 - 4 (id field)

// Header: 76 bytes (verified with C sizeof)
pub const HEADER_SIZE: usize = 76;
// Tensor: 124 bytes (verified with C sizeof)
pub const TENSOR_SIZE: usize = 124;
// Layer: 112 bytes (verified with C sizeof)
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

/// Data formats - matching NNA hardware expectations from uranus_common_type.h
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum DataFormat {
    Nhwc = 0,       // Feature: [N, H, W, C] - basic layout
    Ndhwc32 = 1,    // Feature: [N, D_C32, H, W, CHN_32] - NNA native, 32-channel groups
    Hwio = 2,       // Weight: [H, W, I, O]
    Nmhwsoib2 = 3,  // Weight: [N_OFP, M_IFP, H, W, S_BIT2, OFP, IFP] - NNA native packed
    Nmc32 = 4,      // Bias/BN: [N_OFP, M_BT, CHN_32]
    D1 = 5,         // Scale/LUT: [d1]
    Ohwi = 6,       // Weight: [O, H, W, I]
    Nchw = 7,       // Feature: [N, C, H, W] - ONNX default
    Oihw = 8,       // Weight: [O, I, H, W] - ONNX default
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
    BatchNorm = 18,  // BatchNormalization: y = (x - mean) / sqrt(var + eps) * gamma + beta
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

/// Mars file header (76 bytes) - matches C mars_header_t exactly
/// Offsets verified: magic(0), version_major(4), version_minor(6), flags(8),
/// num_layers(12), num_tensors(16), num_inputs(20), num_outputs(24),
/// weights_offset(28), weights_size(36), input_tensor_ids(44), output_tensor_ids(60)
#[derive(Debug, Clone)]
pub struct MarsHeader {
    pub magic: u32,                     // offset 0, 4 bytes
    pub version_major: u16,             // offset 4, 2 bytes
    pub version_minor: u16,             // offset 6, 2 bytes
    pub flags: u32,                     // offset 8, 4 bytes (reserved)
    pub num_layers: u32,                // offset 12, 4 bytes
    pub num_tensors: u32,               // offset 16, 4 bytes
    pub num_inputs: u32,                // offset 20, 4 bytes
    pub num_outputs: u32,               // offset 24, 4 bytes
    pub weights_offset: u64,            // offset 28, 8 bytes
    pub weights_size: u64,              // offset 36, 8 bytes
    pub input_tensor_ids: [u32; 4],     // offset 44, 16 bytes
    pub output_tensor_ids: [u32; 4],    // offset 60, 16 bytes
}                                       // Total: 76 bytes

impl MarsHeader {
    pub fn new() -> Self {
        Self {
            magic: MARS_MAGIC,
            version_major: MARS_VERSION_MAJOR,
            version_minor: MARS_VERSION_MINOR,
            flags: 0,
            num_layers: 0,
            num_tensors: 0,
            num_inputs: 0,
            num_outputs: 0,
            weights_offset: 0,
            weights_size: 0,
            input_tensor_ids: [0xFFFFFFFF; 4],
            output_tensor_ids: [0xFFFFFFFF; 4],
        }
    }

    pub fn write<W: Write>(&self, w: &mut W) -> io::Result<()> {
        w.write_u32::<LittleEndian>(self.magic)?;
        w.write_u16::<LittleEndian>(self.version_major)?;
        w.write_u16::<LittleEndian>(self.version_minor)?;
        w.write_u32::<LittleEndian>(self.flags)?;
        w.write_u32::<LittleEndian>(self.num_layers)?;
        w.write_u32::<LittleEndian>(self.num_tensors)?;
        w.write_u32::<LittleEndian>(self.num_inputs)?;
        w.write_u32::<LittleEndian>(self.num_outputs)?;
        w.write_u64::<LittleEndian>(self.weights_offset)?;
        w.write_u64::<LittleEndian>(self.weights_size)?;
        for id in &self.input_tensor_ids {
            w.write_u32::<LittleEndian>(*id)?;
        }
        for id in &self.output_tensor_ids {
            w.write_u32::<LittleEndian>(*id)?;
        }
        Ok(())
    }
}

/// Tensor descriptor (124 bytes) - matches C mars_tensor_t exactly
/// Offsets: id(0), name(4), dtype(64), format(68), ndims(72), shape(76),
/// data_offset(100), data_size(108), scale(116), zero_point(120)
#[derive(Debug, Clone)]
pub struct MarsTensor {
    pub id: u32,                        // offset 0, 4 bytes
    pub name: String,                   // offset 4, 60 bytes (MARS_MAX_NAME)
    pub dtype: DataType,                // offset 64, 4 bytes
    pub format: DataFormat,             // offset 68, 4 bytes
    pub ndims: u32,                     // offset 72, 4 bytes
    pub shape: [i32; MARS_MAX_DIMS],    // offset 76, 24 bytes (6 * 4)
    pub data_offset: u64,               // offset 100, 8 bytes
    pub data_size: u64,                 // offset 108, 8 bytes
    pub scale: f32,                     // offset 116, 4 bytes
    pub zero_point: i32,                // offset 120, 4 bytes
}                                       // Total: 124 bytes

impl MarsTensor {
    pub fn new(id: u32, name: &str) -> Self {
        Self {
            id,
            name: name.to_string(),
            dtype: DataType::Int8,
            format: DataFormat::Nhwc,
            ndims: 4,
            shape: [0; MARS_MAX_DIMS],
            data_offset: 0,
            data_size: 0,
            scale: 1.0,
            zero_point: 0,
        }
    }

    pub fn write<W: Write>(&self, w: &mut W) -> io::Result<()> {
        w.write_u32::<LittleEndian>(self.id)?;

        // Write name (60 bytes, null-padded)
        let mut name_bytes = [0u8; MARS_MAX_NAME];
        let name_len = self.name.len().min(MARS_MAX_NAME - 1);
        name_bytes[..name_len].copy_from_slice(&self.name.as_bytes()[..name_len]);
        w.write_all(&name_bytes)?;

        w.write_u32::<LittleEndian>(self.dtype as u32)?;
        w.write_u32::<LittleEndian>(self.format as u32)?;
        w.write_u32::<LittleEndian>(self.ndims)?;
        for dim in &self.shape {
            w.write_i32::<LittleEndian>(*dim)?;
        }
        w.write_u64::<LittleEndian>(self.data_offset)?;
        w.write_u64::<LittleEndian>(self.data_size)?;
        w.write_f32::<LittleEndian>(self.scale)?;
        w.write_i32::<LittleEndian>(self.zero_point)?;
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


/// Pack weights from OIHW (ONNX) to NMHWSOIB2 (NNA native format)
///
/// NMHWSOIB2 format for INT8:
/// - Shape: [N_OFP, M_IFP, KH, KW, OFP, IFP]
/// - N_OFP = ceil(out_ch/32), M_IFP = ceil(in_ch/32)
/// - OFP = IFP = 32 (channel packing factor)
/// - Size: N_OFP × M_IFP × KH × KW × 1024 bytes
pub fn pack_weights_nmhwsoib2(
    weights: &[i8],
    out_ch: usize,
    in_ch: usize,
    kh: usize,
    kw: usize,
) -> Vec<u8> {
    let n_ofp = (out_ch + 31) / 32;
    let m_ifp = (in_ch + 31) / 32;
    let packed_size = n_ofp * m_ifp * kh * kw * 32 * 32;
    let mut packed = vec![0u8; packed_size];

    for o in 0..out_ch {
        for i in 0..in_ch {
            for h in 0..kh {
                for w in 0..kw {
                    // Source index in OIHW format
                    let src_idx = ((o * in_ch + i) * kh + h) * kw + w;

                    // Destination in NMHWSOIB2: [n_ofp, m_ifp, kh, kw, ofp, ifp]
                    let n = o / 32;
                    let ofp = o % 32;
                    let m = i / 32;
                    let ifp = i % 32;
                    let dst_idx = ((((n * m_ifp + m) * kh + h) * kw + w) * 32 + ofp) * 32 + ifp;

                    if src_idx < weights.len() && dst_idx < packed.len() {
                        packed[dst_idx] = weights[src_idx] as u8;
                    }
                }
            }
        }
    }

    packed
}

/// Calculate NMHWSOIB2 packed size in bytes
pub fn nmhwsoib2_size(out_ch: usize, in_ch: usize, kh: usize, kw: usize) -> usize {
    let n_ofp = (out_ch + 31) / 32;
    let m_ifp = (in_ch + 31) / 32;
    n_ofp * m_ifp * kh * kw * 1024  // 32 * 32 = 1024
}

/// Calculate NDHWC32 tensor size for feature maps
///
/// NDHWC32 format:
/// - Shape: [N, D_C32, H, W, 32]
/// - D_C32 = ceil(channels/32)
/// - Size: N × D_C32 × H × W × 32 bytes (for INT8)
pub fn ndhwc32_size(batch: usize, channels: usize, height: usize, width: usize) -> usize {
    let d_c32 = (channels + 31) / 32;
    batch * d_c32 * height * width * 32
}

/// Convert NCHW tensor to NDHWC32 format (in-place would require same size)
pub fn convert_nchw_to_ndhwc32(
    input: &[u8],
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
) -> Vec<u8> {
    let d_c32 = (channels + 31) / 32;
    let out_size = batch * d_c32 * height * width * 32;
    let mut output = vec![0u8; out_size];

    for n in 0..batch {
        for c in 0..channels {
            for h in 0..height {
                for w in 0..width {
                    // Source: NCHW
                    let src_idx = ((n * channels + c) * height + h) * width + w;

                    // Dest: NDHWC32 = [N, D_C32, H, W, 32]
                    let d = c / 32;
                    let c32 = c % 32;
                    let dst_idx = (((n * d_c32 + d) * height + h) * width + w) * 32 + c32;

                    if src_idx < input.len() && dst_idx < output.len() {
                        output[dst_idx] = input[src_idx];
                    }
                }
            }
        }
    }

    output
}