//! Binary data parser for MGK serialized layer parameters
//!
//! The MGK model contains serialized layer parameters in a binary format.
//! This module parses the serialized data to extract layer configurations.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::io::{Cursor, Read};
use byteorder::{LittleEndian, ReadBytesExt};

/// Parsed layer parameter from binary data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerParam {
    pub op_type: i32,
    pub layer_id: u16,
    pub param_index: u64,
    pub flags: u32,
}

/// Common layer parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommonParam {
    pub layer_name: String,
    pub input_tensors: Vec<String>,
    pub output_tensors: Vec<String>,
    pub shape_dims: Vec<i32>,
}

/// Tensor parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorParam {
    pub is_valid: bool,
    pub num_dims: i32,
    pub shapes: Vec<Vec<i32>>,
    pub offset: i32,
    pub size: i32,
}

/// Binary data reader helper
pub struct BinaryReader<'a> {
    cursor: Cursor<&'a [u8]>,
}

impl<'a> BinaryReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            cursor: Cursor::new(data),
        }
    }

    pub fn position(&self) -> u64 {
        self.cursor.position()
    }

    pub fn set_position(&mut self, pos: u64) {
        self.cursor.set_position(pos);
    }

    /// Read a 32-bit signed integer (little-endian)
    pub fn read_i32(&mut self) -> Result<i32> {
        self.cursor.read_i32::<LittleEndian>()
            .map_err(|e| anyhow!("Failed to read i32: {}", e))
    }

    /// Read a 32-bit unsigned integer (little-endian)
    pub fn read_u32(&mut self) -> Result<u32> {
        self.cursor.read_u32::<LittleEndian>()
            .map_err(|e| anyhow!("Failed to read u32: {}", e))
    }

    /// Read a 16-bit unsigned integer (little-endian)
    pub fn read_u16(&mut self) -> Result<u16> {
        self.cursor.read_u16::<LittleEndian>()
            .map_err(|e| anyhow!("Failed to read u16: {}", e))
    }

    /// Read a 64-bit unsigned integer (little-endian)
    pub fn read_u64(&mut self) -> Result<u64> {
        self.cursor.read_u64::<LittleEndian>()
            .map_err(|e| anyhow!("Failed to read u64: {}", e))
    }

    /// Read a single byte
    pub fn read_u8(&mut self) -> Result<u8> {
        self.cursor.read_u8()
            .map_err(|e| anyhow!("Failed to read u8: {}", e))
    }

    /// Read a length-prefixed string
    pub fn read_string(&mut self) -> Result<String> {
        let len = self.read_i32()? as usize;
        if len == 0 {
            return Ok(String::new());
        }
        let mut buf = vec![0u8; len];
        self.cursor.read_exact(&mut buf)
            .map_err(|e| anyhow!("Failed to read string data: {}", e))?;
        String::from_utf8(buf)
            .map_err(|e| anyhow!("Invalid UTF-8 string: {}", e))
    }

    /// Read a vector of strings
    pub fn read_string_vector(&mut self) -> Result<Vec<String>> {
        let count = self.read_i32()? as usize;
        let mut strings = Vec::with_capacity(count);
        for _ in 0..count {
            strings.push(self.read_string()?);
        }
        Ok(strings)
    }

    /// Read a vector of i32
    pub fn read_i32_vector(&mut self) -> Result<Vec<i32>> {
        let count = self.read_i32()? as usize;
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            values.push(self.read_i32()?);
        }
        Ok(values)
    }
}

/// Read layer parameters from binary data
pub fn read_layer_param(reader: &mut BinaryReader) -> Result<LayerParam> {
    let op_type = reader.read_i32()?;
    let layer_id = reader.read_u16()?;
    let param_index = reader.read_u64()?;
    let flags = reader.read_u32()?;
    
    Ok(LayerParam {
        op_type,
        layer_id,
        param_index,
        flags,
    })
}

/// Read common layer parameters from binary data
pub fn read_common_param(reader: &mut BinaryReader) -> Result<CommonParam> {
    let layer_name = reader.read_string()?;
    let input_tensors = reader.read_string_vector()?;
    let output_tensors = reader.read_string_vector()?;
    let shape_dims = reader.read_i32_vector()?;
    
    Ok(CommonParam {
        layer_name,
        input_tensors,
        output_tensors,
        shape_dims,
    })
}

