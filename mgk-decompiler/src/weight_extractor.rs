//! Weight data extractor for MGK files
//!
//! MGK files contain weight data appended after the ELF sections.
//! This module extracts and analyzes that weight data.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::io::{Read, Seek, SeekFrom};

/// Weight data header information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightHeader {
    /// Offset in file where appended data starts
    pub data_offset: u64,
    /// Total size of appended data
    pub data_size: u64,
    /// Offset where dense weight data starts (relative to data_offset)
    pub weights_offset: u64,
    /// Size of dense weight data
    pub weights_size: u64,
    /// Header entries (metadata before weights)
    pub header_entries: Vec<HeaderEntry>,
}

/// Header entry in the appended data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeaderEntry {
    pub offset: u64,
    pub value1: i64,
    pub value2: i64,
}

/// Weight block information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightBlock {
    pub offset: u64,
    pub size: u64,
    pub layer_name: Option<String>,
    pub data_type: String,
}

/// Extract weight data information from an MGK file
pub fn extract_weight_info<R: Read + Seek>(reader: &mut R) -> Result<WeightHeader> {
    // Get file size
    reader.seek(SeekFrom::End(0))?;
    let file_size = reader.stream_position()?;
    
    // Read ELF header to find section header offset
    reader.seek(SeekFrom::Start(0x20))?;
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    let shoff = u32::from_le_bytes(buf) as u64;
    
    // Read section header entry size and count
    reader.seek(SeekFrom::Start(0x2e))?;
    let mut buf2 = [0u8; 2];
    reader.read_exact(&mut buf2)?;
    let shentsize = u16::from_le_bytes(buf2) as u64;
    
    reader.seek(SeekFrom::Start(0x30))?;
    reader.read_exact(&mut buf2)?;
    let shnum = u16::from_le_bytes(buf2) as u64;
    
    // Calculate where appended data starts (after section headers)
    let data_offset = shoff + shentsize * shnum;
    let data_size = file_size - data_offset;
    
    // Parse header entries (256-byte blocks with sparse data)
    let mut header_entries = Vec::new();
    let mut weights_offset = 0u64;
    
    reader.seek(SeekFrom::Start(data_offset))?;
    for i in 0..16 {
        let block_offset = i * 256;
        reader.seek(SeekFrom::Start(data_offset + block_offset))?;
        
        let mut block = [0u8; 256];
        reader.read_exact(&mut block)?;
        
        // Count non-zero bytes
        let non_zero: usize = block.iter().filter(|&&b| b != 0).count();
        
        if non_zero > 200 {
            // Dense data starts here
            weights_offset = block_offset;
            break;
        }
        
        if non_zero > 0 && non_zero <= 32 {
            // Parse as header entry (two i64 values)
            let value1 = i64::from_le_bytes(block[0..8].try_into()?);
            let value2 = i64::from_le_bytes(block[8..16].try_into()?);
            header_entries.push(HeaderEntry {
                offset: block_offset,
                value1,
                value2,
            });
        }
    }
    
    let weights_size = data_size - weights_offset;
    
    Ok(WeightHeader {
        data_offset,
        data_size,
        weights_offset,
        weights_size,
        header_entries,
    })
}

/// Extract raw weight data from an MGK file
pub fn extract_weights<R: Read + Seek>(reader: &mut R, header: &WeightHeader) -> Result<Vec<u8>> {
    let weights_start = header.data_offset + header.weights_offset;
    reader.seek(SeekFrom::Start(weights_start))?;
    
    let mut weights = vec![0u8; header.weights_size as usize];
    reader.read_exact(&mut weights)?;
    
    Ok(weights)
}

/// Analyze weight data statistics
pub fn analyze_weights(weights: &[u8]) -> WeightStats {
    let mut stats = WeightStats::default();
    stats.total_bytes = weights.len();
    
    // Calculate histogram for INT8 weights
    let mut histogram = [0u64; 256];
    for &b in weights {
        histogram[b as usize] += 1;
    }
    
    // Find min/max non-zero values
    for (i, &count) in histogram.iter().enumerate() {
        if count > 0 {
            if stats.min_value.is_none() {
                stats.min_value = Some(i as i8 as i32);
            }
            stats.max_value = Some(i as i8 as i32);
        }
    }
    
    // Calculate mean
    let sum: i64 = weights.iter().map(|&b| b as i8 as i64).sum();
    stats.mean = sum as f64 / weights.len() as f64;
    
    // Count zeros
    stats.zero_count = histogram[0];
    stats.zero_percentage = (stats.zero_count as f64 / weights.len() as f64) * 100.0;
    
    stats
}

/// Weight data statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WeightStats {
    pub total_bytes: usize,
    pub min_value: Option<i32>,
    pub max_value: Option<i32>,
    pub mean: f64,
    pub zero_count: u64,
    pub zero_percentage: f64,
}

