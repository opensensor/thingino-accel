//! Weight data extractor for MGK files
//!
//! MGK files contain weight data appended after the ELF sections.
//! This module extracts and analyzes that weight data.
//!
//! Weight format: NMHWSOIB2
//! Layout: [N_OFP, M_IFP, KERNEL_H, KERNEL_W, S_BIT2, OFP, IFP]
//! - N_OFP = ceil(out_channels/32)
//! - M_IFP = ceil(in_channels/32)
//! - S_BIT2 = 4 for 8-bit quantization
//! - OFP = IFP = 32 (channel packing factor)
//! - Size formula: N_OFP × M_IFP × KH × KW × 1024 bytes

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

    // Segment weights into sections
    let sections = segment_weights(weights);
    if !sections.is_empty() {
        stats.sections = Some(sections);
    }

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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sections: Option<Vec<WeightSection>>,
}

/// A section of weight data with similar sparsity characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightSection {
    pub offset: u64,
    pub size: u64,
    pub sparsity: f64,
    pub mean: f64,
    pub layer_hint: Option<String>,
}

/// Segment weights into sections based on sparsity patterns
pub fn segment_weights(weights: &[u8]) -> Vec<WeightSection> {
    let mut sections = Vec::new();
    let window_size = 1024;

    if weights.len() < window_size {
        return sections;
    }

    let mut section_start = 0usize;
    let mut prev_sparsity = 0.0f64;

    for i in (0..weights.len()).step_by(window_size) {
        let end = (i + window_size).min(weights.len());
        let chunk = &weights[i..end];

        let zero_count = chunk.iter().filter(|&&b| b == 0).count();
        let sparsity = zero_count as f64 / chunk.len() as f64;

        // Detect section boundary when sparsity changes by >20%
        if (sparsity - prev_sparsity).abs() > 0.2 && i > 0 {
            let section_data = &weights[section_start..i];
            let section_zeros = section_data.iter().filter(|&&b| b == 0).count();
            let section_sparsity = section_zeros as f64 / section_data.len() as f64;
            let section_mean: f64 = section_data.iter()
                .map(|&b| b as i8 as f64)
                .sum::<f64>() / section_data.len() as f64;

            // Guess layer type based on size and sparsity
            let layer_hint = guess_layer_type(section_data.len(), section_sparsity);

            sections.push(WeightSection {
                offset: section_start as u64,
                size: section_data.len() as u64,
                sparsity: section_sparsity,
                mean: section_mean,
                layer_hint,
            });

            section_start = i;
        }

        prev_sparsity = sparsity;
    }

    // Add final section
    if section_start < weights.len() {
        let section_data = &weights[section_start..];
        let section_zeros = section_data.iter().filter(|&&b| b == 0).count();
        let section_sparsity = section_zeros as f64 / section_data.len() as f64;
        let section_mean: f64 = section_data.iter()
            .map(|&b| b as i8 as f64)
            .sum::<f64>() / section_data.len() as f64;
        let layer_hint = guess_layer_type(section_data.len(), section_sparsity);

        sections.push(WeightSection {
            offset: section_start as u64,
            size: section_data.len() as u64,
            sparsity: section_sparsity,
            mean: section_mean,
            layer_hint,
        });
    }

    sections
}

/// Guess layer type based on weight block characteristics
fn guess_layer_type(size: usize, sparsity: f64) -> Option<String> {
    // GRU weights are typically:
    // - Large (>10KB per gate matrix)
    // - Dense (low sparsity for active weights)
    if size > 30000 && sparsity < 0.1 {
        return Some("GRU_weights".to_string());
    }

    if size > 10000 && sparsity < 0.15 {
        return Some("GRU_or_Conv".to_string());
    }

    // BatchNorm has small weight blocks (gamma, beta, mean, var)
    if size < 2000 && sparsity < 0.2 {
        return Some("BatchNorm_params".to_string());
    }

    // High sparsity sections might be padding or sparse activations
    if sparsity > 0.5 {
        return Some("sparse_or_padding".to_string());
    }

    None
}

/// Known layer weight offsets for AEC model
/// These were discovered through reverse engineering
pub static KNOWN_LAYER_OFFSETS: &[(&str, u64, u64)] = &[
    ("layer_46_gru_bidir", 0x00000, 12864),
    ("layer_63_feature", 0x03500, 448),
    ("layer_68_feature", 0x03900, 448),
    ("layer_35_feature", 0x03d00, 704),
    ("layer_73_feature", 0x04100, 448),
    ("layer_44_feature", 0x11f00, 576),
    ("layer_58_feature", 0x12300, 576),
    ("layer_78_feature", 0x12700, 320),
    ("layer_4_feature", 0x12a00, 3648),
    ("layer_16_feature", 0x13b00, 2112),
    ("layer_2_feature", 0x14b00, 320),
    ("layer_20_feature", 0x21180, 832),
    ("layer_26_feature", 0x215c0, 832),
    ("layer_28_feature", 0x21a40, 1408),
    ("layer_37_gru", 0x220c0, 4096),
    ("layer_10_feature", 0x231c0, 2496),
    ("layer_32_feature", 0x23cc0, 768),
    ("layer_41_feature", 0x24100, 704),
    ("layer_8_feature", 0x24500, 1024),
    ("layer_14_feature", 0x24a00, 1024),
    ("layer_22_feature", 0x25140, 1772),
];

/// Layer weight mapping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerWeightMapping {
    pub layer_name: String,
    pub offset: u64,
    pub size: u64,
}

/// Get known layer weight mappings
pub fn get_known_layer_mappings() -> Vec<LayerWeightMapping> {
    KNOWN_LAYER_OFFSETS
        .iter()
        .map(|(name, offset, size)| LayerWeightMapping {
            layer_name: name.to_string(),
            offset: *offset,
            size: *size,
        })
        .collect()
}

/// Weight block analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightBlockAnalysis {
    pub block_index: usize,
    pub offset: u64,
    pub nonzero_count: usize,
    pub std_dev: f64,
    pub min_val: i8,
    pub max_val: i8,
    pub is_dense: bool,
}

/// Analyze weight data in 1024-byte blocks
/// Dense blocks have: nonzero > 900 && std > 20
pub fn analyze_weight_blocks(weights: &[u8]) -> Vec<WeightBlockAnalysis> {
    let block_size = 1024;
    let mut blocks = Vec::new();

    for (i, chunk) in weights.chunks(block_size).enumerate() {
        let nonzero_count = chunk.iter().filter(|&&b| b != 0).count();

        // Calculate statistics
        let sum: i64 = chunk.iter().map(|&b| b as i8 as i64).sum();
        let mean = sum as f64 / chunk.len() as f64;

        let variance: f64 = chunk.iter()
            .map(|&b| {
                let diff = b as i8 as f64 - mean;
                diff * diff
            })
            .sum::<f64>() / chunk.len() as f64;
        let std_dev = variance.sqrt();

        let min_val = chunk.iter().map(|&b| b as i8).min().unwrap_or(0);
        let max_val = chunk.iter().map(|&b| b as i8).max().unwrap_or(0);

        // Dense block criteria: nonzero > 900 && std > 20
        let is_dense = nonzero_count > 900 && std_dev > 20.0;

        blocks.push(WeightBlockAnalysis {
            block_index: i,
            offset: (i * block_size) as u64,
            nonzero_count,
            std_dev,
            min_val,
            max_val,
            is_dense,
        });
    }

    blocks
}

/// Count dense weight blocks
pub fn count_dense_blocks(weights: &[u8]) -> usize {
    analyze_weight_blocks(weights)
        .iter()
        .filter(|b| b.is_dense)
        .count()
}

/// NMHWSOIB2 weight unpacking parameters
#[derive(Debug, Clone)]
pub struct NmhwsoibParams {
    pub out_channels: usize,
    pub in_channels: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
}

impl NmhwsoibParams {
    /// Calculate the packed size in bytes
    pub fn packed_size(&self) -> usize {
        let n_ofp = (self.out_channels + 31) / 32;
        let m_ifp = (self.in_channels + 31) / 32;
        n_ofp * m_ifp * self.kernel_h * self.kernel_w * 1024
    }

    /// Calculate dimensions for unpacking
    pub fn packed_dims(&self) -> (usize, usize, usize, usize) {
        let n_ofp = (self.out_channels + 31) / 32;
        let m_ifp = (self.in_channels + 31) / 32;
        (n_ofp, m_ifp, self.kernel_h, self.kernel_w)
    }
}

/// Unpack weights from NMHWSOIB2 format to standard OIHW format
///
/// NMHWSOIB2 layout: [N_OFP, M_IFP, KERNEL_H, KERNEL_W, 32, 32]
/// Output layout: [out_channels, in_channels, kernel_h, kernel_w]
pub fn unpack_nmhwsoib2(data: &[u8], params: &NmhwsoibParams) -> Result<Vec<i8>> {
    let (n_ofp, m_ifp, kh, kw) = params.packed_dims();
    let expected_size = n_ofp * m_ifp * kh * kw * 1024;

    if data.len() < expected_size {
        anyhow::bail!(
            "Data too small: expected {} bytes, got {}",
            expected_size,
            data.len()
        );
    }

    // Reshape: [n_ofp, m_ifp, kh, kw, 32, 32]
    // Transpose: [n_ofp, 32, m_ifp, 32, kh, kw] -> [out_ch, in_ch, kh, kw]
    let out_ch = n_ofp * 32;
    let in_ch = m_ifp * 32;
    let mut output = vec![0i8; out_ch * in_ch * kh * kw];

    for n in 0..n_ofp {
        for m in 0..m_ifp {
            for h in 0..kh {
                for w in 0..kw {
                    for o in 0..32 {
                        for i in 0..32 {
                            // Source index in packed format
                            let src_idx = ((((n * m_ifp + m) * kh + h) * kw + w) * 32 + o) * 32 + i;
                            // Destination index in OIHW format
                            let out_c = n * 32 + o;
                            let in_c = m * 32 + i;
                            let dst_idx = ((out_c * in_ch + in_c) * kh + h) * kw + w;

                            if src_idx < data.len() && dst_idx < output.len() {
                                output[dst_idx] = data[src_idx] as i8;
                            }
                        }
                    }
                }
            }
        }
    }

    // Trim to actual dimensions
    let actual_out = params.out_channels;
    let actual_in = params.in_channels;
    let mut trimmed = vec![0i8; actual_out * actual_in * kh * kw];

    for oc in 0..actual_out {
        for ic in 0..actual_in {
            for h in 0..kh {
                for w in 0..kw {
                    let src_idx = ((oc * in_ch + ic) * kh + h) * kw + w;
                    let dst_idx = ((oc * actual_in + ic) * kh + h) * kw + w;
                    trimmed[dst_idx] = output[src_idx];
                }
            }
        }
    }

    Ok(trimmed)
}

/// Detect weight block boundaries based on sparsity changes
pub fn detect_weight_boundaries(weights: &[u8]) -> Vec<u64> {
    let blocks = analyze_weight_blocks(weights);
    let mut boundaries = vec![0u64];

    for i in 1..blocks.len() {
        let prev = &blocks[i - 1];
        let curr = &blocks[i];

        // Boundary when transitioning between dense and sparse
        if prev.is_dense != curr.is_dense {
            boundaries.push(curr.offset);
        }
        // Or when there's a significant change in std_dev
        else if (prev.std_dev - curr.std_dev).abs() > 30.0 {
            boundaries.push(curr.offset);
        }
    }

    boundaries
}

/// Extract weight region from file
pub fn find_weight_region<R: Read + Seek>(reader: &mut R) -> Result<(u64, u64)> {
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

    // Weight region starts after section headers
    let weight_start = shoff + shentsize * shnum;
    let weight_size = file_size - weight_start;

    Ok((weight_start, weight_size))
}
