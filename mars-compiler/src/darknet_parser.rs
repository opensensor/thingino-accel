//! Darknet/SOD model parser
//!
//! Parses Darknet-style .weights files (used by SOD library) and
//! built-in architecture configurations.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

/// Darknet layer types
#[derive(Debug, Clone, PartialEq)]
pub enum DarknetLayerType {
    Convolutional,
    MaxPool,
    Region,
    Route,
    Shortcut,
    Upsample,
    Yolo,
}

/// Darknet layer configuration
#[derive(Debug, Clone)]
pub struct DarknetLayer {
    pub layer_type: DarknetLayerType,
    pub filters: u32,
    pub size: u32,
    pub stride: u32,
    pub pad: u32,
    pub batch_normalize: bool,
    pub activation: String,
    // Region/YOLO specific
    pub classes: u32,
    pub num: u32,  // number of anchors
    pub anchors: Vec<f32>,
}

impl Default for DarknetLayer {
    fn default() -> Self {
        Self {
            layer_type: DarknetLayerType::Convolutional,
            filters: 1,
            size: 3,
            stride: 1,
            pad: 1,
            batch_normalize: false,
            activation: "leaky".to_string(),
            classes: 1,
            num: 5,
            anchors: vec![],
        }
    }
}

/// Darknet network configuration
#[derive(Debug, Clone)]
pub struct DarknetConfig {
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    pub layers: Vec<DarknetLayer>,
}

/// Darknet weights file header
#[derive(Debug)]
pub struct DarknetWeightsHeader {
    pub major: i32,
    pub minor: i32,
    pub revision: i32,
    pub seen: u64,
}

/// Parsed Darknet model with weights
pub struct DarknetModel {
    pub config: DarknetConfig,
    pub weights: Vec<Vec<f32>>,  // weights per layer
}

/// Built-in architecture configurations (extracted from sod.c)
pub fn get_builtin_config(name: &str) -> Option<DarknetConfig> {
    match name {
        ":face" | "face" => Some(face_cnn_config()),
        ":tiny" | "tiny" => Some(tiny_yolo_config()),
        _ => None,
    }
}

/// Face detection CNN config (from zfaceCnn in sod.c)
fn face_cnn_config() -> DarknetConfig {
    DarknetConfig {
        width: 416,
        height: 416,
        channels: 3,
        layers: vec![
            // Conv 8 + MaxPool
            DarknetLayer { layer_type: DarknetLayerType::Convolutional, filters: 8, size: 3, stride: 1, pad: 1, batch_normalize: true, activation: "leaky".into(), ..Default::default() },
            DarknetLayer { layer_type: DarknetLayerType::MaxPool, size: 2, stride: 2, ..Default::default() },
            // Conv 16 + MaxPool
            DarknetLayer { layer_type: DarknetLayerType::Convolutional, filters: 16, size: 3, stride: 1, pad: 1, batch_normalize: true, activation: "leaky".into(), ..Default::default() },
            DarknetLayer { layer_type: DarknetLayerType::MaxPool, size: 2, stride: 2, ..Default::default() },
            // Conv 32 + MaxPool
            DarknetLayer { layer_type: DarknetLayerType::Convolutional, filters: 32, size: 3, stride: 1, pad: 1, batch_normalize: true, activation: "leaky".into(), ..Default::default() },
            DarknetLayer { layer_type: DarknetLayerType::MaxPool, size: 2, stride: 2, ..Default::default() },
            // Conv 64 + MaxPool
            DarknetLayer { layer_type: DarknetLayerType::Convolutional, filters: 64, size: 3, stride: 1, pad: 1, batch_normalize: true, activation: "leaky".into(), ..Default::default() },
            DarknetLayer { layer_type: DarknetLayerType::MaxPool, size: 2, stride: 2, ..Default::default() },
            // Conv 32 + MaxPool
            DarknetLayer { layer_type: DarknetLayerType::Convolutional, filters: 32, size: 3, stride: 1, pad: 1, batch_normalize: true, activation: "leaky".into(), ..Default::default() },
            DarknetLayer { layer_type: DarknetLayerType::MaxPool, size: 2, stride: 2, ..Default::default() },
            // Conv 64
            DarknetLayer { layer_type: DarknetLayerType::Convolutional, filters: 64, size: 3, stride: 1, pad: 1, batch_normalize: true, activation: "leaky".into(), ..Default::default() },
            // Conv 32
            DarknetLayer { layer_type: DarknetLayerType::Convolutional, filters: 32, size: 3, stride: 1, pad: 1, batch_normalize: true, activation: "leaky".into(), ..Default::default() },
            // Conv 64
            DarknetLayer { layer_type: DarknetLayerType::Convolutional, filters: 64, size: 3, stride: 1, pad: 1, batch_normalize: true, activation: "leaky".into(), ..Default::default() },
            // Conv 30 (output: 5 anchors * (4 coords + 1 obj + 1 class) = 30)
            DarknetLayer { layer_type: DarknetLayerType::Convolutional, filters: 30, size: 1, stride: 1, pad: 1, batch_normalize: false, activation: "linear".into(), ..Default::default() },
            // Region layer
            DarknetLayer { 
                layer_type: DarknetLayerType::Region, 
                classes: 1, 
                num: 5,
                anchors: vec![0.7, 0.86, 2.1, 2.1, 4.0, 4.16, 8.1, 8.1, 12.0, 12.16],
                ..Default::default() 
            },
        ],
    }
}

/// TinyYOLO config (from zTiny in sod.c) - placeholder, too large for device
fn tiny_yolo_config() -> DarknetConfig {
    DarknetConfig {
        width: 416,
        height: 416,
        channels: 3,
        layers: vec![
            // This is a simplified version - full TinyYOLO has many more layers
            DarknetLayer { layer_type: DarknetLayerType::Convolutional, filters: 16, size: 3, stride: 1, pad: 1, batch_normalize: true, activation: "leaky".into(), ..Default::default() },
            DarknetLayer { layer_type: DarknetLayerType::MaxPool, size: 2, stride: 2, ..Default::default() },
            // ... more layers would go here
        ],
    }
}

/// Load Darknet weights from a .sod/.weights file
pub fn load_darknet_weights<P: AsRef<Path>>(path: P, config: &DarknetConfig) -> Result<DarknetModel> {
    let mut file = File::open(path.as_ref())
        .with_context(|| format!("Failed to open weights file: {:?}", path.as_ref()))?;

    // Read header
    let header = read_weights_header(&mut file)?;
    println!("Darknet weights: v{}.{}.{}, seen={}",
             header.major, header.minor, header.revision, header.seen);

    // Read weights for each layer
    let mut weights = Vec::new();
    let mut in_channels = config.channels;
    let mut in_h = config.height;
    let mut in_w = config.width;

    for (i, layer) in config.layers.iter().enumerate() {
        match layer.layer_type {
            DarknetLayerType::Convolutional => {
                let layer_weights = read_conv_weights(&mut file, layer, in_channels)?;
                println!("  Layer {}: Conv {}x{}x{} -> {} filters ({} floats)",
                         i, layer.size, layer.size, in_channels, layer.filters, layer_weights.len());
                weights.push(layer_weights);

                // Update spatial dimensions
                if layer.pad > 0 {
                    // Same padding
                    in_h = (in_h + layer.stride - 1) / layer.stride;
                    in_w = (in_w + layer.stride - 1) / layer.stride;
                } else {
                    in_h = (in_h - layer.size) / layer.stride + 1;
                    in_w = (in_w - layer.size) / layer.stride + 1;
                }
                in_channels = layer.filters;
            }
            DarknetLayerType::MaxPool => {
                weights.push(vec![]);  // No weights for pooling
                in_h = in_h / layer.stride;
                in_w = in_w / layer.stride;
            }
            DarknetLayerType::Region | DarknetLayerType::Yolo => {
                weights.push(vec![]);  // No weights for detection layer
            }
            _ => {
                weights.push(vec![]);
            }
        }
    }

    Ok(DarknetModel { config: config.clone(), weights })
}

/// Read Darknet weights file header
fn read_weights_header(file: &mut File) -> Result<DarknetWeightsHeader> {
    let mut buf = [0u8; 4];

    file.read_exact(&mut buf)?;
    let major = i32::from_le_bytes(buf);

    file.read_exact(&mut buf)?;
    let minor = i32::from_le_bytes(buf);

    file.read_exact(&mut buf)?;
    let revision = i32::from_le_bytes(buf);

    // seen is 64-bit in newer versions, 32-bit in older
    let seen = if major * 10 + minor >= 2 {
        let mut buf8 = [0u8; 8];
        file.read_exact(&mut buf8)?;
        u64::from_le_bytes(buf8)
    } else {
        file.read_exact(&mut buf)?;
        u32::from_le_bytes(buf) as u64
    };

    Ok(DarknetWeightsHeader { major, minor, revision, seen })
}

/// Read weights for a convolutional layer
/// Order: biases, [scales, means, variances if batch_norm], weights
fn read_conv_weights(file: &mut File, layer: &DarknetLayer, in_channels: u32) -> Result<Vec<f32>> {
    let n = layer.filters as usize;
    let c = in_channels as usize;
    let k = layer.size as usize;

    let mut all_weights = Vec::new();

    // 1. Biases: n floats
    let biases = read_floats(file, n)?;
    all_weights.extend(&biases);

    // 2. Batch norm params if enabled: scales, means, variances (n each)
    if layer.batch_normalize {
        let scales = read_floats(file, n)?;
        let means = read_floats(file, n)?;
        let variances = read_floats(file, n)?;
        all_weights.extend(&scales);
        all_weights.extend(&means);
        all_weights.extend(&variances);
    }

    // 3. Weights: n * c * k * k floats
    let num_weights = n * c * k * k;
    let weights = read_floats(file, num_weights)?;
    all_weights.extend(&weights);

    Ok(all_weights)
}

/// Read n float32 values from file
fn read_floats(file: &mut File, n: usize) -> Result<Vec<f32>> {
    let mut buf = vec![0u8; n * 4];
    file.read_exact(&mut buf)?;

    Ok(buf.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

/// Parse a Darknet .cfg file (for custom architectures)
pub fn parse_darknet_cfg<P: AsRef<Path>>(path: P) -> Result<DarknetConfig> {
    let file = File::open(path.as_ref())?;
    let reader = BufReader::new(file);

    let mut width = 416;
    let mut height = 416;
    let mut channels = 3;
    let mut layers = Vec::new();
    let mut current_section = String::new();
    let mut current_layer = DarknetLayer::default();

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();

        // Skip comments and empty lines
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Section header
        if line.starts_with('[') && line.ends_with(']') {
            // Save previous layer if it was a real layer
            if !current_section.is_empty() && current_section != "net" {
                layers.push(current_layer.clone());
            }

            current_section = line[1..line.len()-1].to_string();
            current_layer = DarknetLayer::default();
            current_layer.layer_type = match current_section.as_str() {
                "convolutional" => DarknetLayerType::Convolutional,
                "maxpool" => DarknetLayerType::MaxPool,
                "region" => DarknetLayerType::Region,
                "yolo" => DarknetLayerType::Yolo,
                "route" => DarknetLayerType::Route,
                "shortcut" => DarknetLayerType::Shortcut,
                "upsample" => DarknetLayerType::Upsample,
                _ => DarknetLayerType::Convolutional,
            };
            continue;
        }

        // Key=value pair
        if let Some(eq_pos) = line.find('=') {
            let key = line[..eq_pos].trim();
            let value = line[eq_pos+1..].trim();

            match current_section.as_str() {
                "net" => {
                    match key {
                        "width" => width = value.parse().unwrap_or(416),
                        "height" => height = value.parse().unwrap_or(416),
                        "channels" => channels = value.parse().unwrap_or(3),
                        _ => {}
                    }
                }
                "convolutional" => {
                    match key {
                        "filters" => current_layer.filters = value.parse().unwrap_or(1),
                        "size" => current_layer.size = value.parse().unwrap_or(3),
                        "stride" => current_layer.stride = value.parse().unwrap_or(1),
                        "pad" => current_layer.pad = value.parse().unwrap_or(0),
                        "batch_normalize" => current_layer.batch_normalize = value == "1",
                        "activation" => current_layer.activation = value.to_string(),
                        _ => {}
                    }
                }
                "maxpool" => {
                    match key {
                        "size" => current_layer.size = value.parse().unwrap_or(2),
                        "stride" => current_layer.stride = value.parse().unwrap_or(2),
                        _ => {}
                    }
                }
                "region" | "yolo" => {
                    match key {
                        "classes" => current_layer.classes = value.parse().unwrap_or(1),
                        "num" => current_layer.num = value.parse().unwrap_or(5),
                        "anchors" => {
                            current_layer.anchors = value.split(',')
                                .filter_map(|s| s.trim().parse().ok())
                                .collect();
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }
    }

    // Don't forget the last layer
    if !current_section.is_empty() && current_section != "net" {
        layers.push(current_layer);
    }

    Ok(DarknetConfig { width, height, channels, layers })
}

