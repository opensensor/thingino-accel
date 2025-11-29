//! ELF parser for MGK files

use crate::types::{Architecture, MgkFile, Section, Symbol, SymbolType};
use anyhow::{bail, Context, Result};
use cpp_demangle::Symbol as CppSymbol;
use goblin::elf::Elf;
use std::path::Path;

/// Parse an MGK file and extract relevant information
pub fn parse_mgk_file(path: &Path) -> Result<MgkFile> {
    let data = std::fs::read(path).context("Failed to read file")?;
    
    let elf = Elf::parse(&data).context("Failed to parse ELF")?;
    
    // Verify architecture
    let architecture = match (elf.header.e_machine, elf.little_endian) {
        (goblin::elf::header::EM_MIPS, true) => Architecture::Mips32El,
        _ => Architecture::Unknown,
    };
    
    if matches!(architecture, Architecture::Unknown) {
        bail!("Unsupported architecture: expected MIPS32 little-endian");
    }
    
    // Extract sections
    let sections = extract_sections(&elf, &data)?;
    
    // Extract symbols
    let symbols = extract_symbols(&elf)?;
    
    Ok(MgkFile {
        architecture,
        entry_point: elf.entry,
        sections,
        symbols,
    })
}

/// Extract sections from ELF
fn extract_sections(elf: &Elf, data: &[u8]) -> Result<Vec<Section>> {
    let mut sections = Vec::new();
    
    for section in &elf.section_headers {
        let name = elf.shdr_strtab
            .get_at(section.sh_name)
            .unwrap_or("<unknown>");
        
        // Only include relevant sections
        if name.is_empty() || name.starts_with(".debug") {
            continue;
        }
        
        let offset = section.sh_offset as usize;
        let size = section.sh_size as usize;
        
        // Extract section data if available
        let section_data = if offset + size <= data.len() && size > 0 {
            data[offset..offset + size].to_vec()
        } else {
            Vec::new()
        };
        
        sections.push(Section {
            name: name.to_string(),
            address: section.sh_addr,
            offset: section.sh_offset,
            size: section.sh_size,
            data: section_data,
        });
    }
    
    Ok(sections)
}

/// Extract and demangle symbols from ELF
fn extract_symbols(elf: &Elf) -> Result<Vec<Symbol>> {
    let mut symbols = Vec::new();
    
    for sym in elf.dynsyms.iter() {
        let name = elf.dynstrtab
            .get_at(sym.st_name)
            .unwrap_or("<unknown>");
        
        if name.is_empty() {
            continue;
        }
        
        // Demangle C++ symbol
        let demangled = demangle_symbol(name);
        
        let symbol_type = match sym.st_type() {
            goblin::elf::sym::STT_FUNC => SymbolType::Function,
            goblin::elf::sym::STT_OBJECT => SymbolType::Object,
            goblin::elf::sym::STT_NOTYPE => SymbolType::NoType,
            goblin::elf::sym::STT_SECTION => SymbolType::Section,
            goblin::elf::sym::STT_FILE => SymbolType::File,
            _ => SymbolType::Unknown,
        };
        
        symbols.push(Symbol {
            name: name.to_string(),
            demangled,
            address: sym.st_value,
            size: sym.st_size,
            symbol_type,
        });
    }
    
    // Also include regular symbols
    for sym in elf.syms.iter() {
        let name = elf.strtab
            .get_at(sym.st_name)
            .unwrap_or("<unknown>");
        
        if name.is_empty() {
            continue;
        }
        
        // Skip if already in dynsyms
        if symbols.iter().any(|s| s.name == name) {
            continue;
        }
        
        let demangled = demangle_symbol(name);
        
        let symbol_type = match sym.st_type() {
            goblin::elf::sym::STT_FUNC => SymbolType::Function,
            goblin::elf::sym::STT_OBJECT => SymbolType::Object,
            _ => SymbolType::Unknown,
        };
        
        symbols.push(Symbol {
            name: name.to_string(),
            demangled,
            address: sym.st_value,
            size: sym.st_size,
            symbol_type,
        });
    }
    
    // Sort by address
    symbols.sort_by_key(|s| s.address);
    
    Ok(symbols)
}

/// Demangle a C++ symbol name
fn demangle_symbol(name: &str) -> String {
    if let Ok(sym) = CppSymbol::new(name) {
        sym.to_string()
    } else {
        name.to_string()
    }
}

