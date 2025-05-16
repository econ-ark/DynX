#!/usr/bin/env python3
"""
Script to integrate individual section files into a complete modcraft-summary.md document.
"""

import os
import re
from datetime import datetime

# Configuration
SECTIONS_DIR = "docs/sections"
OUTPUT_FILE = "modcraft-summary.md"
SECTION_FILES = [
    "theory.md",
    "modularity.md",
    "graphs.md",
    "computational_elements.md",
    "circuit_structure.md",
    "theoretical_concordance.md",
    "examples.md"
]

def read_file_content(file_path):
    """Read content from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

def write_file_content(file_path, content):
    """Write content to a file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f"Successfully wrote to {file_path}")
        return True
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")
        return False

def generate_header():
    """Generate the document header with metadata."""
    today = datetime.now().strftime("%B %d, %Y")
    header = f"""---
title: "ModCraft: A Framework for Dynamic Programming Models"
date: "{today}"
author: "ModCraft Development Team"
---

# ModCraft Documentation

*A modern framework for building, solving, and analyzing dynamic programming models.*

**Version:** 1.0.0  
**Last Updated:** {today}

---

## Table of Contents

1. [Theory and Mathematical Framework](#1-theory-and-mathematical-framework)
2. [Modularity and Component Design](#2-modularity-and-component-design)
3. [Graph Structure and Connectivity](#3-graph-structure-and-connectivity)
4. [Computational Elements](#4-computational-elements)
5. [Circuit Structure](#5-circuit-structure)
6. [Theoretical to Computational Concordance](#6-theoretical-to-computational-concordance)
7. [Examples and Case Studies](#7-examples-and-case-studies)

---

"""
    return header

def main():
    """Main function to integrate all sections into a single document."""
    # Generate the header
    full_content = generate_header()
    
    # Read and append each section file
    for section_file in SECTION_FILES:
        file_path = os.path.join(SECTIONS_DIR, section_file)
        section_content = read_file_content(file_path)
        
        if section_content:
            # Add the section content (without adjusting heading levels)
            full_content += section_content + "\n\n"
    
    # Write the complete document
    success = write_file_content(OUTPUT_FILE, full_content)
    
    if success:
        print(f"Successfully created {OUTPUT_FILE}")
    else:
        print(f"Failed to create {OUTPUT_FILE}")

if __name__ == "__main__":
    main() 