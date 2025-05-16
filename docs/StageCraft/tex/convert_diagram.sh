#!/bin/bash

# Compile the TEX file to PDF
pdflatex stage_cycle_basic.tex

# Check if PDF was created successfully
if [ ! -f "stage_cycle_basic.pdf" ]; then
    echo "Error: PDF compilation failed"
    exit 1
fi

# Convert PDF to PNG using pdftoppm (if available)
if command -v pdftoppm &> /dev/null; then
    pdftoppm -png -r 300 stage_cycle_basic.pdf stage_cycle_basic
    mv stage_cycle_basic-1.png stage_cycle_basic.png
    echo "Converted to PNG using pdftoppm"
# Fall back to ImageMagick if pdftoppm is not available
elif command -v convert &> /dev/null; then
    convert -density 300 -background white -alpha remove stage_cycle_basic.pdf stage_cycle_basic.png
    echo "Converted to PNG using ImageMagick"
else
    echo "Error: Neither pdftoppm nor ImageMagick is installed"
    exit 1
fi

# Clean up temporary files
rm -f stage_cycle_basic.aux stage_cycle_basic.log

echo "Conversion complete: stage_cycle_basic.png created" 