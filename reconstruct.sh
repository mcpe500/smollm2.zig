#!/bin/bash
# Reconstruct smollm2.sm2 from compressed chunks
if [ -f smollm2.sm2 ]; then
    echo "smollm2.sm2 already exists"
    exit 0
fi

echo "Reconstructing model from chunks..."
cat smollm2_part_* > smollm2.sm2.gz
gunzip -f smollm2.sm2.gz
echo "Done. Size: $(du -h smollm2.sm2 | cut -f1)"