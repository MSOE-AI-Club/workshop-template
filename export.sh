#!/bin/bash
files=("embeddings.py" "webscraping.py")

for file in "${files[@]}"; do
  without_extension="${file%.*}"
  marimo export html-wasm "$file" -o ./"$without_extension".html --mode edit
done