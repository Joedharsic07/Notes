#!/bin/bash

# Create models folder if it doesn't exist
mkdir -p ./models

# URL to your LLaMA model (replace this with the actual download URL)
MODEL_URL="https://example.com/Llama-3.2-3B-Instruct-Q4_K_M.gguf"

# Path to save the model
MODEL_PATH="./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"

# Download the model
echo "Downloading LLaMA model..."
wget -O "$MODEL_PATH" "$MODEL_URL"

# Check if download was successful
if [ -f "$MODEL_PATH" ]; then
    echo "Model downloaded successfully to $MODEL_PATH"
else
    echo "Failed to download model."
fi
