#!/bin/bash

# Get the directory of the script
SCRIPT_DIR=$(dirname "$(realpath "$0")")

echo $SCRIPT_DIR

# Base URL for downloading original data
BASE_URL="https://huggingface.co/datasets/karpathy/llmc-starter-pack/resolve/main/"

# Directory paths based on script location
SAVE_DIR_PARENT="$SCRIPT_DIR/.."
SAVE_DIR_TINY="$SCRIPT_DIR/data/tinyshakespeare"
SAVE_DIR_HELLA="$SCRIPT_DIR/data/hellaswag"

# Create the directories if they don't exist
mkdir -p "$SAVE_DIR_TINY"
mkdir -p "$SAVE_DIR_HELLA"

# Files to download (original .bin files for preprocessing)
FILES=(
    "gpt2_tokenizer.bin"            # Tokenizer file (assuming WordPiece compatible)
    "tiny_shakespeare_train.bin"    # Tiny Shakespeare training data
    "tiny_shakespeare_val.bin"      # Tiny Shakespeare validation data
    "hellaswag_val.bin"             # Hellaswag validation data
)

# Function to download files to the appropriate directory
download_file() {
    local FILE_NAME="$1"
    local FILE_URL="${BASE_URL}${FILE_NAME}?download=true"
    local FILE_PATH

    # Determine the save directory based on the file name
    if [[ "$FILE_NAME" == tiny_shakespeare* ]]; then
        FILE_PATH="${SAVE_DIR_TINY}/${FILE_NAME}"
    elif [[ "$FILE_NAME" == hellaswag* ]]; then
        FILE_PATH="${SAVE_DIR_HELLA}/${FILE_NAME}"
    else
        FILE_PATH="${SAVE_DIR_PARENT}/${FILE_NAME}"
    fi

    # Download the file
    curl -s -L -o "$FILE_PATH" "$FILE_URL"
    if [ $? -eq 0 ]; then
        echo "Downloaded $FILE_NAME to $FILE_PATH"
    else
        echo "Failed to download $FILE_NAME"
    fi
}

# Export the function so it's available in subshells
export -f download_file

# Generate download commands
download_commands=()
for FILE in "${FILES[@]}"; do
    download_commands+=("download_file \"$FILE\"")
done

# Function to manage parallel jobs in increments of a given size
run_in_parallel() {
    local batch_size="$1"
    shift
    local i=0
    local command

    for command in "$@"; do
        eval "$command" &
        ((i = (i + 1) % batch_size))
        if [ "$i" -eq 0 ]; then
            wait
        fi
    done

    # Wait for any remaining jobs to finish
    wait
}

# Run the download commands in parallel in batches of 6
run_in_parallel 6 "${download_commands[@]}"

echo "All files downloaded and saved in their respective directories"

# Preprocessing step to convert .bin files to .npy classification format
echo "Preprocessing downloaded data into .npy format for BERT classification..."

# Python script embedded in Bash to preprocess data
python3 - <<EOF
import numpy as np
import os

SCRIPT_DIR = "$SCRIPT_DIR"
SAVE_DIR_TINY = "$SAVE_DIR_TINY"
SAVE_DIR_HELLA = "$SAVE_DIR_HELLA"

def preprocess_bin_to_npy(input_file, output_file, sequence_length=128, num_classes=2):
    with open(input_file, "rb") as f:
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        tokens = np.frombuffer(f.read(), dtype=np.uint16)

    # Split tokens into chunks and assign dummy labels (e.g., binary classification)
    data = []
    for i in range(0, len(tokens) - sequence_length, sequence_length):
        chunk = tokens[i:i+sequence_length].tolist()
        input_ids = chunk + [0] * (sequence_length - len(chunk))  # Pad to sequence_length
        attention_mask = [1] * len(chunk) + [0] * (sequence_length - len(chunk))
        # Dummy label: alternate between 0 and 1 (replace with real labels if available)
        label = i // sequence_length % num_classes
        data.append({'input_ids': input_ids, 'attention_mask': attention_mask, 'label': label})

    np.save(output_file, data)
    print(f"Preprocessed {input_file} to {output_file}")

# Preprocess each downloaded .bin file
preprocess_bin_to_npy(
    f"{SAVE_DIR_TINY}/tiny_shakespeare_train.bin",
    f"{SAVE_DIR_TINY}/tiny_shakespeare_train_class.npy"
)
preprocess_bin_to_npy(
    f"{SAVE_DIR_TINY}/tiny_shakespeare_val.bin",
    f"{SAVE_DIR_TINY}/tiny_shakespeare_val_class.npy"
)
preprocess_bin_to_npy(
    f"{SAVE_DIR_HELLA}/hellaswag_val.bin",
    f"{SAVE_DIR_HELLA}/hellaswag_val_class.npy"
)
EOF

echo "Preprocessing complete. Data saved in .npy format."

