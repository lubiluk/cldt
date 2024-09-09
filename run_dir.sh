#!/bin/bash

# Check if directory is passed as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

# Directory from the command line argument
DIR="$1"

# Iterate through each file in the directory
for file in "$DIR"/*
do
    # Check if it's a regular file (not a directory)
    if [ -f "$file" ]; then
        # Run the program with the file as an argument
        ./spython.sh train_single.py -c "$file"
    fi
done
