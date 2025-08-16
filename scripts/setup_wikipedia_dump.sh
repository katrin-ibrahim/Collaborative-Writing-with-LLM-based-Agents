#!/bin/bash

# Wikipedia Dump Setup Script
# Downloads and processes Wikipedia dump for local retrieval managers

set -e

DUMP_DIR="data/wiki_dump"
DOWNLOAD_DIR="$DUMP_DIR/raw"
OUTPUT_DIR="$DUMP_DIR/text"

echo "Setting up Wikipedia dump for local retrieval..."

# Create directories
mkdir -p "$DOWNLOAD_DIR"
mkdir -p "$OUTPUT_DIR"

# Step 1: Download Wikipedia dump (if not already present)
DUMP_FILE="$DOWNLOAD_DIR/enwiki-latest-pages-articles.xml.bz2"
if [ ! -f "$DUMP_FILE" ]; then
    echo "Downloading Wikipedia dump (~20GB)..."
    cd "$DOWNLOAD_DIR"

    # Use curl on macOS, wget on Linux
    if command -v curl >/dev/null 2>&1; then
        echo "Using curl to download..."
        curl -L -o enwiki-latest-pages-articles.xml.bz2 https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
    elif command -v wget >/dev/null 2>&1; then
        echo "Using wget to download..."
        wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
    else
        echo "Error: Neither curl nor wget found. Please install one of them:"
        echo "  macOS: curl is pre-installed, or 'brew install wget'"
        echo "  Linux: 'apt-get install wget' or 'yum install wget'"
        exit 1
    fi

    cd - > /dev/null
else
    echo "Wikipedia dump already downloaded: $DUMP_FILE"
fi

# Step 2: Install required dependencies
echo "Installing required dependencies..."
pip install mwxml

# Step 3: Verify the downloaded file
echo "Verifying downloaded file..."
if [ ! -f "$DUMP_FILE" ]; then
    echo "Error: Dump file not found: $DUMP_FILE"
    exit 1
fi

# Check if file is valid bz2
if ! bzip2 -t "$DUMP_FILE" 2>/dev/null; then
    echo "Error: Downloaded file is corrupted or not a valid bz2 file"
    echo "File size: $(ls -lh "$DUMP_FILE" | awk '{print $5}')"
    echo "Please delete the file and try downloading again"
    exit 1
fi

echo "File verification passed"

# Step 4: Extract articles to JSON format
if [ ! -d "$OUTPUT_DIR/A00" ]; then
    echo "Extracting Wikipedia articles to JSON format..."
    echo "This may take several hours for the full dump..."

    # Use our custom processor
    python scripts/process_wikipedia_dump.py "$DUMP_FILE" "$OUTPUT_DIR"
else
    echo "Wikipedia articles already extracted: $OUTPUT_DIR"
fi

echo "Wikipedia dump setup complete!"
echo "Articles available in: $OUTPUT_DIR"
echo ""
echo "You can now use the retrieval managers:"
echo "- BM25WikiRM: Keyword search"
echo "- FAISSWikiRM: Semantic search"
echo ""
echo "Test with:"
echo "python -c \"from src.retrieval.factory import create_retrieval_manager; rm = create_retrieval_manager(rm_type='bm25_wiki'); print('Success!')\""
