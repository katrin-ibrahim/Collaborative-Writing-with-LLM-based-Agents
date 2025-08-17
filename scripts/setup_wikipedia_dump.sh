#!/bin/bash

# Enhanced Wikipedia Dump Setup Script
# Downloads and processes Wikipedia dump with proper configuration for multiple output files

set -e  # Exit on any error

# Configuration
WIKI_DUMP_URL="https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2"
DUMP_DIR="data/wiki_dump"
RAW_FILE="$DUMP_DIR/raw/enwiki-latest-pages-articles.xml.bz2"
EXTRACTED_FILE="$DUMP_DIR/raw/enwiki-latest-pages-articles.xml"
OUTPUT_DIR="$DUMP_DIR/text"
LOG_FILE="$DUMP_DIR/processing.log"

# WikiExtractor configuration for proper file splitting
MAX_FILE_SIZE="10M"      # Maximum size per output file (ensures multiple files)
MIN_ARTICLE_SIZE="500"   # Minimum article size in characters
PROCESSES="4"            # Number of parallel processes

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."

    # Check Python
    if ! command -v python3 &> /dev/null; then
        error "Python3 is required but not installed"
    fi

    # Check pip
    if ! command -v pip3 &> /dev/null; then
        error "pip3 is required but not installed"
    fi

    # Check disk space (need at least 50GB for full Wikipedia)
    available_space=$(df . | tail -1 | awk '{print $4}')
    required_space=52428800  # 50GB in KB

    if [ "$available_space" -lt "$required_space" ]; then
        warning "Low disk space detected. Wikipedia dump requires ~50GB"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi

    success "Prerequisites check passed"
}

# Install WikiExtractor
install_wikiextractor() {
    log "Installing/updating WikiExtractor..."

    if pip3 show wikiextractor &> /dev/null; then
        log "WikiExtractor already installed, updating..."
        pip3 install --upgrade wikiextractor
    else
        log "Installing WikiExtractor..."
        pip3 install wikiextractor
    fi

    # Verify installation
    if ! python3 -c "import wikiextractor" &> /dev/null; then
        error "WikiExtractor installation failed"
    fi

    success "WikiExtractor installed successfully"
}

# Create directory structure
setup_directories() {
    log "Setting up directory structure..."

    mkdir -p "$DUMP_DIR/raw"
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$(dirname "$LOG_FILE")"

    success "Directories created"
}

# Download Wikipedia dump
download_dump() {
    log "Checking for existing Wikipedia dump..."

    if [ -f "$RAW_FILE" ]; then
        log "Raw dump file already exists: $RAW_FILE"

        # Check if file is complete (basic size check)
        file_size=$(stat -f%z "$RAW_FILE" 2>/dev/null || stat -c%s "$RAW_FILE" 2>/dev/null || echo 0)
        min_expected_size=15000000000  # ~15GB minimum for recent dumps

        if [ "$file_size" -lt "$min_expected_size" ]; then
            warning "Dump file seems incomplete (size: $file_size bytes)"
            read -p "Re-download? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                rm -f "$RAW_FILE"
            fi
        fi
    fi

    if [ ! -f "$RAW_FILE" ]; then
        log "Downloading Wikipedia dump (this will take a while)..."
        log "URL: $WIKI_DUMP_URL"

        # Use wget with resume capability
        if command -v wget &> /dev/null; then
            wget -c -O "$RAW_FILE" "$WIKI_DUMP_URL" 2>&1 | tee -a "$LOG_FILE"
        elif command -v curl &> /dev/null; then
            curl -C - -o "$RAW_FILE" "$WIKI_DUMP_URL" 2>&1 | tee -a "$LOG_FILE"
        else
            error "Neither wget nor curl is available for downloading"
        fi

        success "Download completed"
    else
        success "Using existing dump file"
    fi
}

# Extract bz2 file
extract_dump() {
    log "Checking for extracted dump..."

    if [ -f "$EXTRACTED_FILE" ]; then
        log "Extracted file already exists: $EXTRACTED_FILE"
        return
    fi

    if [ ! -f "$RAW_FILE" ]; then
        error "Raw dump file not found: $RAW_FILE"
    fi

    log "Extracting Wikipedia dump (this will take some time)..."

    # Extract with progress indication
    bunzip2 -ckv "$RAW_FILE" > "$EXTRACTED_FILE" 2>&1 | tee -a "$LOG_FILE"

    success "Extraction completed"
}

# Process with WikiExtractor
process_dump() {
    log "Processing Wikipedia dump with WikiExtractor..."

    # Check if output already exists
    if [ -d "$OUTPUT_DIR" ] && [ "$(ls -A $OUTPUT_DIR)" ]; then
        log "Output directory is not empty: $OUTPUT_DIR"
        read -p "Remove existing output and reprocess? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$OUTPUT_DIR"/*
        else
            log "Using existing processed files"
            return
        fi
    fi

    if [ ! -f "$EXTRACTED_FILE" ]; then
        error "Extracted dump file not found: $EXTRACTED_FILE"
    fi

    log "WikiExtractor configuration:"
    log "  Max file size: $MAX_FILE_SIZE"
    log "  Processes: $PROCESSES"
    log "  Output directory: $OUTPUT_DIR"

    # Try different ways to run WikiExtractor
    if command -v wikiextractor &> /dev/null; then
        WIKIEXTRACTOR_CMD="wikiextractor"
    elif python3 -c "import wikiextractor" &> /dev/null; then
        WIKIEXTRACTOR_CMD="python3 -m wikiextractor.WikiExtractor"
    elif [ -f "$HOME/Library/Python/3.9/bin/wikiextractor" ]; then
        WIKIEXTRACTOR_CMD="$HOME/Library/Python/3.9/bin/wikiextractor"
    else
        error "Cannot find WikiExtractor executable"
    fi

    log "Using WikiExtractor command: $WIKIEXTRACTOR_CMD"

    # Run WikiExtractor with proper configuration for multiple files
    $WIKIEXTRACTOR_CMD \
        --output "$OUTPUT_DIR" \
        --bytes "$MAX_FILE_SIZE" \
        --processes "$PROCESSES" \
        --json \
        --no-templates \
        --quiet \
        "$EXTRACTED_FILE" 2>&1 | tee -a "$LOG_FILE"

    success "WikiExtractor processing completed"
}

# Validate output
validate_output() {
    log "Validating WikiExtractor output..."

    # Count output files
    json_files=$(find "$OUTPUT_DIR" -name "wiki_*.json" | wc -l)

    if [ "$json_files" -eq 0 ]; then
        error "No output files found in $OUTPUT_DIR"
    fi

    log "Found $json_files output files"

    # Check file sizes
    total_size=0
    small_files=0

    for file in $(find "$OUTPUT_DIR" -name "wiki_*.json"); do
        size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo 0)
        total_size=$((total_size + size))

        if [ "$size" -lt 1000000 ]; then  # Less than 1MB
            small_files=$((small_files + 1))
        fi
    done

    log "Total output size: $(echo $total_size | awk '{print $1/1024/1024/1024 " GB"}')"

    if [ "$small_files" -gt 0 ]; then
        warning "$small_files files are smaller than 1MB (may indicate processing issues)"
    fi

    # Sample validation - check first few files for valid JSON
    log "Validating JSON format in sample files..."

    sample_files=$(find "$OUTPUT_DIR" -name "wiki_*.json" | head -3)
    validation_errors=0

    for file in $sample_files; do
        log "Checking $file..."

        # Count lines and check JSON validity
        total_lines=0
        valid_lines=0

        while IFS= read -r line; do
            if [ -n "$(echo "$line" | tr -d '[:space:]')" ]; then
                total_lines=$((total_lines + 1))

                if echo "$line" | python3 -c "import json, sys; json.load(sys.stdin)" &> /dev/null; then
                    valid_lines=$((valid_lines + 1))
                fi
            fi
        done < "$file"

        if [ "$total_lines" -eq 0 ]; then
            warning "File $file is empty"
            validation_errors=$((validation_errors + 1))
        elif [ "$valid_lines" -lt "$total_lines" ]; then
            invalid_lines=$((total_lines - valid_lines))
            warning "File $file has $invalid_lines invalid JSON lines out of $total_lines"
            validation_errors=$((validation_errors + 1))
        else
            log "File $file: $total_lines valid JSON lines"
        fi
    done

    if [ "$validation_errors" -gt 0 ]; then
        warning "Validation found issues in $validation_errors files"
        warning "Consider re-running WikiExtractor or using the enhanced data loader"
    else
        success "Output validation passed"
    fi

    # Final summary
    log "=== PROCESSING SUMMARY ==="
    log "Output files: $json_files"
    log "Total size: $(echo $total_size | awk '{print $1/1024/1024/1024 " GB"}')"
    log "Location: $OUTPUT_DIR"

    if [ "$json_files" -eq 1 ]; then
        warning "Only 1 output file generated. Consider using smaller --bytes parameter for more files"
    fi
}

# Cleanup function
cleanup() {
    log "Cleaning up temporary files..."

    # Optionally remove extracted XML (keep compressed)
    if [ -f "$EXTRACTED_FILE" ]; then
        read -p "Remove extracted XML file to save space? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -f "$EXTRACTED_FILE"
            success "Extracted XML file removed"
        fi
    fi
}

# Main execution
main() {
    log "Starting Wikipedia dump setup process..."
    log "Target directory: $DUMP_DIR"

    check_prerequisites
    install_wikiextractor
    setup_directories
    download_dump
    extract_dump
    process_dump
    validate_output
    cleanup

    success "Wikipedia dump setup completed successfully!"
    log "You can now use the enhanced data loader to access the articles"
    log "Log file: $LOG_FILE"
}

# Handle interruption
trap 'error "Process interrupted"' INT TERM

# Run main function
main "$@"
