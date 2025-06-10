# quick_fix_freshwiki.py - Run this to fix your FreshWiki dataset immediately

import json
import os
import re
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quick_download_freshwiki():
    """Quick download and setup of FreshWiki dataset."""
    
    try:
        from datasets import load_dataset
        logger.info("Datasets library found")
    except ImportError:
        logger.error("Installing datasets library...")
        os.system("pip install datasets")
        from datasets import load_dataset
    
    # Setup directories
    output_path = Path("data/freshwiki")
    json_dir = output_path / "json"
    txt_dir = output_path / "txt"
    
    json_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Downloading FreshWiki from HuggingFace...")
    
    # Download dataset
    dataset = load_dataset("EchoShao8899/FreshWiki", split="train")
    logger.info(f"Found {len(dataset)} entries")
    
    # Process entries
    count = 0
    for entry in dataset:
        if count >= 20:  # Limit for quick setup
            break
            
        text = entry.get("text", "").strip()
        if not text:
            continue
            
        lines = text.split('\n')
        title = lines[0].strip()
        
        if not title or len(title) > 100:
            continue
        
        # Clean filename
        filename = re.sub(r'[^\w\-_\(\)]', '_', title.replace(' ', '_'))
        filename = re.sub(r'_+', '_', filename).strip('_')
        
        if not filename:
            continue
        
        # Create JSON
        json_content = {
            "title": title,
            "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
            "summary": "Article content from FreshWiki dataset",
            "content": [
                {
                    "section_title": section.lstrip('#').strip(),
                    "content": "Content for " + section.lstrip('#').strip(),
                    "subsections": []
                }
                for line in lines[1:20]  # First 20 lines
                for section in [line] if line.strip().startswith('#')
            ] or [
                {
                    "section_title": "Introduction",
                    "content": text[:500] + "...",
                    "subsections": []
                }
            ]
        }
        
        # Save files
        json_file = json_dir / f"{filename}.json"
        txt_file = txt_dir / f"{filename}.txt"
        
        try:
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(json_content, f, indent=2, ensure_ascii=False)
            
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(text)
            
            count += 1
            logger.info(f"Created files for: {title}")
            
        except Exception as e:
            logger.warning(f"Failed to create files for {title}: {e}")
    
    logger.info(f"Created {count} FreshWiki entries")
    return count > 0

if __name__ == "__main__":
    print("Quick FreshWiki Setup")
    print("====================")
    
    if quick_download_freshwiki():
        print("\n✓ FreshWiki dataset setup complete!")
        print("You can now run: python src/main_storm.py --method all --num_topics 5")
    else:
        print("\n✗ Setup failed. Please check the error messages above.")