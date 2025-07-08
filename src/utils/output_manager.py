
import shutil
from pathlib import Path
import logging

from utils.data_models import Article

logger = logging.getLogger(__name__)


class OutputManager:
    """Unified output management for all baseline methods."""
    
    def __init__(self, base_output_dir: str, debug_mode: bool = False):
        self.base_dir = Path(base_output_dir)
        self.debug_mode = debug_mode
        
        # Create directory structure
        self.articles_dir = self.base_dir / "articles"
        self.articles_dir.mkdir(parents=True, exist_ok=True)
        
        if self.debug_mode:
            self.debug_dir = self.base_dir / "debug"
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Debug mode enabled - intermediate files will be saved")
    
    def save_article(self, article: Article, method: str) -> Path:
        """Save article to standardized location."""
        filename = f"{method}_{article.title.replace(' ', '_').replace('/', '_')}.md"
        filepath = self.articles_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(article.content)
            logger.info(f"Saved {method} article: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save article {filepath}: {e}")
            raise
    
    def setup_storm_output_dir(self, topic: str) -> str:
        """Setup STORM output directory based on debug mode."""
        if self.debug_mode:
            # Use debug directory for STORM intermediate files
            storm_dir = self.debug_dir / "storm" / topic.replace(" ", "_").replace("/", "_")
            storm_dir.mkdir(parents=True, exist_ok=True)
            return str(storm_dir)
        else:
            # Use temporary directory that gets cleaned up
            temp_dir = self.base_dir / "temp_storm" / topic.replace(" ", "_").replace("/", "_")
            temp_dir.mkdir(parents=True, exist_ok=True)
            return str(temp_dir)
    
    def cleanup_storm_temp(self, topic: str):
        """Clean up temporary STORM files if not in debug mode."""
        if not self.debug_mode:
            temp_dir = self.base_dir / "temp_storm" / topic.replace(" ", "_").replace("/", "_")
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                logger.debug(f"Cleaned up temporary STORM files for {topic}")