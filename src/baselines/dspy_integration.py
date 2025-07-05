"""
DSPy integration and configuration for STORM.
"""
import sys
import logging

logger = logging.getLogger(__name__)

def setup_sqlite_fix():
    """Fix sqlite3 compatibility issues."""
    try:
        import pysqlite3 as sqlite3
        sys.modules['sqlite3'] = sqlite3
        logger.info("✅ Using pysqlite3 as sqlite3 replacement")
        return True
    except ImportError:
        logger.warning("⚠️ pysqlite3 not available")
        return False

def configure_dspy(query_handler, max_tokens=512, temperature=0.7):
    """Configure DSPy to use our local model."""
    try:
        import dspy
        logger.info("🔧 Configuring DSPy...")
        
        class DSPyLocalLLM(dspy.LM):
            def __init__(self, query_handler, max_tokens=512, temperature=0.7):
                self.query_handler = query_handler
                self.max_tokens = max_tokens
                self.temperature = temperature
                super().__init__("qwen-local")
            
            def basic_request(self, prompt, **kwargs):
                try:
                    response = self.query_handler.query(
                        prompt=str(prompt),
                        max_new_tokens=kwargs.get('max_tokens', self.max_tokens),
                        temperature=kwargs.get('temperature', self.temperature)
                    )
                    return [response] if response else ["Error: No response"]
                except Exception as e:
                    return [f"Error: {str(e)}"]
            
            def __call__(self, prompt, **kwargs):
                return self.basic_request(prompt, **kwargs)
        
        dspy_lm = DSPyLocalLLM(query_handler, max_tokens, temperature)
        dspy.configure(lm=dspy_lm)
        logger.info("✅ DSPy configured successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to configure DSPy: {e}")
        return False

def setup_dspy_integration(query_handler, max_tokens=512, temperature=0.7):
    """Complete DSPy integration setup."""
    logger.info("🚀 Setting up DSPy integration...")
    
    if not setup_sqlite_fix():
        logger.error("Failed to fix sqlite3")
        return False
    
    if not configure_dspy(query_handler, max_tokens, temperature):
        logger.error("Failed to configure DSPy")
        return False
    
    logger.info("✅ DSPy integration setup complete")
    return True
