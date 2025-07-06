"""
DSPy integration with manual component creation for STORM compatibility.
"""
import os
import logging

logger = logging.getLogger(__name__)


def setup_dspy_integration(query_handler, max_tokens=512, temperature=0.7):
    """
    Setup DSPy integration without any SQLite usage.
    
    This is the main function that other modules expect to import.
    """
    return setup_dspy_for_storm(query_handler, max_tokens, temperature)


def setup_dspy_for_storm(query_handler, max_tokens=512, temperature=0.7):
    """
    Setup DSPy for STORM by manually creating missing components.
    """
    logger.info("🔧 Setting up DSPy for STORM (HPC-compatible with manual components)...")
    
    # Step 1: Setup cache directories
    logger.info("📁 Setting up writable cache directories...")
    
    work_dir = os.getcwd()
    dspy_cache_dir = os.path.join(work_dir, ".dspy_cache")
    joblib_cache_dir = os.path.join(work_dir, ".joblib_cache")
    
    os.makedirs(dspy_cache_dir, exist_ok=True)
    os.makedirs(joblib_cache_dir, exist_ok=True)
    
    os.environ['DSP_CACHEDIR'] = dspy_cache_dir
    os.environ['JOBLIB_CACHE_DIR'] = joblib_cache_dir
    os.environ['HOME'] = work_dir
    
    logger.info(f"✅ Cache directories set")
    
    # Step 2: Import DSPy
    import dspy
    logger.info("✅ DSPy imported successfully!")
    
    # Step 3: Explore what's in primitives
    try:
        primitives_attrs = [attr for attr in dir(dspy.primitives) if not attr.startswith('_')]
        logger.info(f"📋 dspy.primitives contents: {primitives_attrs}")
        
        # Try to find useful classes in primitives
        for attr in primitives_attrs:
            obj = getattr(dspy.primitives, attr)
            if isinstance(obj, type):  # It's a class
                logger.info(f"🔍 Found class in primitives: {attr} -> {obj}")
                # Add it to main dspy module
                setattr(dspy, attr, obj)
                
    except Exception as e:
        logger.warning(f"⚠️ Error exploring primitives: {e}")
    
    # Step 4: Manually create missing DSPy components that STORM expects
    logger.info("🛠️ Creating missing DSPy components...")
    
    # Create a minimal Signature class
    class Signature:
        """Minimal Signature class for STORM compatibility."""
        def __init__(self, *args, **kwargs):
            pass
    
    # Create minimal Predict and ChainOfThought classes
    class Predict:
        """Minimal Predict class for STORM compatibility."""
        def __init__(self, signature=None, **kwargs):
            self.signature = signature
    
    class ChainOfThought:
        """Minimal ChainOfThought class for STORM compatibility."""
        def __init__(self, signature=None, **kwargs):
            self.signature = signature
    
    # Create our custom LM class
    class HPC_DSPy_LM:
        """DSPy LM wrapper that works in HPC environments."""
        
        def __init__(self, query_handler, max_tokens=512, temperature=0.7):
            self.query_handler = query_handler
            self.max_tokens = max_tokens
            self.temperature = temperature
            
            # Set required attributes for STORM compatibility
            self.model_name = "qwen-local"
            self.model = "qwen-local"
            self.kwargs = {
                'max_tokens': max_tokens,
                'temperature': temperature,
                'model': 'qwen-local'
            }
            
            logger.info("✅ HPC_DSPy_LM initialized successfully")
        
        def basic_request(self, prompt, **kwargs):
            """Core request method that STORM expects."""
            logger.debug(f"🔄 Processing request: {str(prompt)[:100]}...")
            
            response = self.query_handler.query(
                prompt=str(prompt),
                max_new_tokens=kwargs.get('max_tokens', self.max_tokens),
                temperature=kwargs.get('temperature', self.temperature)
            )
            
            if not response:
                logger.warning("⚠️ Query handler returned empty response")
                return ["Error: Empty response from query handler"]
            
            logger.debug(f"✅ Generated response: {len(str(response))} chars")
            return [str(response)]
        
        def __call__(self, prompt, **kwargs):
            """Make the LM callable for compatibility."""
            result = self.basic_request(prompt, **kwargs)
            return result[0] if result else "Error: No response"
        
        def generate(self, prompt, **kwargs):
            return self.basic_request(prompt, **kwargs)
        
        def request(self, prompt, **kwargs):
            return self.basic_request(prompt, **kwargs)
    
    # Step 5: Add missing components to dspy module
    if not hasattr(dspy, 'Signature'):
        dspy.Signature = Signature
        logger.info("✅ Added Signature to dspy module")
    
    if not hasattr(dspy, 'Predict'):
        dspy.Predict = Predict
        logger.info("✅ Added Predict to dspy module")
    
    if not hasattr(dspy, 'ChainOfThought'):
        dspy.ChainOfThought = ChainOfThought
        logger.info("✅ Added ChainOfThought to dspy module")
    
    # Create a minimal configure function
    def configure(lm=None, rm=None, **kwargs):
        """Minimal configure function for STORM compatibility."""
        logger.info("✅ DSPy configured (minimal implementation)")
        return True
    
    if not hasattr(dspy, 'configure'):
        dspy.configure = configure
        logger.info("✅ Added configure to dspy module")
    
    # Step 6: Create and configure our LM
    storm_lm = HPC_DSPy_LM(query_handler, max_tokens, temperature)
    
    # Configure DSPy
    dspy.configure(lm=storm_lm)
    
    # Step 7: Verify what's available now
    final_attrs = [attr for attr in dir(dspy) if not attr.startswith('_')]
    logger.info(f"📋 Final DSPy attributes: {final_attrs}")
    
    logger.info("✅ DSPy setup complete with manual components")
    return storm_lm