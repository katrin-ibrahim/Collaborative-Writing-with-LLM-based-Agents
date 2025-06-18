#!/usr/bin/env python3
"""
Test script to verify Ollama API connectivity on SLURM cluster.
"""

import os
import sys
import time

def test_ollama_connection() -> bool:
    """Test connection to Ollama API server."""
    try:
        from ollama import Client
        
        # Get Ollama host from environment or use default
        host = os.getenv("OLLAMA_HOST", "http://10.167.31.201:11434")
        print(f"Testing connection to Ollama at: {host}")
        
        # Create client
        client = Client(host=host)
        
        # Test with a simple chat
        print("Sending test message...")
        start_time = time.time()
        
        response = client.chat(
            model="qwen2.5:32b",
            messages=[
                {
                    "role": "user",
                    "content": "Hello, this is a test from the SLURM cluster. Please respond with 'Connection successful'.",
                }
            ],
        )
        
        elapsed = time.time() - start_time
        print(f"Response received in {elapsed:.2f} seconds")
        print(f"Model response: {response['message']['content']}")
        
        # Verify we got a reasonable response
        if response and response.get('message') and response['message'].get('content'):
            print("✓ Ollama connection test PASSED")
            return True
        else:
            print("✗ Ollama connection test FAILED: Empty response")
            return False
            
    except ImportError:
        print("✗ ERROR: ollama package not installed")
        print("Install with: pip install ollama")
        return False
        
    except Exception as e:
        print(f"✗ Ollama connection test FAILED: {e}")
        return False

def test_model_availability():
    """Test which models are available on the Ollama server."""
    try:
        from ollama import Client
        
        host = os.getenv("OLLAMA_HOST", "http://10.167.31.201:11434")
        client = Client(host=host)
        
        print("\nTesting model availability...")
        
        # List of models to test
        test_models = ["qwen2.5:32b", "llama3.1:8b", "llama3.1:70b"]
        
        for model in test_models:
            try:
                print(f"Testing model: {model}")
                response = client.chat(
                    model=model,
                    messages=[{"role": "user", "content": "Hi"}],
                )
                if response:
                    print(f"  ✓ {model} - Available")
                else:
                    print(f"  ✗ {model} - No response")
            except Exception as e:
                print(f"  ✗ {model} - Error: {e}")
                
    except Exception as e:
        print(f"Model availability test failed: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("Ollama Connection Test for SLURM Cluster")
    print("=" * 60)
    
    # Test basic connection
    success = test_ollama_connection()
    
    # Test model availability
    if success:
        test_model_availability()
    
    print("=" * 60)
    
    if success:
        print("All tests passed! Ready to run experiments.")
        sys.exit(0)
    else:
        print("Tests failed! Check Ollama server and network connectivity.")
        sys.exit(1)