#!/bin/bash
#
#SBATCH --job-name=test
#SBATCH --output=test_%j.out
#SBATCH --error=test_%j.err
#SBATCH --time=00:01:00
#SBATCH --qos=gpu-small
#SBATCH --gres=gpu:1

cd /storage/ukp/work/ibrahim1

# Activate your Python 3.11 virtual environment
source /storage/ukp/work/ibrahim1/python_env/bin/activate

echo "Using Python: $(which python)"
echo "Python version: $(python --version)"

# Test ollama import first
python -c "
try:
    import ollama
    print('✓ ollama imported successfully')
except ImportError as e:
    print('✗ ollama import failed:', e)
    exit(1)
"
# Slurm -> local model
#  Ollama -> UKP vpn
# Test the actual Ollama API
python -c "
from ollama import Client
print('Testing Ollama API...')
client = Client(host='http://10.167.31.201:11434/')
print('Ollama client created successfully')
try:
    response = client.chat(
        model='qwen2.5:32b',
        messages=[
            {
                'role': 'user',
                'content': 'Hello, how are you?',
            },
        ],
    )
    print('Response:', response['message']['content'])
    print('✓ Ollama test successful')
except Exception as e:
    print('✗ Ollama test failed:', e)
"

echo "Job finished at $(date)"