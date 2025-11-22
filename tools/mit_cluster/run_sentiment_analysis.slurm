#!/bin/bash
#SBATCH -J sentiment_analysis
#SBATCH -o sentiment_analysis.out
#SBATCH -e sentiment_analysis.err
#SBATCH -t 5:00:00          # 5 hours (matches partition limit)
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32   # Adjust based on your needs
#SBATCH --mem=32G           # Adjust based on your needs

# Using system Python 3.11 (no module load needed)

cd $SLURM_SUBMIT_DIR

# Add debug output
echo "=== Job started at $(date) ==="
echo "Working directory: $(pwd)"
echo "Python version: $(python --version)"

source venv/bin/activate
echo "Virtual environment activated"

# Set environment variables for LLM API
# Using Groq (FREE fast Llama API) with multiple keys for rotation
# NOTE: Set GROQ_API_KEYS environment variable before running this script
# Example: export GROQ_API_KEYS="key1,key2,key3"
# Get free keys at https://console.groq.com
if [ -z "$GROQ_API_KEYS" ]; then
    echo "ERROR: GROQ_API_KEYS environment variable is not set!"
    echo "Please set it before submitting this job:"
    echo "  export GROQ_API_KEYS='your-key1,your-key2,...'"
    exit 1
fi
export GROQ_MODEL="llama-3.1-8b-instant"

echo "Environment variables set"
echo "Starting Python script..."

# Run the sentiment analysis pipeline with unbuffered output
python -u main_project/initial_test/llm_text/main.py

echo "=== Job finished at $(date) ==="
