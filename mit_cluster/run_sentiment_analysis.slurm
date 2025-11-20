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
export GROQ_API_KEYS="your-groq-api-key-1,your-groq-api-key-2,your-groq-api-key-3,your-groq-api-key-4,your-groq-api-key-5"
export GROQ_MODEL="llama-3.1-8b-instant"

echo "Environment variables set"
echo "Starting Python script..."

# Run the sentiment analysis pipeline with unbuffered output
python -u main_project/initial_test/llm_text/main.py

echo "=== Job finished at $(date) ==="
