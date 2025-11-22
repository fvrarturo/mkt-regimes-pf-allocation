# Step-by-Step Guide: Running Sentiment Analysis on MIT Engaging with Groq API

## 🎯 Overview

This guide walks you through setting up and running the sentiment analysis pipeline on MIT Engaging cluster using **Groq API** (FREE, fast Llama inference). The pipeline processes news articles week-by-week and generates sentiment scores for inflation, economic growth, monetary policy, and market volatility.

**Key Features:**
- ✅ **FREE** - Uses Groq API (no cost, fast Llama models)
- ✅ **Automatic API key rotation** - Handles rate limits by rotating through multiple keys
- ✅ **Incremental saving** - Saves progress after each week, can resume if interrupted
- ✅ **Unbuffered output** - Real-time progress monitoring

---

## Step 1: Get Groq API Keys (FREE)

**Groq offers free API access** with generous rate limits. Get your keys:

1. Go to https://console.groq.com
2. Sign up/login (free account)
3. Navigate to **API Keys** section
4. Create one or more API keys (recommended: 3-5 keys for rotation)

**Note**: Each key has rate limits. Using multiple keys allows automatic rotation when limits are hit.

---

## Step 2: Prepare Your Project for Upload

**On your Mac**, navigate to your project directory:

```bash
cd ~/Desktop/Agents_News
```

**Create a compressed archive** (excluding unnecessary files):

```bash
# Remove old archive if exists
rm -f agents_news.tar.gz

# Create lightweight archive (~200-300KB)
tar -czf agents_news.tar.gz \
    --exclude='venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='*.csv' \
    --exclude='*.png' \
    --exclude='*.jpg' \
    --exclude='*.svg' \
    --exclude='sentiment_scores*.csv' \
    --exclude='openai/__pycache__' \
    --exclude='openai/.git' \
    --exclude='openai/docs' \
    --exclude='openai/examples' \
    --exclude='openai/tests' \
    --exclude='openai/Makefile' \
    --exclude='openai/mkdocs.yml' \
    --exclude='openai/uv.lock' \
    --exclude='main_project/data/news_data/initial_work' \
    --exclude='main_project/data/news_data/basic_keyword' \
    --exclude='main_project/data/news_data/old_data' \
    --exclude='main_project/data/macro' \
    .

# Verify size (should be small)
ls -lh agents_news.tar.gz
```

**Note**: We're excluding:
- `venv/` - Will recreate on cluster
- `__pycache__/` - Python cache
- Large data files (CSV, PNG) - Upload separately
- Old analysis results

**But we're keeping**:
- All source code (`main_project/initial_test/llm_text/`)
- `openai/` directory (agents framework)
- `requirements.txt`
- `cluster/` directory (SLURM scripts)

---

## Step 3: Upload to MIT Engaging

**Make sure you're connected to MIT VPN** first!

**Upload the archive:**

```bash
rsync -avz --progress agents_news.tar.gz favara@orcd-login001.mit.edu:~
```

**Alternative**: If you prefer the web interface:
1. Go to https://ood.mit.edu
2. Navigate to **Files → Home Directory**
3. Click **Upload** and select `agents_news.tar.gz`

---

## Step 4: SSH into MIT Engaging

```bash
ssh favara@orcd-login001.mit.edu
```

**If this is your first time**, you'll need to:
1. Enter your Kerberos password
2. Approve Duo authentication on your device

---

## Step 5: Extract and Navigate to Project

**On the cluster:**

```bash
cd ~
tar -xzf agents_news.tar.gz
cd Agents_News
```

**Note**: macOS extended attribute warnings (`LIBARCHIVE.xattr.*`) are harmless - ignore them.

---

## Step 6: Upload Data File

**Upload the news data CSV** (required for sentiment analysis):

**On your Mac** (in a new terminal):

```bash
rsync -avz --progress ~/Desktop/Agents_News/main_project/data/news_data/full_factiva.csv \
    favara@orcd-login001.mit.edu:~/Agents_News/main_project/data/news_data/
```

**Or use Open OnDemand web interface** to upload the CSV file.

**Verify on cluster:**

```bash
ls -lh ~/Agents_News/main_project/data/news_data/full_factiva.csv
```

---

## Step 7: Set Up Python Environment

**On the cluster:**

```bash
cd ~/Agents_News

# Create virtual environment using system Python 3.11
/usr/bin/python3.11 -m venv venv

# Activate environment
source venv/bin/activate

# Verify Python version
python --version  # Should show Python 3.11.x

# Upgrade pip
pip install --upgrade pip

# Install the agents framework first (editable install)
pip install -e openai/

# Install other dependencies
pip install -r requirements.txt
```

**Verify installation:**

```bash
python -c "from agents import Agent; from pydantic import BaseModel; import pandas; print('All packages installed!')"
```

---

## Step 8: Configure Groq API Keys

**The code automatically uses Groq if API keys are provided.**

**Set your Groq API keys** (on cluster):

```bash
# Single key (simple)
export GROQ_API_KEY="gsk_your-api-key-here"

# OR multiple keys for rotation (recommended)
export GROQ_API_KEYS="gsk_key1,gsk_key2,gsk_key3,gsk_key4,gsk_key5"

# Set model (optional, defaults to llama-3.1-8b-instant)
export GROQ_MODEL="llama-3.1-8b-instant"
```

**For persistence**, add to your `~/.bashrc`:

```bash
echo 'export GROQ_API_KEYS="gsk_key1,gsk_key2,gsk_key3,gsk_key4,gsk_key5"' >> ~/.bashrc
echo 'export GROQ_MODEL="llama-3.1-8b-instant"' >> ~/.bashrc
source ~/.bashrc
```

**The code will automatically:**
- Use **Groq API** if `GROQ_API_KEY` or `GROQ_API_KEYS` is set (FREE, fast)
- Rotate through multiple keys if rate limits are hit
- Fall back to OpenAI if Groq keys are exhausted (requires `OPENAI_API_KEY`)

**Available Groq models:**
- `llama-3.1-8b-instant` (default, fastest)
- `llama-3.1-70b-versatile` (more capable, slower)
- `mixtral-8x7b-32768` (good balance)

---

## Step 9: Update SLURM Script

**The SLURM script is already configured**, but you can customize it:

```bash
nano cluster/run_sentiment_analysis.slurm
```

**Current configuration:**
- **Time limit**: 5 hours (`--t 5:00:00`)
- **CPUs**: 32 (`--cpus-per-task=32`)
- **Memory**: 32G (`--mem=32G`)
- **GPUs**: 2 (`--gres=gpu:2`) - Not used for Groq API, but reserved

**Key parameters to adjust:**
- `--cpus-per-task=32` - Increase if you need more parallel processing
- `--mem=32G` - Increase if you run out of memory
- `--t 5:00:00` - Time limit (check partition limits with `scontrol show partition`)

**The script already includes:**
- Groq API keys (update with your own keys)
- Unbuffered Python output (`python -u`) for real-time monitoring
- Debug output for tracking progress

**Important**: Update the `GROQ_API_KEYS` in the SLURM script with your own keys:

```bash
export GROQ_API_KEYS="your-key1,your-key2,your-key3,your-key4,your-key5"
```

---

## Step 10: Test the Setup (Optional)

**Before submitting a full job, test locally on the login node** (quick test only - don't run long jobs here):

```bash
cd ~/Agents_News
source venv/bin/activate

# Set API keys
export GROQ_API_KEYS="your-key1,your-key2,your-key3"
export GROQ_MODEL="llama-3.1-8b-instant"

# Run a quick test (modify main.py to process just 1-2 weeks)
python main_project/initial_test/llm_text/test_setup.py
```

**If test works**, proceed to submit the full job.

---

## Step 11: Submit the Job

**On the cluster:**

```bash
cd ~/Agents_News

# Submit the job
sbatch cluster/run_sentiment_analysis.slurm
```

**You'll see output like:**
```
Submitted batch job 12345678
```

**Save the job ID** - you'll need it for monitoring.

---

## Step 12: Monitor Your Job

**Check job status:**

```bash
squeue -u $USER
```

**Watch output in real-time:**

```bash
tail -f sentiment_analysis.out
```

**Watch errors:**

```bash
tail -f sentiment_analysis.err
```

**Exit tail**: Press `Ctrl+C`

**Check detailed job info:**

```bash
scontrol show job <JOBID>
```

**Check progress** (see which weeks are processed):

```bash
tail -100 sentiment_analysis.out | grep -E 'Processing week|✓ Approved|✗ Rejected|rotating|API key'
```

**Check sentiment scores** (updated incrementally):

```bash
wc -l main_project/initial_test/llm_text/sentiment_scores.csv
head -20 main_project/initial_test/llm_text/sentiment_scores.csv
```

---

## Step 13: Download Results

**After job completes** (or while running, since it saves incrementally), download results to your Mac:

**On your Mac:**

```bash
# Download sentiment scores
rsync -avz --progress favara@orcd-login001.mit.edu:~/Agents_News/main_project/initial_test/llm_text/sentiment_scores.csv \
    ~/Desktop/Agents_News/main_project/initial_test/llm_text/

# Download output/error files
rsync -avz --progress favara@orcd-login001.mit.edu:~/Agents_News/sentiment_analysis.out \
    ~/Desktop/Agents_News/
rsync -avz --progress favara@orcd-login001.mit.edu:~/Agents_News/sentiment_analysis.err \
    ~/Desktop/Agents_News/
```

---

## Troubleshooting

### Job fails immediately

**Check error file:**
```bash
cat sentiment_analysis.err
```

**Common issues:**
- **Missing API keys**: Verify `GROQ_API_KEYS` is set in SLURM script
- **Import errors**: Verify packages installed correctly (`pip install -r requirements.txt`)
- **File not found**: Check that `full_factiva.csv` exists in the correct location
- **No LLM configured**: Ensure `GROQ_API_KEYS` environment variable is set

### API rate limits

**The code automatically handles rate limits by:**
- Rotating to the next API key when a limit is hit
- Retrying with exponential backoff
- Stopping with a message if all keys are exhausted

**If all keys are exhausted:**
- Get more Groq API keys from https://console.groq.com
- Update `GROQ_API_KEYS` in the SLURM script
- Resubmit the job (it will resume from where it stopped)

**Monitor API key rotation:**
```bash
tail -f sentiment_analysis.out | grep -E 'rotating|API key|🔑'
```

### Out of memory

**Increase memory in SLURM script:**
```bash
#SBATCH --mem=64G  # or 128G
```

### Job takes too long / Time limit reached

**The job saves incrementally**, so you can:

1. **Resubmit** - It will automatically skip already-processed weeks:
   ```bash
   sbatch cluster/run_sentiment_analysis.slurm
   ```

2. **Extend time limit** (if partition allows):
   ```bash
   scontrol update jobid=<JOBID> TimeLimit=10:00:00
   ```

3. **Check progress**:
   ```bash
   wc -l main_project/initial_test/llm_text/sentiment_scores.csv
   ```

### No output appearing

**Python output is unbuffered** (`python -u`), so output should appear immediately. If not:

1. Check if job is actually running:
   ```bash
   squeue -u $USER
   ps aux | grep python
   ```

2. Check for errors:
   ```bash
   cat sentiment_analysis.err
   ```

3. Verify the script is executing:
   ```bash
   tail -20 sentiment_analysis.out
   ```

### Job stops after a few seconds

**Possible causes:**
- Import errors (check `sentiment_analysis.err`)
- Missing dependencies (reinstall: `pip install -r requirements.txt`)
- Missing data file (verify `full_factiva.csv` exists)

**Debug:**
```bash
# Check what happened
cat sentiment_analysis.err
cat sentiment_analysis.out

# Test imports manually
source venv/bin/activate
python -c "from agents import Agent; print('OK')"
```

---

## Quick Reference

### On your Mac:

```bash
# Upload project
cd ~/Desktop/Agents_News
tar -czf agents_news.tar.gz --exclude='venv' --exclude='__pycache__' --exclude='*.csv' .
rsync -avz --progress agents_news.tar.gz favara@orcd-login001.mit.edu:~

# Upload data
rsync -avz --progress main_project/data/news_data/full_factiva.csv \
    favara@orcd-login001.mit.edu:~/Agents_News/main_project/data/news_data/

# Download results
rsync -avz --progress favara@orcd-login001.mit.edu:~/Agents_News/main_project/initial_test/llm_text/sentiment_scores.csv \
    main_project/initial_test/llm_text/
```

### On cluster:

```bash
# Setup
cd ~/Agents_News
tar -xzf ~/agents_news.tar.gz
source venv/bin/activate
pip install -e openai/
pip install -r requirements.txt

# Update SLURM script with your Groq API keys
nano cluster/run_sentiment_analysis.slurm
# Edit: export GROQ_API_KEYS="your-key1,your-key2,..."

# Submit job
sbatch cluster/run_sentiment_analysis.slurm

# Monitor
squeue -u $USER
tail -f sentiment_analysis.out

# Check progress
wc -l main_project/initial_test/llm_text/sentiment_scores.csv
```

---

## How It Works

1. **Loads news data** from `full_factiva.csv`
2. **Groups articles by week** (Monday to Sunday)
3. **For each week**:
   - **Sentiment Analyzer Agent** reads news and generates 4 sentiment scores
   - **Fact Checker Agent** validates scores against historical trends
   - **Coordinator Agent** manages the interaction
   - Scores are saved incrementally to `sentiment_scores.csv`
4. **API key rotation**: If rate limits hit, automatically switches to next key
5. **Resume capability**: If job stops, resubmit and it continues from last processed week

**Output format** (`sentiment_scores.csv`):
```csv
date,inflation_sentiment,ec_growth_sentiment,monetary_policy_sentiment,market_vol_sentiment
1990-01-01,0.0,-0.2,-0.5,0.0
1990-01-08,0.0,-0.3,-0.7,-0.3
...
```

---

Good luck! 🚀
