# Complete Guide: Using MIT Engaging Cluster

**Last Updated**: November 2025  
**Cluster**: MIT Engaging (Open OnDemand / OOD)  
**Login Nodes**: `orcd-login001.mit.edu`, `orcd-login002.mit.edu`, `eofe7.mit.edu`, `eofe8.mit.edu`  
**User**: favara (your Kerberos username)

---

## Table of Contents

1. [What is MIT Engaging?](#what-is-mit-engaging)
2. [Access and Authentication](#access-and-authentication)
3. [Initial Setup](#initial-setup)
4. [Uploading Your Code](#uploading-your-code)
5. [Setting Up Python Environment](#setting-up-python-environment)
6. [Running Jobs with SLURM](#running-jobs-with-slurm)
7. [Monitoring Jobs](#monitoring-jobs)
8. [Downloading Results](#downloading-results)
9. [Common Commands Reference](#common-commands-reference)
10. [Troubleshooting](#troubleshooting)
11. [Best Practices](#best-practices)

---

## What is MIT Engaging?

MIT Engaging is MIT's high-performance computing (HPC) cluster system. It provides:
- **Compute nodes**: Powerful servers for running simulations
- **SLURM scheduler**: Job queue system for managing compute resources
- **Open OnDemand (OOD)**: Web interface for accessing cluster resources
- **Storage**: Home directory (`~`) and scratch space

**Key Points**:
- You **cannot** run long computations on login nodes (they're for file management and job submission only)
- All computations must be submitted as **SLURM jobs**
- Jobs run on dedicated compute nodes with specified resources (CPUs, memory, time)

---

## Access and Authentication

### Prerequisites

1. **MIT Kerberos account** (your username, e.g., `favara`)
2. **MIT VPN connection** (required for SSH/SCP access)
3. **Duo authentication** (two-factor authentication)

### Finding Your Login Node

**Option 1: Open OnDemand Web Interface**
- Go to: https://ood.mit.edu
- Navigate to: **Clusters → SSH Connection**
- The page will show your assigned login node and SSH command

**Option 2: Try Common Nodes**
- `orcd-login001.mit.edu` (most common)
- `orcd-login002.mit.edu`
- `eofe7.mit.edu`
- `eofe8.mit.edu`

### SSH Access

**Basic SSH connection:**
```bash
ssh favara@orcd-login001.mit.edu
```

**First-time connection**: You'll be prompted to accept the host key. Type `yes`.

**Authentication**: You'll be prompted for:
1. Your Kerberos password
2. Duo authentication (approve on your phone/device)

---

## Initial Setup

### Step 1: Set Up SSH Keys (Recommended)

SSH keys allow passwordless authentication, making file transfers much easier.

**On your Mac:**

```bash
# Generate SSH key (if you don't have one)
ssh-keygen -t ed25519 -C "favara@mit.edu"
# Press Enter to accept default location (~/.ssh/id_ed25519)
# Optionally set a passphrase (recommended)

# Copy your public key to the cluster
ssh-copy-id favara@orcd-login001.mit.edu
```

**Alternative: Manual key setup**

```bash
# On your Mac: Display your public key
cat ~/.ssh/id_ed25519.pub

# Copy the output, then SSH into cluster
ssh favara@orcd-login001.mit.edu

# On cluster: Append your public key to authorized_keys
mkdir -p ~/.ssh
chmod 700 ~/.ssh
echo "PASTE_YOUR_PUBLIC_KEY_HERE" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

**Verify passwordless login:**
```bash
ssh favara@orcd-login001.mit.edu
# Should connect without password prompt
```

---

## Uploading Your Code

### Method 1: SCP (Command Line) - Recommended

**On your Mac**, create a compressed archive of your project:

```bash
cd ~/Desktop/Plasma_Physics

# Create archive, excluding unnecessary files
tar -czf plasma_project.tar.gz \
    --exclude='venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='data/runs' \
    --exclude='.git' \
    --exclude='*.h5' \
    --exclude='*.log' \
    .

# Upload to cluster
scp plasma_project.tar.gz favara@orcd-login001.mit.edu:~
```

**What to exclude:**
- `venv/` - Virtual environment (recreate on cluster)
- `__pycache__/`, `*.pyc` - Python cache files
- `data/runs/` - Old simulation results (download new ones separately)
- `.git/` - Git repository (optional, but reduces size)
- `*.h5`, `*.log` - Large data files

**What to include:**
- All source code (`src/`, `scripts/`)
- Configuration files (`config/`)
- `requirements.txt`
- SLURM scripts (`cluster/*.slurm`)
- Any input data files (e.g., `data/nist_levels.csv`)

### Method 2: Open OnDemand Web Interface

1. Go to: https://ood.mit.edu
2. Navigate to: **Files → Home Directory**
3. Click **Upload** button
4. Select your `plasma_project.tar.gz` file
5. Wait for upload to complete

### Method 3: rsync (For Incremental Updates)

If you've already uploaded once and only need to update specific files:

```bash
# On your Mac
rsync -avz --exclude='venv' --exclude='__pycache__' \
    --exclude='data/runs' \
    ~/Desktop/Plasma_Physics/ \
    favara@orcd-login001.mit.edu:~/Plasma_Physics/
```

---

## Setting Up Python Environment

### Important: Python Version on MIT Engaging

**MIT Engaging does NOT have a `python/3.11` module.**

Instead, Python 3.11 is available directly at `/usr/bin/python3.11`.

### Step-by-Step Environment Setup

**SSH into cluster:**
```bash
ssh favara@orcd-login001.mit.edu
```

**Extract your project:**
```bash
cd ~
tar -xzf plasma_project.tar.gz
cd Plasma_Physics
```

**Note**: macOS extended attributes warnings (`LIBARCHIVE.xattr.*`) are harmless and can be ignored.

**Create virtual environment:**
```bash
# Use system Python 3.11 directly (no module load needed)
/usr/bin/python3.11 -m venv venv

# Activate environment
source venv/bin/activate

# Verify Python version
python --version  # Should show Python 3.11.x

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

**Verify installation:**
```bash
python -c "import numpy, scipy, matplotlib, bokeh, h5py, yaml; print('All packages installed!')"
```

**Expected packages:**
- numpy, scipy, matplotlib
- bokeh, h5py, pyyaml
- numba, jupyter, pandas

---

## Running Jobs with SLURM

### Understanding SLURM

SLURM (Simple Linux Utility for Resource Management) is the job scheduler. You submit jobs, and SLURM runs them when resources are available.

**Key SLURM directives:**
- `#SBATCH -J jobname` - Job name
- `#SBATCH -o output.out` - Standard output file
- `#SBATCH -e error.err` - Standard error file
- `#SBATCH -t HH:MM:SS` - Time limit (wall clock time)
- `#SBATCH --nodes=1` - Number of nodes
- `#SBATCH --cpus-per-task=N` - CPUs per task
- `#SBATCH --mem=NG` - Memory (e.g., `16G`, `32G`)

### Partition Limits

**MIT Engaging partitions:**
- `mit_normal` - Default partition, **2-hour time limit**
- `mit_long` - Longer jobs (check limits with `scontrol show partition mit_long`)

**Check partition limits:**
```bash
scontrol show partition mit_normal
```

### Creating SLURM Scripts

**Example: Single gas simulation (`run_single.slurm`)**

```bash
#!/bin/bash
#SBATCH -J plasma_single
#SBATCH -o plasma_single.out
#SBATCH -e plasma_single.err
#SBATCH -t 2:00:00          # 2 hours (matches partition limit)
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Using system Python 3.11 (no module load needed)

cd $SLURM_SUBMIT_DIR
source venv/bin/activate

python scripts/run_case.py \
    --pulse config/pulse.yml \
    --grid config/grid.yml \
    --medium config/medium.yml \
    --out data/runs/test_case_cluster
```

**Example: All gases simulation (`run_all_gases.slurm`)**

```bash
#!/bin/bash
#SBATCH -J plasma_all_gases
#SBATCH -o plasma_all_gases.out
#SBATCH -e plasma_all_gases.err
#SBATCH -t 2:00:00          # 2 hours (matches partition limit)
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8   # More CPUs for faster execution
#SBATCH --mem=32G

# Using system Python 3.11 (no module load needed)

cd $SLURM_SUBMIT_DIR
source venv/bin/activate

python scripts/run_all_gases.py \
    --out data/runs/all_gases
```

**Important notes:**
- **DO NOT** include `module load python/3.11` (it doesn't exist)
- Time limit must match partition limits (2 hours for `mit_normal`)
- `$SLURM_SUBMIT_DIR` is the directory where you ran `sbatch`

### Submitting Jobs

**On cluster:**
```bash
cd ~/Plasma_Physics

# Submit single job
sbatch cluster/run_single.slurm

# Or submit all gases
sbatch cluster/run_all_gases.slurm
```

**Output:**
```
Submitted batch job 12345678
```

**Save the job ID** - you'll need it for monitoring and cancellation.

---

## Monitoring Jobs

### Check Job Status

```bash
# Check all your jobs
squeue -u $USER

# Output example:
# JOBID     PARTITION  NAME            USER    ST  TIME  NODES NODELIST
# 12345678  mit_normal plasma_all_gas  favara  R   0:05  1     node1611
```

**Job states:**
- `PD` (Pending) - Waiting in queue
- `R` (Running) - Currently executing
- `CG` (Completing) - Finishing up
- `CA` (Cancelled) - Job was cancelled
- `F` (Failed) - Job failed

### View Job Details

```bash
# Detailed information about a specific job
scontrol show job <JOBID>

# Example:
scontrol show job 12345678
```

### Monitor Output in Real-Time

**Watch standard output:**
```bash
# For single run
tail -f plasma_single.out

# For all gases
tail -f plasma_all_gases.out
```

**Watch error output:**
```bash
tail -f plasma_all_gases.err
```

**Exit tail**: Press `Ctrl+C`

### Check Job History

```bash
# Show completed jobs
sacct -u $USER

# Show specific job details
sacct -j <JOBID> --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS
```

### Cancel a Job

```bash
# Cancel specific job
scancel <JOBID>

# Cancel all your jobs
scancel -u $USER
```

---

## Downloading Results

### Method 1: SCP (Command Line)

**On your Mac** (after job completes):

```bash
# Download single run
scp -r favara@orcd-login001.mit.edu:~/Plasma_Physics/data/runs/test_case_cluster \
     ~/Desktop/Plasma_Physics/data/runs/

# Download all gases
scp -r favara@orcd-login001.mit.edu:~/Plasma_Physics/data/runs/all_gases \
     ~/Desktop/Plasma_Physics/data/runs/
```

**Note**: Large files (several GB) may take 10-30 minutes to download.

### Method 2: Open OnDemand Web Interface

1. Go to: https://ood.mit.edu
2. Navigate to: **Files → Home Directory → Plasma_Physics → data → runs**
3. Select files/folders to download
4. Click **Download** button

### Method 3: rsync (For Incremental Downloads)

```bash
# On your Mac
rsync -avz favara@orcd-login001.mit.edu:~/Plasma_Physics/data/runs/ \
    ~/Desktop/Plasma_Physics/data/runs/
```

---

## Common Commands Reference

### On Your Mac

```bash
# Upload project
cd ~/Desktop/Plasma_Physics
tar -czf plasma_project.tar.gz --exclude='venv' --exclude='__pycache__' --exclude='data/runs' .
scp plasma_project.tar.gz favara@orcd-login001.mit.edu:~

# SSH into cluster
ssh favara@orcd-login001.mit.edu

# Download results
scp -r favara@orcd-login001.mit.edu:~/Plasma_Physics/data/runs/all_gases \
     ~/Desktop/Plasma_Physics/data/runs/
```

### On Cluster

```bash
# Navigate to project
cd ~/Plasma_Physics

# Activate environment
source venv/bin/activate

# Submit job
sbatch cluster/run_all_gases.slurm

# Check job status
squeue -u $USER

# Monitor output
tail -f plasma_all_gases.out

# Cancel job
scancel <JOBID>

# Check disk space
df -h ~

# List files
ls -lh data/runs/all_gases/

# Check Python version
python --version

# Test imports
python -c "import numpy; print(numpy.__version__)"
```

---

## Troubleshooting

### Connection Issues

**Problem**: `ssh: connect to host orcd-login001.mit.edu port 22: Connection refused`

**Solutions**:
1. **Check MIT VPN**: Make sure you're connected to MIT VPN
2. **Try different login node**: `eofe8.mit.edu` or `orcd-login002.mit.edu`
3. **Check Open OnDemand**: Verify your assigned login node

**Problem**: `scp: Connection closed`

**Solutions**:
1. Set up SSH keys (see [Initial Setup](#initial-setup))
2. Try SSH first: `ssh favara@orcd-login001.mit.edu`
3. Use Open OnDemand web interface for file transfer

### Python Environment Issues

**Problem**: `Lmod has detected the following error: The following module(s) are unknown: "python/3.11"`

**Solution**: MIT Engaging doesn't have a `python/3.11` module. Use system Python directly:
```bash
/usr/bin/python3.11 -m venv venv
```

**Problem**: `ERROR: No matching distribution found for numpy>=1.24.0`

**Solution**: You're using an old Python version. Check and use Python 3.11:
```bash
which python3.11  # Should show /usr/bin/python3.11
/usr/bin/python3.11 -m venv venv
source venv/bin/activate
python --version  # Should show 3.11.x
```

### Job Submission Issues

**Problem**: Job is pending with reason `(PartitionTimeLimit)`

**Solution**: Your requested time exceeds the partition limit. Check limits:
```bash
scontrol show partition mit_normal
```
Reduce time limit in SLURM script to match (e.g., `#SBATCH -t 2:00:00`).

**Problem**: Job fails immediately

**Solutions**:
1. Check error file: `cat plasma_all_gases.err`
2. Verify Python environment: `python --version`
3. Verify packages: `python -c "import numpy; print(numpy.__version__)"`
4. Check config files exist: `ls config/*.yml`

**Problem**: Job runs but produces no output

**Solutions**:
1. Check output file: `cat plasma_all_gases.out`
2. Verify disk space: `df -h ~`
3. Check file permissions: `ls -la data/runs/`

### Performance Issues

**Problem**: Job takes too long

**Solutions**:
1. Check cluster load: `squeue` (many jobs = slower)
2. Request more CPUs: `#SBATCH --cpus-per-task=8`
3. Check if simulation is progressing: `tail -f plasma_all_gases.out`

**Problem**: Out of memory errors

**Solutions**:
1. Increase memory: `#SBATCH --mem=64G`
2. Reduce save interval in `config/grid.yml` (e.g., `save_interval: 100`)

### File Transfer Issues

**Problem**: `tar: Ignoring unknown extended header keyword 'LIBARCHIVE.xattr.*'`

**Solution**: This is harmless - macOS extended attributes. Ignore the warnings.

**Problem**: Upload/download is very slow

**Solutions**:
1. Compress files before transfer: `tar -czf`
2. Exclude unnecessary files (venv, __pycache__, old data)
3. Use `rsync` for incremental updates
4. Consider using Open OnDemand web interface for large files

---

## Best Practices

### 1. Project Organization

**Keep cluster-specific files separate:**
```
Plasma_Physics/
├── cluster/
│   ├── run_single.slurm
│   ├── run_all_gases.slurm
│   └── MIT_ENGAGING_COMPLETE_GUIDE.md
├── config/
├── src/
├── scripts/
└── requirements.txt
```

### 2. Archive Creation

**Always exclude:**
- `venv/` - Recreate on cluster
- `__pycache__/`, `*.pyc` - Python cache
- `data/runs/` - Old results (download separately)
- `.git/` - Optional (reduces size)

**Command:**
```bash
tar -czf plasma_project.tar.gz \
    --exclude='venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='data/runs' \
    --exclude='.git' \
    .
```

### 3. Resource Allocation

**Start conservative, then scale up:**
- **Single test run**: 4 CPUs, 16GB RAM, 2 hours
- **All gases**: 8 CPUs, 32GB RAM, 2 hours
- **Large parameter sweeps**: Consider splitting into multiple jobs

### 4. Monitoring

**Always monitor your first job:**
```bash
# Submit job
sbatch cluster/run_single.slurm

# Immediately check status
squeue -u $USER

# Watch output
tail -f plasma_single.out
```

### 5. Data Management

**Download results promptly:**
- Cluster storage may have quotas
- Download completed runs to your local machine
- Delete old runs on cluster if needed: `rm -rf data/runs/old_run`

**Check disk space:**
```bash
df -h ~
```

### 6. Version Control

**Keep code in Git:**
- Commit changes before uploading
- Tag releases for reproducibility
- Document changes in commit messages

### 7. Documentation

**Document your workflow:**
- Keep notes of successful configurations
- Record runtime and resource usage
- Document any issues and solutions

---

## Quick Start Checklist

**First-time setup:**
- [ ] Connect to MIT VPN
- [ ] Set up SSH keys
- [ ] Find your login node
- [ ] Create project archive (excluding venv, __pycache__, old data)
- [ ] Upload archive to cluster
- [ ] SSH into cluster
- [ ] Extract archive
- [ ] Create Python virtual environment (`/usr/bin/python3.11 -m venv venv`)
- [ ] Install dependencies (`pip install -r requirements.txt`)
- [ ] Verify installation

**Running a job:**
- [ ] Create/update SLURM script (check time limits!)
- [ ] Submit job (`sbatch cluster/run_single.slurm`)
- [ ] Monitor job (`squeue -u $USER`, `tail -f *.out`)
- [ ] Wait for completion
- [ ] Verify output files exist
- [ ] Download results to local machine

**Troubleshooting:**
- [ ] Check error files (`*.err`)
- [ ] Check output files (`*.out`)
- [ ] Verify Python environment
- [ ] Check disk space
- [ ] Verify config files exist
- [ ] Check partition limits

---

## Additional Resources

- **Open OnDemand**: https://ood.mit.edu
- **MIT Engaging Documentation**: Check MIT IS&T documentation
- **SLURM Documentation**: https://slurm.schedmd.com/documentation.html
- **SSH Key Setup**: https://kb.mit.edu/confluence/display/istcontrib/SSH+Key+Setup

---

## Summary

**Your typical workflow:**

1. **On Mac**: Create archive → Upload to cluster
2. **On Cluster**: Extract → Setup Python environment → Submit job
3. **Monitor**: Check job status → Watch output
4. **Download**: After completion → Download results to Mac
5. **Analyze**: Visualize results locally with Bokeh

**Key points to remember:**
- Use `/usr/bin/python3.11` directly (no module load)
- Time limit must be ≤ 2 hours for `mit_normal` partition
- Always exclude `venv/` and `__pycache__/` from uploads
- Monitor your first job to catch issues early
- Download results promptly to avoid storage issues

---

**Good luck with your simulations! 🚀**

