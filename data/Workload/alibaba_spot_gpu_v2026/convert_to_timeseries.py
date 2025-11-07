#!/usr/bin/env python3
"""
Convert Alibaba Spot GPU cluster trace to hourly time series format.

Input files:
- job_info_df.csv: Job submissions (submit_time, duration, resources)
- node_info_df.csv: Cluster capacity (nodes, GPUs, CPUs)

Output files:
- Alibaba_Spot_GPU_Hourly.csv: CPU utilization time series (SustainDC compatible)
- Alibaba_Spot_GPU_and_CPU_Hourly.csv: CPU + GPU utilization time series
- conversion_stats.txt: Conversion statistics

Author: Claude (Anthropic)
Date: November 6, 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 80)
print("ALIBABA SPOT GPU TRACE → TIME SERIES CONVERTER")
print("=" * 80)

# File paths
DATA_DIR = Path(__file__).parent
JOB_FILE = DATA_DIR / 'job_info_df.csv'
NODE_FILE = DATA_DIR / 'node_info_df.csv'
OUTPUT_CPU = DATA_DIR / 'Alibaba_Spot_GPU_Hourly.csv'
OUTPUT_BOTH = DATA_DIR / 'Alibaba_Spot_GPU_and_CPU_Hourly.csv'
STATS_FILE = DATA_DIR / 'conversion_stats.txt'

# Step 1: Load data
print("\n[1/6] Loading data...")
jobs_df = pd.read_csv(JOB_FILE)
nodes_df = pd.read_csv(NODE_FILE)

print(f"  ✓ Loaded {len(jobs_df):,} jobs")
print(f"  ✓ Loaded {len(nodes_df):,} nodes")

# Step 2: Calculate cluster capacity
print("\n[2/6] Calculating cluster capacity...")
total_gpus = nodes_df['gpu_capacity_num'].sum()
total_cpus = nodes_df['cpu_num'].sum()

print(f"  ✓ Total GPUs: {total_gpus:,}")
print(f"  ✓ Total CPUs: {total_cpus:,}")

# Step 3: Calculate job end times and total resources
print("\n[3/6] Processing job data...")
jobs_df['end_time'] = jobs_df['submit_time'] + jobs_df['duration']
jobs_df['total_cpu'] = jobs_df['cpu_request'] * jobs_df['worker_num']
jobs_df['total_gpu'] = jobs_df['gpu_request'] * jobs_df['worker_num']

# Remove any NaN rows
jobs_df = jobs_df.dropna(subset=['submit_time', 'duration', 'cpu_request', 'gpu_request', 'worker_num'])

print(f"  ✓ Valid jobs after filtering: {len(jobs_df):,}")
print(f"  ✓ Trace duration: {jobs_df['end_time'].max() / 3600:.1f} hours ({jobs_df['end_time'].max() / 86400:.1f} days)")

# Step 4: Create hourly time series
print("\n[4/6] Generating hourly time series...")

# Determine time range
max_time_seconds = int(jobs_df['end_time'].max())
num_hours = (max_time_seconds // 3600) + 1

print(f"  ✓ Time grid: {num_hours:,} hours")

# Initialize arrays
cpu_utilization = np.zeros(num_hours)
gpu_utilization = np.zeros(num_hours)

# For each hour, calculate active jobs
print("  ⏳ Processing hours (this may take a minute)...")
for hour in range(num_hours):
    hour_start = hour * 3600
    hour_end = (hour + 1) * 3600

    # Find active jobs: started before hour_end AND ended after hour_start
    active_mask = (jobs_df['submit_time'] < hour_end) & (jobs_df['end_time'] > hour_start)
    active_jobs = jobs_df[active_mask]

    # Aggregate resources
    total_cpu_demand = active_jobs['total_cpu'].sum()
    total_gpu_demand = active_jobs['total_gpu'].sum()

    # Normalize
    cpu_utilization[hour] = total_cpu_demand / total_cpus
    gpu_utilization[hour] = total_gpu_demand / total_gpus

    # Progress indicator every 500 hours
    if (hour + 1) % 500 == 0:
        print(f"    ... processed {hour + 1:,} / {num_hours:,} hours")

print(f"  ✓ Time series generated: {num_hours:,} hourly data points")

# Step 5: Save output files
print("\n[5/6] Saving output files...")

# Output 1: CPU-only (SustainDC compatible)
cpu_df = pd.DataFrame({
    'cpu_load': cpu_utilization
})
cpu_df.index.name = ''
cpu_df.to_csv(OUTPUT_CPU)
print(f"  ✓ Saved: {OUTPUT_CPU.name} ({OUTPUT_CPU.stat().st_size / 1024:.1f} KB)")

# Output 2: CPU + GPU
both_df = pd.DataFrame({
    'cpu_load': cpu_utilization,
    'gpu_load': gpu_utilization
})
both_df.index.name = ''
both_df.to_csv(OUTPUT_BOTH)
print(f"  ✓ Saved: {OUTPUT_BOTH.name} ({OUTPUT_BOTH.stat().st_size / 1024:.1f} KB)")

# Step 6: Generate statistics
print("\n[6/6] Generating statistics...")

stats = []
stats.append("=" * 80)
stats.append("CONVERSION STATISTICS")
stats.append("=" * 80)
stats.append("")
stats.append("INPUT DATA:")
stats.append(f"  Jobs processed: {len(jobs_df):,}")
stats.append(f"  Nodes in cluster: {len(nodes_df):,}")
stats.append("")
stats.append("CLUSTER CAPACITY:")
stats.append(f"  Total GPUs: {total_gpus:,}")
stats.append(f"  Total CPUs: {total_cpus:,}")
stats.append("")
stats.append("OUTPUT TRACE:")
stats.append(f"  Duration: {num_hours:,} hours ({num_hours / 24:.1f} days)")
stats.append(f"  Timesteps: {num_hours:,}")
stats.append("")
stats.append("CPU UTILIZATION STATISTICS:")
stats.append(f"  Min: {cpu_utilization.min():.4f}")
stats.append(f"  Max: {cpu_utilization.max():.4f}")
stats.append(f"  Mean: {cpu_utilization.mean():.4f}")
stats.append(f"  Median: {np.median(cpu_utilization):.4f}")
stats.append(f"  Std Dev: {cpu_utilization.std():.4f}")
stats.append("")
stats.append("GPU UTILIZATION STATISTICS:")
stats.append(f"  Min: {gpu_utilization.min():.4f}")
stats.append(f"  Max: {gpu_utilization.max():.4f}")
stats.append(f"  Mean: {gpu_utilization.mean():.4f}")
stats.append(f"  Median: {np.median(gpu_utilization):.4f}")
stats.append(f"  Std Dev: {gpu_utilization.std():.4f}")
stats.append("")
stats.append("OUTPUT FILES:")
stats.append(f"  {OUTPUT_CPU.name} - CPU utilization only (SustainDC compatible)")
stats.append(f"  {OUTPUT_BOTH.name} - CPU + GPU utilization")
stats.append("")
stats.append("PREVIEW (first 10 hours):")
stats.append("  Hour | CPU Util | GPU Util")
stats.append("  -----|----------|----------")
for i in range(min(10, num_hours)):
    stats.append(f"  {i:4d} | {cpu_utilization[i]:8.4f} | {gpu_utilization[i]:8.4f}")
stats.append("")
stats.append("=" * 80)

stats_text = "\n".join(stats)
STATS_FILE.write_text(stats_text)
print(f"  ✓ Saved: {STATS_FILE.name}")

# Print key statistics to console
print("\n" + "=" * 80)
print("CONVERSION COMPLETE!")
print("=" * 80)
print(f"\n📊 CPU Utilization: min={cpu_utilization.min():.4f}, max={cpu_utilization.max():.4f}, mean={cpu_utilization.mean():.4f}")
print(f"📊 GPU Utilization: min={gpu_utilization.min():.4f}, max={gpu_utilization.max():.4f}, mean={gpu_utilization.mean():.4f}")
print(f"\n📁 Output: {num_hours:,} hours ({num_hours / 24:.1f} days) of time series data")
print(f"\n✅ Ready to use in SustainDC!")
print(f"   Update config: workload_file: 'alibaba_spot_gpu_v2026/{OUTPUT_CPU.name}'")
print("=" * 80)
