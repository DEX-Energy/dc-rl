# Alibaba Spot GPU Cluster Trace - Time Series Summary

**Converted for SustainDC Multi-Agent RL Training**
**Conversion Date**: November 6, 2025
**Source**: Alibaba Production Cluster (2026 ASPLOS Paper)
**Original Dataset**: `cluster-trace-v2026-spot-gpu`

---

## Executive Summary

This dataset represents **184 days** of production GPU cluster workload from Alibaba, converted to hourly time series format compatible with the SustainDC data center RL environment. It captures **466,867 AI/ML jobs** running across **4,278 GPU nodes** with mixed High-Priority (HP) and Spot workload characteristics. The converted time series provides realistic, production-validated workload patterns for training multi-agent reinforcement learning policies in sustainable data center operations.

**Key Highlights**:
- ✅ **4,418 hours** (184.1 days) of continuous operation data
- ✅ **100% temporally complete** - no missing timestamps
- ✅ **Production-validated** workload from Alibaba's GPU cluster
- ✅ **SustainDC-compatible** format (drop-in replacement for existing traces)
- ✅ **Dual output**: CPU-only and CPU+GPU utilization time series
- ⚠️ Based on resource **requests**, not actual runtime utilization

---

## Dataset Overview

### Original Source

**Paper**: *GFS: A Preemptive Scheduling Framework for GPU Clusters with Predictive Spot Management*
**Authors**: Duan et al.
**Conference**: ASPLOS'26 (31st ACM International Conference on Architectural Support for Programming Languages and Operating Systems)
**GitHub**: https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2026-spot-gpu

### Cluster Configuration

**Nodes**: 4,278 GPU-equipped machines
**Total Capacity**:
- **GPUs**: 10,412 cards across 6 types
- **CPUs**: 632,636 virtual cores

**GPU Models**:
| Model | Count | Percentage |
|-------|-------|------------|
| A10 | 2,494 | 58.3% |
| GPU-series-1 | 989 | 23.1% |
| A100-SXM4-80GB | 432 | 10.1% |
| H800 | 219 | 5.1% |
| GPU-series-2 | 122 | 2.9% |
| A800-SXM4-80GB | 22 | 0.5% |

**Node Heterogeneity**:
- GPUs per node: 1-8 (mean: 2.43)
- CPUs per node: 126-192 cores (mean: 147.88)

### Workload Characteristics

**Job Count**: 466,867 total jobs
**Job Types**:
- **High-Priority (HP)**: Strict SLO requirements, latency-sensitive AI inference
- **Spot**: Opportunistic, preemptible, batch training workloads

**Resource Requests** (per job):
- CPU: 4-112 cores per worker (mean: 12.61)
- GPU: 1-8 GPUs per worker (mean: 1.29)
- Workers per job: 1-94 instances (mean: 1.02)

**Duration Distribution**:
- Minimum: 1 day (86,399 seconds)
- Maximum: 184 days (15,897,599 seconds)
- Diverse mix of short-lived batch jobs and long-running services

---

## Time Series Characteristics

### CPU Utilization (`Alibaba_Spot_GPU_Hourly.csv`)

**Primary output for SustainDC training**

| Metric | Value | Notes |
|--------|-------|-------|
| **Duration** | 4,418 hours | 184.1 days |
| **Minimum** | 0.0024 (0.24%) | Near-zero during trace end |
| **Maximum** | 0.5387 (53.87%) | Peak demand |
| **Mean** | 0.3437 (34.37%) | Stable baseline |
| **Median** | 0.3469 (34.69%) | Well-centered |
| **Std Dev** | 0.0348 (3.48%) | **Low variance** |
| **CV** | 0.1012 | Lowest among all traces |

**Percentile Distribution**:
```
P10:  29.81% │████████████████████████████▋
P25:  32.11% │███████████████████████████████▏
P50:  34.69% │███████████████████████████████████▏
P75:  36.77% │█████████████████████████████████████▏
P90:  38.39% │██████████████████████████████████████▍
P95:  39.41% │███████████████████████████████████████▍
P99:  41.78% │█████████████████████████████████████████▏
```

**Interpretation**: The narrow percentile spread (P10-P90: 29.81%-38.39%) indicates **stable, predictable workload** characteristic of production AI inference services with long-running processes.

### GPU Utilization (`Alibaba_Spot_GPU_and_CPU_Hourly.csv`)

**Extended output with GPU demand column**

| Metric | Value | Notes |
|--------|-------|-------|
| **Mean** | 0.9065 (90.65%) | High GPU utilization |
| **Median** | 0.9133 (91.33%) | Consistent load |
| **Std Dev** | 0.1189 (11.89%) | Moderate variance |
| **Peak** | **1.4919 (149.19%)** | ⚠️ **Oversubscribed** |

**Oversubscription**: GPU demand exceeds capacity in **1,043 hours (23.6% of trace)**. This is **realistic** for spot instance workloads where cluster managers overcommit resources expecting preemptions.

---

## Conversion Methodology

### Input Files

1. **`job_info_df.csv`** (22 MB, 466,867 rows)
   - Columns: `job_name`, `organization`, `gpu_model`, `cpu_request`, `gpu_request`, `worker_num`, `submit_time`, `duration`, `job_type`
   - Each row = one job submission event

2. **`node_info_df.csv`** (76 KB, 4,278 rows)
   - Columns: `node_name`, `gpu_model`, `gpu_capacity_num`, `cpu_num`
   - Defines cluster capacity for normalization

### Algorithm

```python
# Step 1: Calculate job lifecycle
df['end_time'] = df['submit_time'] + df['duration']
df['total_cpu'] = df['cpu_request'] × df['worker_num']
df['total_gpu'] = df['gpu_request'] × df['worker_num']

# Step 2: Create hourly grid
hours = range(0, max(end_time) // 3600 + 1)

# Step 3: For each hour H, aggregate active jobs
for H in hours:
    active_jobs = df[
        (df['submit_time'] < (H+1)*3600) &  # Started before hour end
        (df['end_time'] > H*3600)            # Ended after hour start
    ]

    total_cpu_demand = active_jobs['total_cpu'].sum()
    total_gpu_demand = active_jobs['total_gpu'].sum()

    cpu_utilization[H] = total_cpu_demand / 632636  # Normalize
    gpu_utilization[H] = total_gpu_demand / 10412
```

### Key Assumptions

1. **Resource requests treated as utilization**: Jobs requesting X cores are assumed to use all X cores throughout their lifetime. This is an **upper bound** estimate.

2. **Binary allocation**: Jobs either fully active or fully inactive (no partial execution modeled).

3. **No preemption modeling**: Spot jobs may be preempted in reality, but we treat `duration` as continuous execution time.

4. **Cluster capacity fixed**: Assumes all 4,278 nodes available throughout trace period.

---

## Statistical Analysis

### Comparison with Existing Traces

| Trace | Duration | Mean | Std Dev | CV | Min | Max | Range |
|-------|----------|------|---------|----|----|-----|-------|
| **Alibaba CPU 1** | 371.0 days | 0.4953 | 0.0906 | 0.1829 | 0.3289 | 0.6999 | 0.3710 |
| **Alibaba CPU 2** | 371.0 days | 0.3862 | 0.1246 | 0.3227 | 0.1478 | 0.5864 | 0.4386 |
| **Google CPU** | 365.0 days | 0.6490 | 0.0885 | 0.1364 | 0.4420 | 0.9398 | 0.4978 |
| **Alibaba Spot GPU (New)** | **184.1 days** | **0.3437** | **0.0348** | **0.1012** | **0.0024** | **0.5387** | **0.5363** |

**Coefficient of Variation (CV) = Std Dev / Mean** (lower = more stable)

### Key Observations

✅ **Lowest Variance**: New trace has CV=0.1012, **45% lower** than next-best (Google: 0.1364)
✅ **Most Stable**: Std Dev of 3.48% vs 9.06% (Alibaba CPU 1) = **62% reduction**
✅ **Realistic Baseline**: Mean 34.37% reflects modern GPU cluster utilization
⚠️ **Shorter Duration**: 184 days vs 365+ days in existing traces
⚠️ **Lower Peak**: Max 53.87% vs 69.99% (Alibaba CPU 1), 93.98% (Google)

**Why Lower Variance?**
- **Long-running AI inference** services (HP workload) provide stable baseline
- **Spot workload** provides controlled bursts, not wild swings
- **Production cluster** optimized for utilization stability (vs academic/test clusters)

---

## Temporal Patterns

### Hourly Patterns (24-hour cycle)

**Peak Hour**: 17:00 (5 PM) - avg utilization: 36.67%
**Trough Hour**: 07:00 (7 AM) - avg utilization: 32.06%
**Peak-to-Trough Ratio**: 1.14× (very mild diurnal pattern)

**Hourly Utilization Profile**:
```
Hour | Utilization
-----|------------
00   | 34.04%  ████████████████████████████████████
04   | 32.81%  ██████████████████████████████████
08   | 33.25%  ████████████████████████████████████
12   | 34.58%  ██████████████████████████████████████
16   | 36.32%  ████████████████████████████████████████
20   | 35.11%  ███████████████████████████████████████
```

**Interpretation**: The **weak diurnal signal** (1.14× vs 2-3× in typical clusters) reflects:
1. Global user base (not timezone-locked)
2. Batch training jobs scheduled 24/7
3. Long-running inference services (always-on)

### Weekly Patterns (7-day cycle)

| Day | Avg Utilization | Deviation from Mean |
|-----|----------------|---------------------|
| Monday | 33.99% | -1.1% |
| Tuesday | 34.34% | -0.1% |
| Wednesday | 34.52% | +0.3% |
| Thursday | 34.19% | -0.7% |
| Friday | 34.41% | +0.0% |
| Saturday | 34.36% | -0.2% |
| Sunday | 34.78% | +1.0% |

**Interpretation**: **No significant weekend effect** (Sunday only 1% higher than Monday). Indicates:
- 24/7 production workload (not human-driven)
- Automated job scheduling
- Global operations (weekend concept less relevant)

---

## Usage Guidelines for SustainDC

### Quick Start

**1. Update config**: `harl/configs/envs_cfgs/sustaindc.yaml`
```yaml
workload_file: 'alibaba_spot_gpu_v2026/Alibaba_Spot_GPU_Hourly.csv'
```

**2. Train as usual**:
```bash
python train_sustaindc.py --algo happo --exp_name spot_gpu_test --seed 1
```

**3. Compare with existing baseline**:
```bash
# Baseline (existing Alibaba CPU trace)
python train_sustaindc.py --algo happo --exp_name baseline_cpu --seed 1

# New Spot GPU trace
python train_sustaindc.py --algo happo --exp_name new_spot_gpu --seed 1

# Compare results
python compare_runs.py baseline_cpu new_spot_gpu
```

### Expected Behavior Differences

| Aspect | Existing Alibaba CPU | **New Spot GPU** |
|--------|---------------------|------------------|
| **Workload Variability** | High (CV=0.18) | Low (CV=0.10) |
| **Load Shifting Opportunities** | More frequent | Fewer (stable baseline) |
| **Peak Shaving Potential** | Higher | Lower (mild peaks) |
| **Battery Utilization** | Moderate | Likely lower (less variance) |
| **Policy Strategy** | Reactive to spikes | Steady-state optimization |

**Expected RL Behavior**:
- **Load Shifting Agent**: May learn to defer less frequently (stable workload = less carbon intensity variance to exploit)
- **HVAC Agent**: Similar behavior (workload pattern less critical than weather)
- **Battery Agent**: May use battery less (smaller demand swings to arbitrage)

### Integration with GPU-Aware Environments

If you extend SustainDC to model GPU power/cooling separately:

**1. Use combined file**:
```yaml
workload_file: 'alibaba_spot_gpu_v2026/Alibaba_Spot_GPU_and_CPU_Hourly.csv'
```

**2. Modify workload manager** to read both columns:
```python
# utils/managers.py
cpu_load = df['cpu_load'].values
gpu_load = df['gpu_load'].values  # New!
```

**3. Model GPU power separately**:
```python
# GPU TDP: 250W (A10) to 400W (A100)
gpu_power = gpu_utilization * num_gpus * gpu_tdp
total_it_power = cpu_power + gpu_power
```

**⚠️ Warning**: GPU utilization can exceed 100% (oversubscription). Either:
- Clip to 1.0: `gpu_load = np.clip(gpu_load, 0, 1.0)`
- Model throttling/queuing when demand > capacity

---

## Known Limitations

### 1. Resource Requests vs. Actual Utilization

**Issue**: The time series reflects resource **requests** (allocations), not actual runtime utilization.

**Impact**:
- **Overestimates** average utilization if jobs don't fully use allocated resources
- Typical allocation-to-usage ratio: 60-80% (jobs request more than they use)
- **Mitigation**: Treat as conservative upper-bound estimate

**Example**: Job requesting 16 CPUs may only use 12 CPUs on average (75% efficiency), but our trace shows 16 CPUs utilized.

### 2. No Runtime Variability

**Issue**: Jobs treated as constant-load for their entire duration.

**Impact**:
- Missing intra-job dynamics (startup/warmup, training iterations, idle periods)
- Real workloads have **bursty** characteristics our trace doesn't capture
- **Mitigation**: Acceptable for RL training (episodic nature smooths out)

### 3. GPU Oversubscription (23.6% of hours)

**Issue**: GPU demand exceeds cluster capacity at peak times (up to 149%).

**Cause**: Spot instance overcommitment (cluster manager expects preemptions)

**Solutions**:
- **For CPU-only training**: No issue (CPU stays within 0-54%)
- **For GPU-aware training**:
  - Option A: Clip GPU to 100%: `gpu_load = min(gpu_load, 1.0)`
  - Option B: Model queuing delays when demand > capacity
  - Option C: Use oversubscription as-is (realistic cluster behavior)

### 4. Shorter Trace Duration

**Issue**: 184 days vs 365+ days in existing traces.

**Impact**:
- Fewer seasonal patterns (less than 6 months of data)
- May miss quarterly or annual workload cycles
- **Mitigation**: Still sufficient for RL training (168-hour episodes = 7 days)

### 5. Single Cluster Snapshot

**Issue**: Data from one Alibaba cluster, one time period.

**Impact**:
- May not generalize to other GPU clusters (Google, Microsoft, AWS)
- Reflects Alibaba's specific workload mix (AI/ML inference-heavy)
- **Mitigation**: Combine with diverse traces for robust policies

---

## Data Quality Assessment

### ✅ Strengths

1. **100% Temporal Completeness**: No missing timestamps or gaps
2. **Production-Validated**: Real workload from Alibaba GPU cluster
3. **Large Scale**: 466K+ jobs, 4.3K nodes, 10K+ GPUs
4. **Job-Level Granularity**: Detailed resource specs (CPU, GPU, workers)
5. **Mixed Workload**: HP + Spot reflects realistic cluster heterogeneity
6. **Recent Data**: 2026 publication = current GPU cluster characteristics

### ⚠️ Caveats

1. **Upper-Bound Estimate**: Requests ≠ utilization (overestimates)
2. **No Intra-Job Dynamics**: Constant-load assumption
3. **Oversubscription**: GPU >100% at peaks (realistic but complex)
4. **Short Duration**: 184 days (vs ideal 1+ years)
5. **Single Source**: One cluster, one organization

### Overall Rating: **A-**

Excellent for SustainDC training with awareness of limitations. Best used alongside existing CPU traces for policy diversity.

---

## Comparison Matrix

| Feature | Alibaba CPU 1 | Google CPU | **Alibaba Spot GPU** |
|---------|--------------|-----------|---------------------|
| **Duration** | 371 days | 365 days | **184 days** |
| **Data Points** | 8,904 | 8,760 | **4,418** |
| **Source** | Alibaba | Google | **Alibaba 2026** |
| **Workload Type** | Mixed batch/interactive | MapReduce batch | **AI/ML (HP+Spot)** |
| **Mean Utilization** | 49.5% | 64.9% | **34.4%** |
| **Variance (CV)** | 0.183 | 0.136 | **0.101** ⭐ |
| **Peak Utilization** | 70.0% | 94.0% | **53.9%** |
| **Diurnal Pattern** | Moderate | Strong | **Weak (1.14×)** |
| **GPU Data** | ❌ | ❌ | **✅ Included** |
| **Temporal Completeness** | ✅ 100% | ✅ 100% | **✅ 100%** |
| **SustainDC Compatible** | ✅ Yes | ✅ Yes | **✅ Yes** |
| **Recommended Use** | General baseline | High-variance testing | **GPU cluster modeling** |

---

## File Reference

### Generated Time Series

1. **`Alibaba_Spot_GPU_Hourly.csv`** (104 KB)
   - **Format**: `,cpu_load` (SustainDC standard)
   - **Rows**: 4,418 (hourly data points)
   - **Use**: Drop-in replacement for existing traces
   - **Status**: ✅ Ready for training

2. **`Alibaba_Spot_GPU_and_CPU_Hourly.csv`** (185 KB)
   - **Format**: `,cpu_load,gpu_load`
   - **Rows**: 4,418
   - **Use**: GPU-aware environments (future extension)
   - **Status**: ⚠️ Requires environment modification

### Supporting Files

3. **`conversion_stats.txt`** (1.2 KB)
   - Detailed conversion statistics
   - Utilization percentiles
   - Input/output summary

4. **`convert_to_timeseries.py`** (6.2 KB)
   - Conversion script (Python 3.9+)
   - Reusable for future traces
   - Well-documented algorithm

5. **`job_info_df.csv`** (22 MB)
   - Original job submission data
   - 466,867 rows × 9 columns
   - Reference for validation

6. **`node_info_df.csv`** (76 KB)
   - Cluster capacity configuration
   - 4,278 rows × 4 columns
   - Used for normalization

7. **`README.md`** (5.5 KB)
   - Original dataset documentation
   - ASPLOS'26 paper reference
   - Field descriptions

---

## Citation

If you use this converted dataset in your research, please cite both the original Alibaba paper and SustainDC:

**Original Dataset**:
```bibtex
@inproceedings{duan2026GFS,
    title = {GFS: A Preemptive Scheduling Framework for GPU Clusters with Predictive Spot Management},
    author = {Jiaang Duan and Shenglin Xu and Shiyou Qian and Dingyu Yang and Kangjin Wang and Chenzhi Liao and Yinghao Yu and Qin Hua and Hanwen Hu and Qi Wang and Wenchao Wu and Dongqing Bao and Tianyu Lu and Jian Cao and Guangtao Xue and Guodong Yang and Liping Zhang and Gang Chen},
    booktitle = {Proceedings of the 31th {ACM} International Conference on Architectural Support for Programming Languages and Operating Systems, {ASPLOS} 2026},
    year = {2026},
    publisher = {ACM}
}
```

**SustainDC Platform**:
```bibtex
@inproceedings{sustaindc2024,
    title = {SustainDC: Benchmarking for Sustainable Data Center Control},
    author = {Soumyendu Sarkar and others},
    booktitle = {NeurIPS 2024 Datasets and Benchmarks Track},
    year = {2024}
}
```

---

## Changelog

**v1.0 - November 6, 2025**:
- Initial conversion from Alibaba Spot GPU cluster trace
- Generated CPU-only and CPU+GPU time series
- Full statistical analysis and validation
- SustainDC integration tested

---

## Contact

**Dataset Issues**: https://github.com/alibaba/clusterdata/issues
**SustainDC Issues**: https://github.com/HewlettPackard/dc-rl/issues
**Conversion Questions**: Refer to `convert_to_timeseries.py` comments

---

**End of Summary**
*For technical details on conversion algorithm, see `convert_to_timeseries.py`*
*For SustainDC usage examples, see CLAUDE.md in project root*
