#!/usr/bin/env python3
# plot_nbody_results.py
# Parses the txt/CSV you provided and produces 3 PNG plots + CSV outputs
# Usage: python3 plot_nbody_results.py

import re, os, sys
import pandas as pd
import matplotlib.pyplot as plt

# Adjust these paths if your files are elsewhere
TXT_FILE = "/users/lchidaga/itcs6145/nbody_simulation/nbody_sim/Parallel_p6_execution_times.txt"
OPENMP_CSV = "users/lchidaga/itcs6145/nbody_simulation/nbody_sim/nbody_openmp_output.csv"  # the CSV you already uploaded (if present)
OUTDIR = "plots_and_csvs"
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------
# 1. Parse the TXT file to structured tables
# ---------------------------
txt = open(TXT_FILE, "r", encoding="utf-8").read()

# Parse sequential section: lines like "64 - 0.373971"
seq_pattern = r"(\b\d+\b)\s*-\s*([0-9]+\.[0-9]+)\s*sec"
seq_matches = re.findall(seq_pattern, txt)
seq_rows = []
for m in seq_matches:
    n = int(m[0])
    t = float(m[1])
    seq_rows.append((n, t))

seq_df = pd.DataFrame(seq_rows, columns=["num_bodies", "runtime_sec"]).drop_duplicates().sort_values("num_bodies")
seq_df.to_csv(os.path.join(OUTDIR, "sequential_times.csv"), index=False)

# ---------------------------
# Parse OpenMP times: structure in the file is blocks like:
# 64 - 2 threads - 0.615671
# We'll search blocks of "N ...  - 2 threads ... - <time>"
# ---------------------------
omp_rows = []
# This regex finds lines like: 512     -   2 threads         -  6.49814 sec
omp_pattern = r"(\b\d+\b)\s*-\s*2 threads\s*-\s*([0-9]+\.[0-9]+)|" \
              r"(\b\d+\b)\s*-\s*4 threads\s*-\s*([0-9]+\.[0-9]+)|" \
              r"(\b\d+\b)\s*-\s*8 threads\s*-\s*([0-9]+\.[0-9]+)|" \
              r"(\b\d+\b)\s*-\s*16 threads\s*-\s*([0-9]+\.[0-9]+)"
# Simpler approach: iterate lines and parse lines that mention "threads"
for line in txt.splitlines():
    if "threads" in line:
        # collapse multiple spaces
        line2 = re.sub(r'\s+', ' ', line.strip())
        parts = line2.split('-')
        if len(parts) >= 3:
            try:
                n = int(parts[0].strip())
                thread_text = parts[1].strip()
                time_text = parts[2].strip()
                # extract threads number and time
                threads = int(re.search(r'(\d+)', thread_text).group(1))
                t = float(re.search(r'([0-9]+\.[0-9]+)', time_text).group(1))
                omp_rows.append((n, threads, t))
            except Exception:
                pass

omp_df = pd.DataFrame(omp_rows, columns=["num_bodies", "threads", "runtime_sec"]).sort_values(["num_bodies","threads"])
omp_df.to_csv(os.path.join(OUTDIR, "openmp_times_parsed.csv"), index=False)

# ---------------------------
# Parse CUDA blocksize table in your txt
# It contains blocks like: "4096 - 128 - 11.671 ms (0.011671 s)"
# We'll capture num_bodies, blocksize, ms and sec values
# ---------------------------
cuda_rows = []
cuda_pattern = r"(\b\d+\b)\s*-\s*(\b\d+\b)\s*-\s*([0-9]+\.[0-9]+)\s*ms\s*\(\s*([0-9]+\.[0-9]+)\s*s\s*\)"
for m in re.findall(cuda_pattern, txt):
    n = int(m[0]); block = int(m[1]); ms = float(m[2]); sec = float(m[3])
    cuda_rows.append((n, block, ms, sec))

cuda_df = pd.DataFrame(cuda_rows, columns=["num_bodies", "block_size", "runtime_ms", "runtime_sec"]).sort_values(["num_bodies","block_size"])
cuda_df.to_csv(os.path.join(OUTDIR, "cuda_times_parsed.csv"), index=False)

print("Parsed files saved to", OUTDIR)
print("Sequential:", seq_df.shape, "OpenMP:", omp_df.shape, "CUDA:", cuda_df.shape)

# ---------------------------
# 2. Create plots
#    Plot A: Bodies vs Runtime (seq vs best OpenMP vs best GPU)
# ---------------------------
# best OpenMP per N (minimum runtime among threads)
if not omp_df.empty:
    best_omp = omp_df.groupby("num_bodies", as_index=False)["runtime_sec"].min().rename(columns={"runtime_sec":"omp_best_sec"})
else:
    best_omp = pd.DataFrame(columns=["num_bodies","omp_best_sec"])

# best GPU per N (minimum runtime_sec from cuda_df)
if not cuda_df.empty:
    best_cuda = cuda_df.groupby("num_bodies", as_index=False)["runtime_sec"].min().rename(columns={"runtime_sec":"gpu_best_sec"})
else:
    best_cuda = pd.DataFrame(columns=["num_bodies","gpu_best_sec"])

# join tables
plotA = seq_df.merge(best_omp, on="num_bodies", how="outer").merge(best_cuda, on="num_bodies", how="outer").sort_values("num_bodies")
plotA.to_csv(os.path.join(OUTDIR, "combined_times.csv"), index=False)

plt.figure(figsize=(8,5))
if not plotA.empty:
    if "runtime_sec" in plotA.columns:
        plt.plot(plotA["num_bodies"], plotA["runtime_sec"], marker='o', label='Sequential')
    if "omp_best_sec" in plotA.columns:
        plt.plot(plotA["num_bodies"], plotA["omp_best_sec"], marker='o', label='OpenMP best')
    if "gpu_best_sec" in plotA.columns:
        plt.plot(plotA["num_bodies"], plotA["gpu_best_sec"], marker='o', label='GPU best')
plt.xscale('log'); plt.yscale('log')
plt.xlabel("Number of bodies (log scale)")
plt.ylabel("Runtime (seconds, log scale)")
plt.title("Bodies vs Runtime (sequential vs OpenMP vs GPU)")
plt.legend(); plt.grid(True, which="both", ls="--")
plt.savefig(os.path.join(OUTDIR, "bodies_vs_runtime.png"), dpi=150)
plt.close()

# ---------------------------
# 3. Plot B: Threads vs runtime for each num_bodies (OpenMP)
# ---------------------------
if not omp_df.empty:
    for n, g in omp_df.groupby("num_bodies"):
        plt.figure(figsize=(6,4))
        plt.plot(g["threads"], g["runtime_sec"], marker='o')
        plt.xlabel("Threads")
        plt.ylabel("Runtime (sec)")
        plt.title(f"OpenMP scaling for N={n}")
        plt.grid(True)
        plt.savefig(os.path.join(OUTDIR, f"openmp_threads_N{n}.png"), dpi=150)
        plt.close()

# ---------------------------
# 4. Plot C: GPU block size tuning (for a chosen N, default choose 4096 if present)
# ---------------------------
if not cuda_df.empty:
    chosen = 4096 if 4096 in cuda_df["num_bodies"].values else cuda_df["num_bodies"].unique()[0]
    sub = cuda_df[cuda_df["num_bodies"]==chosen]
    plt.figure(figsize=(6,4))
    plt.plot(sub["block_size"], sub["runtime_ms"], marker='o')
    plt.xlabel("Block size")
    plt.ylabel("Runtime (ms)")
    plt.title(f"GPU blocksize tuning (N={chosen})")
    plt.grid(True)
    plt.savefig(os.path.join(OUTDIR, f"gpu_blocksize_N{chosen}.png"), dpi=150)
    plt.close()

print("Plots created in", OUTDIR)
