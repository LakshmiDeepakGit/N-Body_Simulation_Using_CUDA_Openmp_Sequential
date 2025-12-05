🚀 High-Performance N-Body Gravitational Simulation
CPU (Sequential) → OpenMP (Multicore) → CUDA (GPU) + Visualization + 1-Hour Challenge Benchmark

This repository contains a full high-performance N-Body simulation pipeline implemented across three execution models:

🟦 Sequential CPU (C++)

🟧 OpenMP Parallel CPU (C++)

🟩 CUDA GPU (NVIDIA GPUs)

🎞 2D visualization (Python Matplotlib)

🏆 1-Hour GPU Challenge (Massive Performance Benchmark)

The goal of this project is to understand how different parallel computing models scale for the classical N-Body gravitational problem, and to optimize performance across CPU and GPU architectures.

⭐ 1. What is the N-Body Simulation?

The N-Body problem models how objects (“bodies”) move under gravity.

Each body has:

(x, y, z) → position

(vx, vy, vz) → velocity

mass

At every time-step:

Each body pulls on every other body

Gravitational force is computed using Newton’s law: F= G.((m1.m2)/r^2)

Forces accumulate

Velocities and positions are updated

Repeat thousands or millions of times

Because every body interacts with every other, this becomes:

O(N^2) complexity
→ Extremely expensive for large N

This is why HPC (High-Performance Computing) techniques are essential.

⭐ 2. Repository Structure

📂 nbody-simulation/
│
├── nbody.cpp                 # Sequential CPU implementation
├── nbody_openmp.cpp          # OpenMP multi-threaded version
├── nbody_cuda.cu             # CUDA GPU implementation
├── plot_nbody_results.py
├── animate_nbody.py   # GIF generation
├── sequential_times.csv
├── openmp_times.csv
├── cuda_times.csv
├── bodies_vs_runtime.png
├── openmp_threads_NumBodies.png ( for all Num_bodies)
├── cuda_blocksize_tuning.png
└── Nbody_Simulation_execution_times.txt
└── README.md                 # (this file)

⭐ 3. How to Run the Simulation (  Note that before running the code, Check the notepad file(Nbody_Simulation_execution_times.txt) provided and note all the execution times along with num_bodies, threads which you are executing in the same format like Nbody_Simulation_execution_times.txt )
3.1 Sequential CPU Version  -->  try changing with different NUM_BODIES sizes inside the code
Compile
g++ -std=c++11 nbody.cpp -o nbody

Run
./nbody


Output:

Completed step 0/8000
Completed step 500/8000
...
nbody_output.csv generated

⭐ 3.2 OpenMP Multicore Version   ---> try changing with different NUM_BODIES sizes in code and everytime we change the NUM_BODIES, we need to complile the code before running the code
Compile
g++ -std=c++11 -fopenmp -O3 nbody_openmp.cpp -o nbody_openmp

Set threads
export OMP_NUM_THREADS=16    ---> Try with different number of threads

Run
./nbody_openmp

⭐ 3.3 CUDA GPU Version
Step 1 — Get GPU Node on HPC
srun --partition=GPU --gres=gpu:1 --time=2:00:00 --pty bash

Step 2 — Load CUDA
module load cuda/12.1
nvcc --version

Step 3 — Compile CUDA Code ------>    try changing with different NUM_BODIES sizes inside the code
nvcc -O3 --use_fast_math -arch=sm_61 nbody_cuda.cu -o nbody_cuda

Step 4 — Run
./nbody_cuda

After executing all three implementations — Sequential, OpenMP, and CUDA — upload the notepad file (containing the recorded execution times, number of bodies, and CPU thread configurations) to the HPC system  where you are working.

⭐ 4. Running the 1-Hour GPU Challenge (Bonus Task)

Goal:
Find the maximum bodies × steps that can run within 1 hour on GPU.

Example command:

./nbody_cuda 15000 4110621 0.01


My highest score achieved:

🏆 61.66 Billion body-updates/hour
(15,000 bodies × 4,110,621 steps)

A complete CSV of all tested combinations is included in challenge_results.csv.

⭐ 5. Plotting & Analysis (Python)
Install Python packages on HPC
module load python
python3 -m ensurepip --user
python3 -m pip install --user pandas matplotlib
export PATH=$HOME/.local/bin:$PATH
source ~/.bashrc

Run plot generation:  ---> change the notepad file path in the code as per your naming convention
python3 plot_nbody_results.py


This generates:

CPU vs OpenMP vs CUDA runtime graphs

Thread scaling plots

GPU block-size performance

Combined runtime comparison

All saved under plots_and_csvs/.

⭐ 6. Visualization (2D Animation)

Generate GIF:

python3 animate_nbody.py


Output:

nbody_animation.gif


This animates body positions over steps using matplotlib.

⭐ 7. Performance Summary
Bodies	Time/Step (sec)	Steps in 1 Hour	Score (Bodies × Steps)
15000	0.00087578	4,110,621	61.66 Billion
10000	0.00061066	5,895,261	58.95 Billion
30000	0.00281724	1,277,860	38.33 Billion
5000	0.00114558	3,142,513	15.71 Billion

GPU excels dramatically for mid-range body counts due to memory bandwidth balance.

⭐ 8. Why This Project Matters (Industry Relevance)

This project demonstrates real-world engineering concepts:

✔ Scalable systems
✔ Parallel programming
✔ GPU acceleration
✔ Backend robustness under heavy load
✔ Simulation & physics modeling
✔ Performance tuning
✔ Cloud/HPC workflows

These techniques apply to:

AI/ML acceleration

High-traffic backend systems

Robotics + physics engines

Game development

Video analytics pipelines

Scientific computing

⭐ 9. How to Cite / Reference

If you use this project for learning or enhancements, please reference:

High-Performance N-Body Simulation (CPU + OpenMP + CUDA)
Author: Lakshmi Deepak Chidagam

⭐ 10. License

MIT License (or whatever you choose)

⭐ 11. Contact

If you want to collaborate on:
🐳 HPC • ⚡ CUDA • 🤖 AI/ML Acceleration • ☁ Cloud Systems • 🧠 Robotics • High-traffic backend applications
— feel free to connect!
Lakshmi Deepak Chidagam
mail : chdeepak4568@gmail.com
