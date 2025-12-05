#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import sys
import os

#############################################
# CONFIG – MODIFY ONLY THESE IF NEEDED
#############################################

CSV_FILE = "nbody_cuda_opt_out.csv"       # change if your CUDA/OpenMP output has a different name
OUT_FILE = "nbody_animation.gif"

MAX_BODIES = 500        # keep ONLY 500 bodies for animation to avoid memory kill
FRAME_SKIP = 20         # use every 20th frame to reduce load
FIG_SIZE = (6, 6)

#############################################
# LOAD DATA
#############################################

print("Loading CSV…")
df = pd.read_csv(CSV_FILE)

# detect which format the CSV uses
if "step" in df.columns:
    print("Detected multi-step CSV.")
else:
    print("Detected single-step CSV (no time dimension).")
    print("Cannot make animation; need multiple steps.")
    sys.exit(1)

#############################################
# DOWN SAMPLE bodies to avoid memory issues
#############################################
print("Selecting subset of bodies…")

unique_ids = df["id"].unique()
if len(unique_ids) > MAX_BODIES:
    selected_ids = np.random.choice(unique_ids, MAX_BODIES, replace=False)
else:
    selected_ids = unique_ids

df = df[df["id"].isin(selected_ids)]

#############################################
# DOWN SAMPLE FRAMES
#############################################
steps = sorted(df["step"].unique())
steps = steps[::FRAME_SKIP]     # skip frames
print(f"Total frames used: {len(steps)}")

#############################################
# PREPARE ANIMATION DATA
#############################################
frames = []
for s in steps:
    frame = df[df["step"] == s][["x", "y"]].values
    frames.append(frame)

#############################################
# CREATE ANIMATION
#############################################

fig, ax = plt.subplots(figsize=FIG_SIZE)
scat = ax.scatter([], [], s=2)

ax.set_xlim(0, 1000)
ax.set_ylim(0, 1000)
ax.set_title("N-body Simulation (2D Animation)")
ax.set_xlabel("X")
ax.set_ylabel("Y")

def update(frame):
    scat.set_offsets(frame)
    return scat,

print("Rendering animation… (this may take 15–30 seconds)")

anim = animation.FuncAnimation(
    fig,
    update,
    frames=frames,
    interval=50,   # ms per frame
    blit=True
)

anim.save(OUT_FILE, writer="pillow", fps=15)
print(f"Animation saved to {OUT_FILE}")
