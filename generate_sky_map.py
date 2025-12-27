#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

"""
Generates random points on the celestial sphere:
 - 8.1×10^7 points distributed in the northern region (Dec >= -10°) with density decreasing smoothly to 0 at -10°
 - 1.9×10^5 uniformly distributed points across the entire sphere
Each category uses a distinct marker in the Aitoff projection plot.

Outputs:
 - CSV file containing RA, Dec, z for all points
 - PNG sky map

Requirements:
 - Python 3
 - NumPy
 - matplotlib
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import csv

# Parameters
N_north = int(8.1e5)
N_uniform = int(1.9e3)
outfile_csv = "sky_map.csv"
outfile_png = "sky_map.png"

fade_limit_rad = np.deg2rad(10.0)  # Fade below equator

# Function to generate northern points with fade (Dec >= -10°, density fades to 0 at -10°)
def generate_fade_north_points(N):
    ra_list, dec_list = [], []
    while len(ra_list) < N:
        dec_prop = np.arcsin(np.random.uniform(np.sin(-fade_limit_rad), 1.0))

        if dec_prop >= 0:
            accept_prob = 1.0
        else:
            accept_prob = 1.0 + dec_prop / fade_limit_rad
            accept_prob = max(0.0, accept_prob)

        if np.random.random() <= accept_prob:
            ra_val = np.random.uniform(0, 2 * np.pi)
            ra_wrapped = (ra_val + np.pi) % (2 * np.pi) - np.pi
            ra_list.append(ra_wrapped)
            dec_list.append(dec_prop)

    ra = np.array(ra_list)
    dec = np.array(dec_list)
    z = np.sin(dec)
    return ra, dec, z

# Function to generate uniformly distributed points over full sphere
def generate_uniform_sphere_points(N):
    z = np.random.uniform(-1.0, 1.0, size=N)
    dec = np.arcsin(z)
    ra = np.random.uniform(0, 2 * np.pi, size=N)
    ra_wrapped = (ra + np.pi) % (2 * np.pi) - np.pi
    return ra_wrapped, dec, z

# Generate all point categories
print("Generating northern region points...")
ra_north, dec_north, z_north = generate_fade_north_points(N_north)

print("Generating uniform full-sphere points...")
ra_uni, dec_uni, z_uni = generate_uniform_sphere_points(N_uniform)

# Concatenate for CSV output
ra_all = np.concatenate([ra_north, ra_uni])
dec_all = np.concatenate([dec_north, dec_uni])
z_all = np.concatenate([z_north, z_uni])

# Save to CSV file
print(f"Writing CSV file: {outfile_csv}")
with open(outfile_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["ra_rad", "dec_rad", "z"])
    for ra, dec, z in zip(ra_all, dec_all, z_all):
        writer.writerow([ra, dec, z])

print(f"CSV file written: {outfile_csv}")

# Visualization with different markers
def plot_aitoff(ra_north, dec_north, ra_uni, dec_uni, out_png):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='aitoff')
    ax.set_title('Sky points: Northern fade + Uniform sphere', fontsize=12)
    ax.grid(True)

    ax.scatter(ra_north, dec_north, s=0.2, alpha=0.6, marker='.', label='Northern fade region')
    ax.scatter(ra_uni, dec_uni, s=3, alpha=0.7, marker='x', label='Uniform full-sphere')

    ax.legend(loc='lower left')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {out_png}")
    plt.show()

plot_aitoff(ra_north, dec_north, ra_uni, dec_uni, outfile_png)

print(f"Generated: {N_north} northern fade points, {N_uniform} uniform-sphere points")
print("Coordinates: RA in radians (-π..π), Dec in radians (-π/2..π/2)")
