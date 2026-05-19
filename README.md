# DFXM Visualizations

> 3D animated visualizations of edge and screw dislocations in crystal lattices, built for the Dark-Field X-ray Microscopy research programme at DTU Physics.

## What this is

Crystal dislocations are line defects whose strain fields determine a material's mechanical and electronic properties. These scripts produce publication-quality 3D animations that walk through a dislocation's geometry step by step — from a perfect lattice, through the morphing introduction of the defect, to a strain-colormapped view of the surrounding stress field. Two dislocation types are covered: edge (Hirth & Lothe displacement-gradient formulation) and screw (isotropic elasticity, Burgers vector shear field). The animations were created to support intuition-building and outreach around the DFXM imaging research.

## Stack

- **Language:** Python 3.10
- **Key libraries:** PyVista 0.46.5 (3D rendering), NumPy, imageio + imageio-ffmpeg (MP4/GIF export)
- **License:** Apache 2.0

## How to run

```bash
git clone https://github.com/borgi-s/DFXM_Visualizations.git
cd DFXM_Visualizations
pip install -r requirements.txt

# Edge dislocation animation
python Edge_disloc_animation.py

# Screw dislocation animation
python Screw_disloc_animation.py
```

Output is written as MP4 (requires FFmpeg on `$PATH`) with automatic fallback to GIF. Both scripts are self-contained — no external data files required.

## Background

Developed during PhD research on Dark-Field X-ray Microscopy at DTU Physics. The physics underpinning the displacement fields follows Hirth & Lothe, *Theory of Dislocations* (2nd ed.) for edge dislocations and standard isotropic elasticity for screw dislocations. No specific publication is linked from this repository — the animations serve primarily as research communication and teaching material.
