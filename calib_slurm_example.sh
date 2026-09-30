#!/bin/bash
# calib_slurm_example.sh
# Per-protein steps 1+2 of the box calibration as a SLURM array.
# Adjust partition/time/memory/environment to your cluster.
#
#   N=$(grep -cv '^\s*#' passing_pdbs.txt)
#   sbatch --array=1-${N}%20 calib_slurm_example.sh
#
# Afterwards, once, on a login/compute node:
#   python calib_fit_model.py --pdb-list passing_pdbs.txt --frame lab --scaling thermal --n-modes 10
#
#SBATCH --job-name=boxcalib
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/boxcalib_%A_%a.out

set -euo pipefail

# source your environment here, e.g.
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate pbc

PDB=$(grep -v '^\s*#' passing_pdbs.txt | sed -n "${SLURM_ARRAY_TASK_ID}p" | awk '{print tolower($1)}')
echo "task ${SLURM_ARRAY_TASK_ID}: ${PDB}"

CALIB=results/${PDB}/calib

if [ ! -f "${CALIB}/required_box.json" ]; then
    python calib_required_box.py --pdb "${PDB}"
fi

if [ ! -f "${CALIB}/envelope.npz" ]; then
    if [ -f "${CALIB}/nma/raw_modes_all.npy" ]; then
        python calib_nm_envelope.py --pdb "${PDB}"
    else
        python calib_nm_envelope.py --pdb "${PDB}" --run-nma
    fi
fi
