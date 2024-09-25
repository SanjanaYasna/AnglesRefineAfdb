#!/bin/bash -l
#SBATCH --job-name=angles_csvs
#SBATCH --output=angles_conversions_fails.txt
##CHANGE TIME BELOW
#SBATCH --partition bigjay
#SBATCH --ntasks 1 --cpus-per-task=50

module load conda
conda activate gcn
cd /kuhpc/work/slusky/s300y051/AnglesRefine/
conda run -n gcn python angles_conversion.py 