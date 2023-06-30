#!/bin/bash
#SBATCH --job-name=Name
#SBATCH --time=00-10:00:00
#SBATCH --mem=32G
cores=8
#SBATCH --cpus-per-task=$cores
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your_gmail
#SBATCH --output=/path/to/your/directory/output.out
#SBATCH --error=/path/to/your/directory/error.out

cd /path/to/your/directory

# Run your Python code with the desired number of cores
python run.py --cores $cores
