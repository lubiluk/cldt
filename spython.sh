#!/bin/bash -l
sbatch <<EOT
#!/bin/bash -l
#SBATCH --job-name=$1
#SBATCH --time=48:00:00
#SBATCH --account=plgfactoryrl-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --gres=gpu
#SBATCH --output=$SCRATCH/output/slurm-%j.out

conda activate cldt
python $@
EOT

