#!/bin/bash -l
## Nazwa zlecenia
#SBATCH -J seg-train
## Liczba alokowanych wêz³ów
#SBATCH -N 1
#SBATCH -n 8
## Iloœæ pamiêci przypadaj¹cej na jeden rdzeñ obliczeniowy (domyœlnie 5GB na rdzeñ)
#SBATCH --mem-per-cpu=5GB
## Maksymalny czas trwania zlecenia (format HH:MM:SS)
#SBATCH --time=12:00:00 
## Nazwa grantu do rozliczenia zu¿ycia zasobów
#SBATCH -A plglaoisi24-gpu-a100
## Specyfikacja partycji
#SBATCH -p plgrid-gpu-a100
## Konfiguracja GPU
#SBATCH --gres=gpu:1
 
 
## przejscie do katalogu z ktorego wywolany zostal sbatch
cd $SLURM_SUBMIT_DIR


srun /bin/hostname
##module add scipy-bundle/2021.10-intel-2021b
##module add cuda
source $SCRATCH/decision_transformers_venv/bin/activate
##source $HOME/venvs/open-mmlab/bin/activate
##export PYTHONPATH=$HOME/workspace/cyfrovet/mmdetection
##cd $HOME/image_processing/
##cd $HOME/longevity-study/
cd $HOME/cldt/

##./tools/dist_train.sh \
 ##   cyfrovet/mask_rcnn_x101_64x4d_fpn_1x_cells_complete.py \
 ##   8 \
  ##  --work-dir ./runs


python3 -s train_single.py  $1 $2
##python -m rl_zoo3.train --env PandaReachDense-v3 --algo tqc --conf-file configs/tqcher_zoo.yaml --folder trained --save-freq 100000 --hyperparams n_envs:4 gradient_steps:-1

##python -m rl_zoo3.train --env PandaReachDense-v3 --algo tqc --conf-file configs/tqcher_zoo.yaml --save-freq 100000 --hyperparams n_envs:4 gradient_steps:-1

##python -m rl_zoo3.train --env PandaReach-v3 --algo tqc --conf-file configs/tqcher_zoo.yaml --save-freq 100000 --hyperparams n_envs:4 gradient_steps:-1


