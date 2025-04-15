#!/bin/bash
#SBATCH -p mmli
#SBATCH --mem=40g
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mail-user cne2@illinois.edu
#SBATCH -J Train-mCLM
#SBATCH -o slurm_outputs/slurm-%A_%a.out

module load CUDA/12.8.0
#module load cuDNN/8.9.2.23-CUDA-11.8.0

source /home/a-m/cne2/.bashrc
conda activate mCLM

nvidia-smi

cd ~/MMLI_projects/mCLM/mCLM/

export WANDB_MODE=offline

echo "Starting Main Script" 


#PYTHONPATH=. srun python mCLM/scripts/main.py --base_model /home/a-m/cne2/MMLI_projects/LLMs/Llama-3.2-1B-Instruct/ \
#    --pretrained_text_model /home/a-m/cne2/MMLI_projects/LLMs/Llama-3.2-1B-Instruct/ --check_val_every_n_steps 10000 \
#    --batch_size=16 --lr 1e-5 --ckpt_path ckpts/1B_TotalA/ --version Total --max_epochs 3 \
#    --data_module Total --task Total
    
    
#PYTHONPATH=. srun python mCLM/scripts/main.py --base_model /home/a-m/cne2/MMLI_projects/LLMs/Llama-3.2-1B-Instruct/ \
#    --pretrained_text_model /home/a-m/cne2/MMLI_projects/LLMs/Llama-3.2-1B-Instruct/ --check_val_every_n_steps 10000 \
#    --batch_size=16 --lr 1e-5 --ckpt_path ckpts/1B_SMolInstruct/ --version SMolInstruct --max_epochs 3 \
#    --data_module SMolInstruct --task SMolInstruct


#PYTHONPATH=. python mCLM/scripts/main.py --base_model /home/a-m/cne2/MMLI_projects/LLMs/Llama-3.2-1B-Instruct/ \
#    --pretrained_text_model /home/a-m/cne2/MMLI_projects/LLMs/Llama-3.2-1B-Instruct/ --check_val_every_n_steps 50 \
#    --batch_size=16 --lr 5e-5 --ckpt_path ckpts/test/

#PYTHONPATH=. python mCLM/scripts/chat.py --base_model /home/a-m/cne2/MMLI_projects/LLMs/Llama-3.2-1B-Instruct/ \
#    --pretrained_text_model /home/a-m/cne2/MMLI_projects/LLMs/Llama-3.2-1B-Instruct/ --ckpt_path ckpts/test/

# Please modify <SMILES> Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C </SMILES> to improve solubility and oral bioactivity.

