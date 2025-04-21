#!/bin/bash
#SBATCH -p mmli
#SBATCH --mem=195g
#SBATCH --gres=gpu:2
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=6
#SBATCH --mail-user cne2@illinois.edu
#SBATCH -J Train-mCLM
#SBATCH -o slurm_outputs/slurm-%A_%a.out

#To run this, you need to comment out the if statement on line 149 of 
# /home/a-m/cne2/miniconda3/envs/mCLM/lib/python3.11/site-packages/lightning/fabric/plugins/environments/slurm.py
# because this cluster needs ntasks-per-node = gres=gpu + 1 to stop from hanging with DDP

#instead, put srun in front of python


module load CUDA/12.8.0
#module load cuDNN/8.9.2.23-CUDA-11.8.0

source /home/a-m/cne2/.bashrc
conda activate mCLM

nvidia-smi

cd ~/MMLI_projects/mCLM/mCLM/

export WANDB_MODE=offline

export PL_FAULT_TOLERANT_TRAINING=1

echo "Starting Main Script" 


#PYTHONPATH=. srun python mCLM/scripts/main.py --base_model /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-0.5B/ \
#    --pretrained_text_model /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-0.5B/ --check_val_every_n_steps 10000 \
#    --pretrained_tokenizer /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-0.5B/ \
#    --batch_size=20 --lr 1e-4 --ckpt_path ckpts/Qwen2.5-0.5B_Total_25kV2/ --version Qwen2.5-0.5B --max_epochs 3 \
#    --data_module Total --task Total_25k-v2 \
#    --save_checkpoint_every_n_steps 2500 \


#PYTHONPATH=. srun python mCLM/scripts/main.py --base_model /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-7B/ \
#    --pretrained_text_model /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-7B/ --check_val_every_n_steps 10000 \
#    --pretrained_tokenizer /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-7B/ \
#    --batch_size=3 --lr 1e-4 --ckpt_path ckpts/Qwen2.5-7B_SMolInstruct/ --version Qwen2.5-7B --max_epochs 3 \
#    --data_module SMolInstruct --task SMolInstruct \
#    --save_checkpoint_every_n_steps 2500 \
#    --max_negative_sampling_schedule 1500 \
#    --negative_sampling_schedule_loss 0.1 \



PYTHONPATH=. srun python mCLM/scripts/main.py --base_model /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-0.5B/ \
    --pretrained_text_model /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-0.5B/ --check_val_every_n_steps 10000 \
    --pretrained_tokenizer /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-0.5B/ \
    --batch_size=16 --lr 1e-4 \
    --ckpt_path ckpts/Qwen2.5-0.5B_SMolInstruct_FreePreGNN/ --version Qwen2.5-0.5B_FreePreGNN \
    --max_epochs 3 \
    --data_module SMolInstruct --task SMolInstruct \
    --save_checkpoint_every_n_steps 2500 \
    --max_negative_sampling_schedule 3000 \
    --negative_sampling_schedule_loss 0.1 \
    --load_GNN_ckpt ckpts_GNN/896_dim/best_val_checkpoint.ckpt \
    --freeze_GNN


#PYTHONPATH=. srun python mCLM/scripts/main.py --base_model /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-0.5B/ \
#    --pretrained_text_model /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-0.5B/ --check_val_every_n_steps 25 \
#    --pretrained_tokenizer /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-0.5B/ \
#    --batch_size=24 --lr 1e-5 --ckpt_path ckpts/Qwen2.5-0.5B_Test/ --version Qwen2.5-0.5B --max_epochs 3 \
#    --data_module Kinase --task Test

#PYTHONPATH=. srun python mCLM/scripts/main.py --base_model /home/a-m/cne2/MMLI_projects/LLMs/Llama-3.2-1B-Instruct/ \
#    --pretrained_text_model /home/a-m/cne2/MMLI_projects/LLMs/Llama-3.2-1B-Instruct/ --check_val_every_n_steps 10000 \
#    --batch_size=16 --lr 1e-5 --ckpt_path ckpts/1Bv4/ --version v4
    
#PYTHONPATH=. python mCLM/scripts/main.py --base_model /home/a-m/cne2/MMLI_projects/LLMs/Llama-3.2-1B-Instruct/ \
#    --pretrained_text_model /home/a-m/cne2/MMLI_projects/LLMs/Llama-3.2-1B-Instruct/ --check_val_every_n_steps 50 \
#    --batch_size=16 --lr 5e-5 --ckpt_path ckpts/test/

#PYTHONPATH=. python mCLM/scripts/chat.py --base_model /home/a-m/cne2/MMLI_projects/LLMs/Llama-3.2-1B-Instruct/ \
#    --pretrained_text_model /home/a-m/cne2/MMLI_projects/LLMs/Llama-3.2-1B-Instruct/ --ckpt_path ckpts/test/

# Please modify <SMILES> Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C </SMILES> to improve solubility and oral bioactivity.

