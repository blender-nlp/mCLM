#!/bin/bash
#SBATCH -p mmli
#SBATCH --mem=480g
#SBATCH --gres=gpu:4
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
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



if false; then
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

#export TORCH_DISTRIBUTED_DEBUG=DETAIL

PYTHONPATH=. srun python mCLM/scripts/main.py --base_model /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-3B/ \
    --pretrained_text_model /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-3B/ \
    --pretrained_tokenizer /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-3B/ \
    --check_val_every_n_steps 10000 \
    --batch_size=4 --lr 2e-5 --mol_lr 2e-6 \
    --ckpt_path ckpts/OnlyBlocks/Qwen2.5-3B_TotalTop1k_Adaptor_splitLoss/ --version Qwen2.5-3B_Adaptor_splitLoss \
    --max_epochs 1 \
    --no_PEFT \
    --train_adapter \
    --accumulate_grad_batches 4 \
    --data_module TotalTopK --task TotalTop500 \
    --num_warmup_steps 2000 \
    --save_checkpoint_every_n_steps 1000 \
    --instruction_data_path /home/a-m/cne2/MMLI_projects/mCLM/data/instruction_onlyblocks_top_500/ \
    --synthetic_data_path /home/a-m/cne2/MMLI_projects/mCLM/data/synthetic_onlyblocks_top_500/ \
    --downsample_tulu 0.1 \
    --tokenizer_cache ../GNN_input_cache/Total.molecule_tokenizer.500.pth \
    --pretrained_embeddings final_embeddings/OnlyBlocks/Top500/128_dim/ \

    #--freeze_GNN \

fi

if false; then
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHONPATH=. srun python mCLM/scripts/main.py --base_model /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-7B/ \
    --pretrained_text_model /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-7B/ \
    --pretrained_tokenizer /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-7B/ \
    --check_val_every_n_steps 10000 \
    --batch_size=1 --lr 2e-5 --mol_lr 2e-6 \
    --ckpt_path ckpts/OnlyBlocks/Qwen2.5-7B_TotalTop1k_PreAdaptor_splitLoss/ --version Qwen2.5-7B_PreAdaptor_splitLoss \
    --max_epochs 5 \
    --no_PEFT \
    --accumulate_grad_batches 4 \
    --data_module TotalTopK --task TotalTop500 \
    --num_warmup_steps 2000 \
    --save_checkpoint_every_n_steps 1000 \
    --instruction_data_path /home/a-m/cne2/MMLI_projects/mCLM/data/instruction_onlyblocks_top_500/ \
    --synthetic_data_path /home/a-m/cne2/MMLI_projects/mCLM/data/synthetic_onlyblocks_top_500/ \
    --downsample_tulu 0.1 \
    --tokenizer_cache ../GNN_input_cache/Total.molecule_tokenizer.500.pth \
    --pretrained_embeddings final_embeddings/OnlyBlocks/Top500/128_dim/ \
    --train_adapter \

fi

if true; then
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_LAUNCH_BLOCKING=1 PYTHONPATH=. srun python mCLM/scripts/main.py --base_model /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-3B/ \
    --pretrained_text_model /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-3B/ \
    --pretrained_tokenizer /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-3B/ \
    --check_val_every_n_steps 10000 \
    --batch_size=4 --lr 2e-5 --mol_lr 2e-6 \
    --ckpt_path ckpts/OnlyBlocks/Qwen2.5-3B_TotalTop1k_FT2_splitLoss/posneg/ --version Qwen2.5-3B_FT2_posneg_GNN_splitLoss \
    --max_epochs 5 \
    --no_PEFT \
    --finetune \
    --accumulate_grad_batches 4 \
    --data_module FinetuneTopK --task FinetuneTop500 \
    --num_warmup_steps 5000 \
    --save_checkpoint_every_n_steps 1000 \
    --synthetic_data_path /home/a-m/cne2/MMLI_projects/mCLM/data/finetune_top_500_v2/ \
    --tokenizer_cache ../GNN_input_cache/Total.molecule_tokenizer.500.pth \
    --load_ckpt ckpts/OnlyBlocks/Qwen2.5-3B_TotalTop1k_PreAdaptor_splitLoss/latest_checkpoint-epoch=04-step=129000.ckpt \
    --pretrained_embeddings final_embeddings/OnlyBlocks/Top500/128_dim/ \

fi

if false; then
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_LAUNCH_BLOCKING=1 PYTHONPATH=. srun python mCLM/scripts/main.py --base_model /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-3B/ \
    --pretrained_text_model /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-3B/ \
    --pretrained_tokenizer /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-3B/ \
    --batch_size=4 --lr 2e-5 --mol_lr 2e-6 \
    --ckpt_path ckpts/OnlyBlocks/Qwen2.5-3B_TotalTop1k_FT2_splitLoss/predOnly_GNN/ --version Qwen2.5-3B_FT2_predOnly_GNN_splitLoss \
    --max_epochs 25 \
    --no_PEFT \
    --finetune \
    --accumulate_grad_batches 4 \
    --data_module FinetuneTopK --task FinetuneTop500 \
    --num_warmup_steps 5000 \
    --save_checkpoint_every_n_steps 1000 \
    --synthetic_data_path /home/a-m/cne2/MMLI_projects/mCLM/data/finetune_top_500/ \
    --tokenizer_cache ../GNN_input_cache/Total.molecule_tokenizer.500.pth \
    --load_ckpt ckpts/OnlyBlocks/Qwen2.5-3B_TotalTop1k_PreAdaptor_splitLoss/latest_checkpoint-epoch=04-step=129000.ckpt \

    #--pretrained_embeddings final_embeddings/OnlyBlocks/Top500/128_dim/ \

fi

if false; then
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHONPATH=. srun python mCLM/scripts/main.py --base_model /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-3B/ \
    --pretrained_text_model /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-3B/ \
    --pretrained_tokenizer /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-3B/ \
    --check_val_every_n_steps 10000 \
    --batch_size=4 --lr 2e-5 --mol_lr 2e-6 \
    --ckpt_path ckpts/OnlyBlocks/Qwen2.5-3B_TotalTop1k_PreAdaptor_splitLoss/ --version Qwen2.5-3B_PreAdaptor_splitLoss \
    --max_epochs 5 \
    --no_PEFT \
    --accumulate_grad_batches 4 \
    --data_module TotalTopK --task TotalTop500 \
    --num_warmup_steps 2000 \
    --save_checkpoint_every_n_steps 1000 \
    --instruction_data_path /home/a-m/cne2/MMLI_projects/mCLM/data/instruction_onlyblocks_top_500/ \
    --synthetic_data_path /home/a-m/cne2/MMLI_projects/mCLM/data/synthetic_onlyblocks_top_500/ \
    --downsample_tulu 0.1 \
    --tokenizer_cache ../GNN_input_cache/Total.molecule_tokenizer.500.pth \
    --load_ckpt ckpts/OnlyBlocks/Qwen2.5-3B_TotalTop1k_Adaptor_splitLoss/latest_checkpoint-epoch=00-step=25000.ckpt \
    --pretrained_embeddings final_embeddings/OnlyBlocks/Top500/128_dim/ \

fi

if false; then
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHONPATH=. srun python mCLM/scripts/main.py --base_model /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-3B/ \
    --pretrained_text_model /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-3B/ \
    --pretrained_tokenizer /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-3B/ \
    --check_val_every_n_steps 10000 \
    --batch_size=4 --lr 2e-5 --mol_lr 2e-6 \
    --ckpt_path ckpts/OnlyBlocks/Qwen2.5-3B_TotalTop1k_splitLoss_10kPretrain/ --version Qwen2.5-3B_splitLoss \
    --max_epochs 2 \
    --no_PEFT \
    --accumulate_grad_batches 4 \
    --data_module TotalTopK --task TotalTop500 \
    --freeze_GNN \
    --num_warmup_steps 2000 \
    --save_checkpoint_every_n_steps 1000 \
    --pretrained_embeddings final_embeddings/OnlyBlocks/Top500/128_dim/ \
    --instruction_data_path /home/a-m/cne2/MMLI_projects/mCLM/data/instruction_onlyblocks_top_500/ \
    --synthetic_data_path /home/a-m/cne2/MMLI_projects/mCLM/data/synthetic_onlyblocks_top_500/ \
    --tokenizer_cache ../GNN_input_cache/Total.molecule_tokenizer.500.pth \
    --load_from_ckpt ckpts/OnlyBlocks/Qwen2.5-3B_TotalTop1k_splitLoss_1kPretrain/latest_checkpoint-epoch=04-step=129000.ckpt \
    --only_text_loss \

fi


if false; then
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

#export TORCH_DISTRIBUTED_DEBUG=DETAIL

PYTHONPATH=. srun python mCLM/scripts/main.py --base_model /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-3B/ \
    --pretrained_text_model /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-3B/ \
    --pretrained_tokenizer /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-3B/ \
    --check_val_every_n_steps 10000 \
    --batch_size=4 --lr 2e-5 --mol_lr 2e-6 \
    --ckpt_path ckpts/OnlyBlocks/Qwen2.5-3B_TotalTop1k_GNN_splitLoss/ --version Qwen2.5-3B_PreGNN_splitLoss \
    --max_epochs 5 \
    --no_PEFT \
    --accumulate_grad_batches 4 \
    --data_module TotalTopK --task TotalTop500 \
    --num_warmup_steps 2000 \
    --save_checkpoint_every_n_steps 1000 \
    --instruction_data_path /home/a-m/cne2/MMLI_projects/mCLM/data/instruction_onlyblocks_top_500/ \
    --synthetic_data_path /home/a-m/cne2/MMLI_projects/mCLM/data/synthetic_onlyblocks_top_500/ \
    --downsample_tulu 0.1 \
    --load_GNN_ckpt ckpts_GNN/OnlyBlocks/128_dim/best_val_checkpoint.ckpt \
    --tokenizer_cache ../GNN_input_cache/Total.molecule_tokenizer.500.pth \
    
    #--freeze_GNN \
    #--pretrained_embeddings final_embeddings/OnlyBlocks/Top500/128_dim/ \

fi

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


if false; then

PYTHONPATH=. srun python mCLM/scripts/main.py --base_model /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-0.5B/ \
    --pretrained_text_model /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-0.5B/ --check_val_every_n_steps 5000 \
    --pretrained_tokenizer /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-0.5B/ \
    --batch_size=16 --lr 1e-4 \
    --ckpt_path ckpts/Qwen2.5-0.5B_Total_PreGNN/ --version Qwen2.5-0.5B_PreGNN_FastV2 \
    --max_epochs 3 \
    --data_module Total --task Total \
    --save_checkpoint_every_n_steps 2500 \
    --num_warmup_steps 2000 \
    --load_GNN_ckpt ckpts_GNN/1536_dim/best_val_checkpoint.ckpt \
    --GNN_cache ../GNN_input_cache/Total.molecule_tokenizer.v3.graphs.pth \

#    --max_negative_sampling_schedule 15000 \
#    --negative_sampling_schedule_loss 0.1 \
#    --pretrained_embeddings final_embeddings/OnlyBlocks/1536_dim/ \

fi

if false; then

PYTHONPATH=. srun python mCLM/scripts/main.py --base_model /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-0.5B/ \
    --pretrained_text_model /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-0.5B/ --check_val_every_n_steps 5000 \
    --pretrained_tokenizer /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-0.5B/ \
    --batch_size=16 --lr 1e-4 \
    --ckpt_path ckpts/Qwen2.5-0.5B_SMolInstruct_PreGNN/ --version Qwen2.5-0.5B_PreGNN_FastV2 \
    --max_epochs 3 \
    --data_module SMolInstruct --task SMolInstruct \
    --save_checkpoint_every_n_steps 2500 \
    --num_warmup_steps 2000 \
    --load_GNN_ckpt ckpts_GNN/1536_dim/best_val_checkpoint.ckpt \
    --GNN_cache ../GNN_input_cache/Total.molecule_tokenizer.v2.graphs.pth \
    --max_negative_sampling_schedule 15000 \
    --negative_sampling_schedule_loss 0.1 \

fi

if false; then #test out the negative sampling code with trainable GNN

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHONPATH=. srun python mCLM/scripts/main.py --base_model /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-0.5B/ \
    --pretrained_text_model /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-0.5B/ \
    --pretrained_tokenizer /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-0.5B/ \
    --batch_size=4 --lr 1e-4 \
    --ckpt_path ckpts/Qwen2.5-0.5B_SMolInstruct_PreGNN/ --version Qwen2.5-0.5B_PreGNN_FastV2_OnlyBlocks \
    --max_epochs 3 \
    --data_module SMolInstruct --task SMolInstruct \
    --num_warmup_steps 2000 \
    --GNN_cache ../GNN_input_cache/Total.molecule_tokenizer.v4.graphs.pth \

    #--load_GNN_ckpt ckpts_GNN/OnlyBlocks/128_dim/best_val_checkpoint.ckpt \
    #that GNN is too big :(

fi


if false; then

PYTHONPATH=. srun python mCLM/scripts/main.py --base_model /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-0.5B/ \
    --pretrained_text_model /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-0.5B/ \
    --pretrained_tokenizer /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-0.5B/ \
    --check_val_every_n_steps 10000 \
    --batch_size=32 --lr 2e-5 --mol_lr 2e-6 \
    --ckpt_path ckpts/OnlyBlocks/Qwen2.5-0.5B_Total_NoGNN_splitLR/ --version Qwen2.5-0.5B_NoGNN_FastV2_Shrink25k_OnlyBlocks2_splitLR \
    --max_epochs 3 \
    --data_module Total --task Total \
    --freeze_GNN \
    --num_warmup_steps 2000 \
    --pretrained_embeddings final_embeddings/OnlyBlocks/128_dim/ \

#    --resume_from_checkpoint ckpts/Qwen2.5-0.5B_SMolInstruct_FreePreGNN/latest_checkpoint-epoch=00-step=20000.ckpt \

fi



if false; then

PYTHONPATH=. srun python mCLM/scripts/main.py --base_model /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-0.5B/ \
    --pretrained_text_model /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-0.5B/ \
    --pretrained_tokenizer /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-0.5B/ \
    --batch_size=4 --lr 2e-5 \
    --ckpt_path ckpts/OnlyBlocks/Qwen2.5-0.5B_Kinase_NoGNN_noPEFT_splitLoss/ --version Qwen2.5-0.5B_NoGNN_FastV3_OnlyBlocks2_splitLoss \
    --max_epochs 100 \
    --no_PEFT \
    --data_module Kinase --task Kinase \
    --freeze_GNN \
    --num_warmup_steps 200 \
    --pretrained_embeddings final_embeddings/OnlyBlocks/Kinase/128_dim/ \

fi

if false; then

PYTHONPATH=. srun python mCLM/scripts/main.py --base_model /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-0.5B/ \
    --pretrained_text_model /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-0.5B/ \
    --pretrained_tokenizer /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-0.5B/ \
    --batch_size=32 --lr 1e-4 \
    --ckpt_path ckpts/OnlyBlocks/Qwen2.5-0.5B_Kinase_NoGNN_lowLR_splitLoss/ --version Qwen2.5-0.5B_NoGNN_FastV3_OnlyBlocks2_lowLR_splitLoss \
    --max_epochs 100 \
    --data_module Kinase --task Kinase \
    --freeze_GNN \
    --num_warmup_steps 200 \
    --pretrained_embeddings final_embeddings/OnlyBlocks/Kinase/128_dim/ \

fi


if false; then

PYTHONPATH=. srun python mCLM/scripts/main.py --base_model /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-0.5B/ \
    --pretrained_text_model /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-0.5B/ \
    --pretrained_tokenizer /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-0.5B/ \
    --check_val_every_n_steps 10000 \
    --batch_size=32 --lr 2e-5 --mol_lr 2e-6 \
    --ckpt_path ckpts/OnlyBlocks/Qwen2.5-0.5B_TotalTop50kV2_NoGNN_splitLR_splitLoss/ --version Qwen2.5-0.5B_NoGNN_splitLR_splitLoss \
    --max_epochs 10 \
    --no_PEFT \
    --data_module TotalTop50k --task TotalTop50k \
    --freeze_GNN \
    --num_warmup_steps 2000 \
    --save_checkpoint_every_n_steps 1000 \
    --pretrained_embeddings final_embeddings/OnlyBlocks/Top50kV2/128_dim/ \
    --resume_from_checkpoint ckpts/OnlyBlocks/Qwen2.5-0.5B_TotalTop50kV2_NoGNN_splitLR_splitLoss/latest_checkpoint-epoch=00-step=40000.ckpt \

fi



if false; then

PYTHONPATH=. srun python mCLM/scripts/main.py --base_model /home/a-m/cne2/MMLI_projects/LLMs/Qwen3-0.6B-Base/ \
    --pretrained_text_model /home/a-m/cne2/MMLI_projects/LLMs/Qwen3-0.6B-Base/ \
    --pretrained_tokenizer /home/a-m/cne2/MMLI_projects/LLMs/Qwen3-0.6B-Base/ \
    --check_val_every_n_steps 10000 \
    --batch_size=32 --lr 2e-5 --mol_lr 2e-6 \
    --ckpt_path ckpts/OnlyBlocks/Qwen3-0.6B-Base_TotalTop50k_NoGNN_splitLR_splitLoss/ --version Qwen3-0.6B-Base_NoGNN_FastV3_OnlyBlocks2_splitLR_splitLoss \
    --max_epochs 10 \
    --no_PEFT \
    --data_module TotalTop50k --task TotalTop50k \
    --freeze_GNN \
    --num_warmup_steps 2000 \
    --pretrained_embeddings final_embeddings/OnlyBlocks/Top50kV2/128_dim/ \

fi


if false; then
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#export CUDA_LAUNCH_BLOCKING=1
PYTHONPATH=. srun python mCLM/scripts/main.py --base_model /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-3B/ \
    --pretrained_text_model /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-3B/ \
    --pretrained_tokenizer /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-3B/ \
    --check_val_every_n_steps 10000 \
    --batch_size=4 --lr 2e-5 --mol_lr 2e-6 \
    --trunc_length 512 \
    --ckpt_path ckpts/OnlyBlocks/Qwen2.5-3B_SMolInstructTop50k_NoGNN_SplitLR_splitLoss/ --version Qwen2.5-3B_NoGNN_FastV3_OnlyBlocks2_SplitLR_splitLoss \
    --max_epochs 20 \
    --accumulate_grad_batches 2 \
    --no_PEFT \
    --data_module SMolInstructTop50k --task SMolInstructTop50k \
    --save_checkpoint_every_n_steps 2500 \
    --freeze_GNN \
    --num_warmup_steps 2000 \
    --pretrained_embeddings final_embeddings/OnlyBlocks/Top50k/128_dim/ \

fi

if false; then
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#export CUDA_LAUNCH_BLOCKING=1
PYTHONPATH=. srun python mCLM/scripts/main.py --base_model /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-7B/ \
    --pretrained_text_model /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-7B/ \
    --pretrained_tokenizer /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-7B/ \
    --check_val_every_n_steps 5000 \
    --batch_size=1 --lr 2e-5 --mol_lr 2e-6 \
    --trunc_length 512 \
    --ckpt_path ckpts/OnlyBlocks/Qwen2.5-7B_MolGenTotal_NoGNN_SplitLR_splitLoss/ --version Qwen2.5-7B_NoGNN_FastV3_OnlyBlocks2_SplitLR_splitLoss \
    --max_epochs 20 \
    --accumulate_grad_batches 4 \
    --use_deepspeed
    --no_PEFT \
    --data_module MolGenTotal --task MolGenTotal \
    --save_checkpoint_every_n_steps 2500 \
    --freeze_GNN \
    --num_warmup_steps 2000 \
    --pretrained_embeddings final_embeddings/OnlyBlocks/128_dim/ \
fi


if false; then

PYTHONPATH=. srun python mCLM/scripts/main.py --base_model /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-0.5B/ \
    --pretrained_text_model /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-0.5B/ \
    --pretrained_tokenizer /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-0.5B/ \
    --batch_size=32 --lr 2e-5 --mol_lr 2e-6 \
    --ckpt_path ckpts/OnlyBlocks/Qwen2.5-0.5B_MolGenSMolInstruct_NoGNN_splitLR_splitLoss/ --version Qwen2.5-0.5B_NoGNN_FastV3_OnlyBlocks2_splitLR_splitLoss \
    --max_epochs 20 \
    --no_PEFT \
    --data_module MolGenSMolInstruct --task MolGenSMolInstruct \
    --freeze_GNN \
    --num_warmup_steps 2000 \
    --pretrained_embeddings final_embeddings/OnlyBlocks/128_dim/ \

    #--only_molecule_loss \
fi

if false; then

PYTHONPATH=. srun python mCLM/scripts/main.py --base_model /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-0.5B/ \
    --pretrained_text_model /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-0.5B/ \
    --pretrained_tokenizer /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-0.5B/ \
    --batch_size=32 --lr 2e-5 --mol_lr 2e-6 \
    --ckpt_path ckpts/OnlyBlocks/Qwen2.5-0.5B_SMolInstruct_NoGNN_noPEFT_splitLR_splitLoss/ --version Qwen2.5-0.5B_NoGNN_FastV3_OnlyBlocks2_splitLR_splitLoss \
    --max_epochs 3 \
    --no_PEFT \
    --data_module SMolInstruct --task SMolInstruct \
    --freeze_GNN \
    --num_warmup_steps 2000 \
    --pretrained_embeddings final_embeddings/OnlyBlocks/128_dim/ \

#    --max_negative_sampling_schedule 500000 \
#    --negative_sampling_schedule_loss 0.1 \

#    --resume_from_checkpoint ckpts/Qwen2.5-0.5B_SMolInstruct_FreePreGNN/latest_checkpoint-epoch=00-step=20000.ckpt \

fi

if false; then
PYTHONPATH=. srun python mCLM/scripts/main.py --base_model /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-7B/ \
    --pretrained_text_model /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-7B/ \
    --pretrained_tokenizer /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-7B/ \
    --check_val_every_n_steps 5000 \
    --batch_size=3 --lr 2e-5 --mol_lr 2e-6 \
    --ckpt_path ckpts/OnlyBlocks/Qwen2.5-7B_SMolInstruct_NoGNN_SplitLR/ --version Qwen2.5-7B_NoGNN_FastV3_OnlyBlocks2_SplitLR \
    --max_epochs 3 \
    --data_module SMolInstruct --task SMolInstruct \
    --save_checkpoint_every_n_steps 2500 \
    --freeze_GNN \
    --num_warmup_steps 2000 \
    --pretrained_embeddings final_embeddings/OnlyBlocks/128_dim/ \

fi


if false; then

PYTHONPATH=. srun python mCLM/scripts/main.py --base_model /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-7B/ \
    --pretrained_text_model /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-7B/ --check_val_every_n_steps 10000 \
    --pretrained_tokenizer /home/a-m/cne2/MMLI_projects/LLMs/Qwen2.5-7B/ \
    --batch_size=3 --lr 1e-4 \
    --ckpt_path ckpts/Qwen2.5-7B_SMolInstruct_NoGNN/ --version Qwen2.5-7B_NoGNN_FastV2 \
    --max_epochs 3 \
    --data_module SMolInstruct --task SMolInstruct \
    --save_checkpoint_every_n_steps 2500 \
    --freeze_GNN \
    --num_warmup_steps 2000 \
    --pretrained_embeddings final_embeddings/1536_dim/ \

#    --resume_from_checkpoint ckpts/Qwen2.5-7B_SMolInstruct_NoGNN/latest_checkpoint-epoch=00-step=10000.ckpt \
#    --max_negative_sampling_schedule 1000000 \
#    --negative_sampling_schedule_loss 0.1 \

fi

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

