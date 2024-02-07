# scaffold split

# 50 shot
python prompt_tuning.py --model_file pretrained_models/edgepred.pth --dataset bace --full_few few --num_layers 2 --n_codebooks 20 --n_samples 10 --lr 0.0001
python prompt_tuning.py --model_file pretrained_models/edgepred.pth --dataset sider --full_few few --num_layers 2 --n_codebooks 10 --n_samples 5 --lr 0.005
python prompt_tuning.py --model_file pretrained_models/edgepred.pth --dataset bbbp --full_few few --num_layers 3 --n_codebooks 20 --n_samples 10 --lr 0.005
python prompt_tuning.py --model_file pretrained_models/edgepred.pth --dataset toxcast --full_few few --num_layers 4  --n_codebooks 50 --n_samples 10 --lr 0.0001
python prompt_tuning.py --model_file pretrained_models/edgepred.pth --dataset clintox --full_few few --num_layers 4  --n_codebooks 50 --n_samples 10 --lr 0.0001 --gamma 0.5
python prompt_tuning.py --model_file pretrained_models/edgepred.pth --dataset tox21 --full_few few --num_layers 3  --n_codebooks 50 --n_samples 10 --lr 0.0005

# full shot
python prompt_tuning.py --model_file pretrained_models/edgepred.pth --dataset clintox --num_layers 3  --n_codebooks 50 --n_samples 10 --lr 0.0005
python prompt_tuning.py --model_file pretrained_models/edgepred.pth --dataset bbbp  --num_layers 2  --n_codebooks 50 --n_samples 10 --lr 0.0001
python prompt_tuning.py --model_file pretrained_models/edgepred.pth --dataset bace  --num_layers 3  --n_codebooks 50 --n_samples 10 --lr 0.0005
python prompt_tuning.py --model_file pretrained_models/edgepred.pth --dataset sider --num_layers 4  --n_codebooks 20 --n_samples 10 --lr 0.0005
python prompt_tuning.py --model_file pretrained_models/edgepred.pth --dataset tox21  --num_layers 3  --n_codebooks 20 --n_samples 10 --lr 0.0001
python prompt_tuning.py --model_file pretrained_models/edgepred.pth --dataset toxcast --num_layers 4  --n_codebooks 20 --n_samples 10 --lr 0.001


# random split

# 50 shot
python prompt_tuning.py --model_file pretrained_models/edgepred.pth --dataset bbbp --full_few few --num_layers 4  --n_codebooks 20 --n_samples 5 --lr 0.001 --split random 
python prompt_tuning.py --model_file pretrained_models/edgepred.pth --dataset tox21 --full_few few --num_layers 2  --n_codebooks 20 --n_samples 5 --lr 0.001 --split random 
python prompt_tuning.py --model_file pretrained_models/edgepred.pth --dataset clintox --full_few few --num_layers 2  --n_codebooks 20 --n_samples 10 --lr 0.001 --split random 
python prompt_tuning.py --model_file pretrained_models/edgepred.pth --dataset toxcast --full_few few --num_layers 2  --n_codebooks 50 --n_samples 5 --lr 0.005 --split random 
python prompt_tuning.py --model_file pretrained_models/edgepred.pth --dataset bace --full_few few --num_layers 4  --n_codebooks 50 --n_samples 5 --lr 0.001 --split random 
python prompt_tuning.py --model_file pretrained_models/edgepred.pth --dataset sider --full_few few --num_layers 4  --n_codebooks 50 --n_samples 5 --lr 0.005 --split random

# full shot
python prompt_tuning.py --model_file pretrained_models/edgepred.pth --dataset toxcast --num_layers 4  --n_codebooks 20 --n_samples 5 --lr 0.001 --split random
python prompt_tuning.py --model_file pretrained_models/edgepred.pth --dataset tox21 --num_layers 4  --n_codebooks 20 --n_samples 5 --lr 0.001. --split random
python prompt_tuning.py --model_file pretrained_models/edgepred.pth --dataset bace --num_layers 4  --n_codebooks 20 --n_samples 5 --lr 0.001. --split random
python prompt_tuning.py --model_file pretrained_models/edgepred.pth --dataset bbbp --num_layers 2  --n_codebooks 20 --n_samples 10 --lr 0.001 --split random
python prompt_tuning.py --model_file pretrained_models/edgepred.pth --dataset sider --num_layers 2  --n_codebooks 20 --n_samples 10 --lr 0.001 --split random
python prompt_tuning.py --model_file pretrained_models/edgepred.pth --dataset clintox --num_layers 2  --n_codebooks 20 --n_samples 10 --lr 0.001 --split random
