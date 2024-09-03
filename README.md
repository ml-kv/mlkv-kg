## Operating system and software dependency requirements
Ubuntu (20.04 LTS x86/64) packages
```
sudo apt-get update
sudo apt-get install -y build-essential python3-dev python3-pip make cmake
```
Python packages
```
conda create -n MLKV-KG python=3.10
conda activate MLKV-KG
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

## DGLKE
* DGL-KE multi-GPU and async-GPU training is only compatible with PyTorch <= 1.6 and CUDA 10.2

Build
```
git clone -b dgl-v0.9.1 git@github.com:ml-kv/mlkv-gnn.git dgl-v0.9.1
cd dgl-v0.9.1
git submodule update --init --recursive
mkdir build && cd build
cmake ../ -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON
make -j16
cd ../python
python3 setup.py install --user
pip3 uninstall dgl
```
Benchmark
```
git clone -b main git@github.com:ml-kv/mlkv-kg.git mlkv-kg
cd mlkv-kg/dglke/
python3 train.py --model DistMult --dataset FB15k --batch_size 1024  \
    --neg_sample_size 256 --hidden_dim 400 --gamma 143.0 --lr 0.08 --batch_size_eval 16  \
    --valid --test -adv --gpu 0 --num_proc 1 --num_thread 12 --max_step 40000 --mix_cpu_gpu \
    --log_interval 1000 --eval_interval 40000
nohup python3 -u train.py --model DistMult --dataset wikikg2 --batch_size 1024 --batch_size_eval 16 \
    --neg_sample_size 256 --neg_sample_size_eval 500 --hidden_dim 400 --gamma 500 --lr 0.1  \
    --valid --test -adv --gpu 0 --num_proc 1 --num_thread 12 --max_step 100000 --mix_cpu_gpu \
    --log_interval 5000 --eval_interval 5000 --no_save_emb &
python3 train.py --model DistMult --dataset Freebase --batch_size 1024 --batch_size_eval 16  \
    --neg_sample_size 256 --neg_sample_size_eval 256 --eval_percent 0.01 --hidden_dim 100 --gamma 143.0 --lr 0.08 \
    --valid --test -adv --gpu 0 --num_proc 1 --num_thread 12 --max_step 40000 --mix_cpu_gpu \
    --log_interval 1000 --eval_interval 40000 --no_save_emb
```
Benchmark Async
```
cd mlkv-kg/dglke-async/
python3 train.py --model DistMult --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 400 --gamma 143.0 --lr 0.08 --batch_size_eval 16 \
    --valid --test -adv --gpu -1 --num_proc 4 --num_thread 8 --max_step 10000 \
    --log_interval 1000 --eval_interval 40000 --async_update --embedding_staleness 16
python3 train.py --model DistMult --dataset Freebase --batch_size 1024 --batch_size_eval 16  \
    --neg_sample_size 256 --neg_sample_size_eval 256 --eval_percent 0.01 --hidden_dim 100 --gamma 143.0 --lr 0.08 \
    --valid --test -adv --gpu 0 --num_proc 1 --num_thread 12 --max_step 40000 --mix_cpu_gpu \
    --log_interval 1000 --eval_interval 40000 --no_save_emb --async_update --embedding_staleness 16
```

## DGLKE with MLKV
Build
```
sudo apt-get install uuid-dev libaio-dev libtbb-dev
git clone -b dgl-v0.9.1-mlkv git@github.com:ml-kv/mlkv-gnn.git dgl-v0.9.1-mlkv
cd dgl-v0.9.1-mlkv
git submodule update --init --recursive
mkdir build && cd build
cmake ../ -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON
make -j16
cd ../python
python3 setup.py install --user
pip3 uninstall dgl
```
Benchmark
```
git clone -b main git@github.com:ml-kv/mlkv-kg.git mlkv-kg
cd mlkv-kg/mlkv/
python3 train.py --model DistMult --dataset FB15k --batch_size 1024  \
    --neg_sample_size 256 --hidden_dim 400 --gamma 143.0 --lr 0.08 --batch_size_eval 16  \
    -adv --gpu 0 --num_proc 1 --num_thread 12 --max_step 5000 --mix_cpu_gpu \
    --log_interval 1000 --no_save_emb
nohup python3 -u train.py --model DistMult --dataset wikikg2 --batch_size 1024 --batch_size_eval 16 \
    --neg_sample_size 256 --neg_sample_size_eval 500 --hidden_dim 400 --gamma 500 --lr 0.1  \
    -adv --gpu 0 --num_proc 1 --num_thread 12 --max_step 5000 --mix_cpu_gpu \
    --log_interval 1000 --no_save_emb &
python3 train.py --model DistMult --dataset Freebase --batch_size 1024 --batch_size_eval 16  \
    --neg_sample_size 256 --neg_sample_size_eval 256 --eval_percent 0.01 --hidden_dim 100 --gamma 143.0 --lr 0.08 \
    -adv --gpu 0 --num_proc 1 --num_thread 12 --max_step 5000 --mix_cpu_gpu \
    --log_interval 1000 --no_save_emb
```
Benchmark Async
```
cd mlkv-kg/mlkv-async/
python3 -u train.py --model DistMult --dataset wikikg2 --batch_size 1024 --batch_size_eval 16 \
    --neg_sample_size 256 --neg_sample_size_eval 500 --hidden_dim 400 --gamma 12 --lr 0.1  \
    -adv --gpu 0 --num_proc 1 --num_thread 12 --max_step 5000 --mix_cpu_gpu \
    --log_interval 1000 --eval_interval 5000 --no_save_emb --embedding_staleness 16
```
