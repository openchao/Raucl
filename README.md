# RAU: Towards Regularized Alignment and Uniformity for Representation Learning in Recommendation

This is the official code for RAU in the paper "[RAU: Towards Regularized Alignment and Uniformity for Representation Learning in Recommendation](https://arxiv.org/abs/xxx.xx)". This code is implemented on [RecBole](https://github.com/RUCAIBox/RecBole).

## How to run

### Set conda environment
```
conda env create -f mawu.yaml
conda activate mawu
```

### Run commands for RAU on MF
```
python run_recbole.py --d=0.5 --alpha=0.9 --std=0.6 --gamma=0.35 --model=RauCL --dataset=Beauty --learning_rate=0.001 --train_batch_size=256 --weight_decay=1e-6 --encoder=MF
python run_recbole.py --d=0.3 --alpha=0.9 --std=1.5 --gamma=5.5 --model=RauCL --dataset=Gowalla --learning_rate=0.001 --train_batch_size=1024 --weight_decay=1e-6 --encoder=MF
python run_recbole.py --d=0 --alpha=0.7 --std=14 --gamma=0.5 --model=RauCL --dataset=Yelp --learning_rate=0.001 --train_batch_size=1024 --weight_decay=1e-6 --encoder=MF
```

### Run commands for RAU on LightGCN
```
python run_recbole.py --d=0.001 --alpha=0.9 --std=13 --gamma=2 --model=RauCL --dataset=Gowalla --learning_rate=0.001 --train_batch_size=1024 --weight_decay=1e-6 --encoder=LightGCN
python run_recbole.py --d=0.001 --alpha=0.9 --std=13 --gamma=2 --model=RauCL --dataset=Gowalla --learning_rate=0.001 --train_batch_size=1024 --weight_decay=1e-6 --encoder=LightGCN
python run_recbole.py --d=0.001 --alpha=0.9 --std=13 --gamma=2 --model=RauCL --dataset=Gowalla --learning_rate=0.001 --train_batch_size=1024 --weight_decay=1e-6 --encoder=LightGCN
```

## Citation
If you find our work helpful, please cite our paper.
```
@inproceedings{rau,
  title={RAU: Towards Regularized Alignment and Uniformity for Representation Learning in Recommendation},
  author={... and ...},
  booktitle={xxx},
  year={xxxx}
}
```

