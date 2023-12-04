# 记录下来目前最好的checkpoint与log
# alpha:0.9, d:0.5, gamma:0.4, gpu_id:8, std:0.01 1129beauty3 (14.51)
# nohup python run_recbole.py --d=0.5 --alpha=0.9 --std=0.01 --gamma=0.4 --model=RauCL --dataset=Beauty --learning_rate=0.001 --train_batch_size=256 --weight_decay=1e-6 --encoder=MF --checkpoint_dir=saved_best --gpu_id=3 >log_zc/1201_DirectA_Beauty_best.log 2>&1 &

# alpha:0.7, d:0, gamma:0.5, gpu_id:3, std:14, train_batch_size:1024, weight_decay:1e-06 1123yelp2 (11.36)
# nohup python run_recbole.py --d=0 --alpha=0.7 --std=14 --gamma=0.5 --model=RauCL --dataset=Yelp --learning_rate=0.001 --train_batch_size=1024 --weight_decay=1e-6 --encoder=MF --checkpoint_dir=saved_best --gpu_id=4 >log_zc/1201_DirectA_Yelp_best.log 2>&1 &

# alpha:0.9, d:0.5, gamma:0.35, gpu_id:4, std:0.6 1201Beauty3(14.57)
# nohup python run_recbole.py --d=0.5 --alpha=0.9 --std=0.6 --gamma=0.35 --model=RauCL --dataset=Beauty --learning_rate=0.001 --train_batch_size=256 --weight_decay=1e-6 --encoder=MF --checkpoint_dir=saved_best --gpu_id=2 >log_zc/1202_DirectA_Beauty_best.log 2>&1 &

# alpha:0.9, d:0.1, gamma:5.5, gpu_id:9, std:1 1201Gowalla3(20.96)
# nohup python run_recbole.py --d=0.1 --alpha=0.9 --std=1 --gamma=5.5 --model=RauCL --dataset=Gowalla --learning_rate=0.001 --train_batch_size=1024 --weight_decay=1e-6 --encoder=MF --checkpoint_dir=saved_best --gpu_id=9 >log_zc/1203_DirectA_Gowalla_best.log 2>&1 &

# (20.98)
# nohup python run_recbole.py --d=0.3 --alpha=0.9 --std=1.5 --gamma=5.5 --model=RauCL --dataset=Gowalla --learning_rate=0.001 --train_batch_size=1024 --weight_decay=1e-6 --encoder=MF --checkpoint_dir=saved_best --gpu_id=9 >log_zc/1203_DirectA_Gowalla_best.log 2>&1 &
