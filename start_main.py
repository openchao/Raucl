import os
import time
from itertools import product
# args_list=[0.1,0.2,0.3,0.5,0.7,0.9]
# args_list=[0.1,0.2,0.5,0.3,0.7,1,0.9]
# weight_decay choice [1e-04,1e-05,1e-06,1e-07]
# args_list=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]


# 网格搜索超参数的
# 
# nohup python run_hyper.py --model=DirectAU --dataset=Beauty --config_files='recbole/properties/overall.yaml recbole/properties/model/DirectAU.yaml recbole/properties/dataset/sample.yaml' --params_file=directau.hyper --output_file=1010_DWUCL_Beauty_time1_1e6.result
# nohup python run_hyper.py --model=DirectAU --dataset=Yelp --config_files='recbole/properties/overall.yaml recbole/properties/model/DirectAU.yaml recbole/properties/dataset/sample.yaml' --params_file=directau.hyper --output_file=1011_DWUCL_Yelp_item0.5.result
# nohup python run_hyper.py --model=DirectAU --dataset=Aminer --config_files='recbole/properties/overall.yaml recbole/properties/model/DirectAU.yaml recbole/properties/dataset/sample.yaml' --params_file=directau.hyper --output_file=1009_BWUCL_Aminer_1e6_times8.result

# witem choice [0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.2,2.4,2.6,2.8,3]


# python run_recbole.py --model=DirectAU --dataset=Gowalla --learning_rate=1e-3 --weight_decay=1e-6 --gamma=5 --encoder=MF --train_batch_size=1024

# embsize 2048
# python run_recbole.py --model=DirectAU --dataset=Beauty --learning_rate=1e-3 --weight_decay=1e-06 --encoder=MF --train_batch_size=256 --embedding_size=2048 --BT=0.2 --bt=0.001 --wuser=0.2 --witem=0.8


#------------------------------------------------------------------------------------------------------
# 消融实验——去掉BT
# nohup python run_recbole.py --model=DirectAU --dataset=Beauty --learning_rate=1e-3 --weight_decay=1e-5 --encoder=MF --train_batch_size=256 --wuser=0.8 --witem=0.1 --BT=0 >abl_beauty_without_BT.log
# nohup python run_recbole.py --model=DirectAU --dataset=Beauty --learning_rate=1e-3 --weight_decay=1e-5 --encoder=MF --train_batch_size=256 --wuser=0.8 --witem=0.8 --BT=0.1 --bt=0.001 >abl_beauty_without_WU0.8.log

# nohup python run_recbole.py --model=DirectAU --dataset=Gowalla --learning_rate=1e-3 --weight_decay=1e-6 --gamma=5 --encoder=MF --train_batch_size=1024 --wuser=0.8 --witem=0.1 --BT=0

# nohup python run_recbole.py --model=DirectAU --dataset=Yelp --learning_rate=1e-3 --weight_decay=1e-6 --gamma=1 --encoder=MF --train_batch_size=1024 --wuser=2.4 --witem=0.6 --BT=0 >abl_yelp_without_BT.log

#------------------------------------------------------------------------------------------------------

# for i_arg in args_list:
#     # str_start = f"python run_recbole.py --model=DirectAU --dataset=Beauty --learning_rate=1e-3 --weight_decay=1e-06 --encoder=MF --train_batch_size=256 --bt=0 --BT={i_arg}"
#     # str_start = f"python3 run_recbole.py --model=DirectAU --dataset=Beauty --learning_rate=1e-3 --weight_decay={i_arg} --encoder=MF --train_batch_size=256"
#     # str_start = "python3 run_hyper.py --model=DirectAU --dataset=Beauty --learning_rate=1e-3 --weight_decay=1e-6 --encoder=MF --train_batch_size=256 --config_files=recbole/properties/model/DirectAU.yaml --params_file=hyper.test"
#     # python run_recbole.py     --model=DirectAU --dataset=Gowalla     --learning_rate=1e-3 --weight_decay=1e-6     --gamma=5 --encoder=MF --train_batch_size=1024

#     # gowalla 数据集
#     str_start= f"python run_recbole.py --model=DirectAU --dataset=Gowalla --learning_rate=1e-3 --weight_decay=1e-6 --gamma=5 --encoder=MF --train_batch_size=1024 --bt={i_arg}"
#     str_cmd = f"nohup {str_start} >0826_directau_gowalla+0.1*BT+{i_arg}*bt_gamma=5.log 2>&1 &"
#     # str_cmd = f"nohup {str_start} >0826_directau^3_gamma_{i_arg}_pow3.log 2>&1 &"
#     # str_cmd = f"nohup {str_start} >0825_directau_A=B={i_arg}_C=1.log 2>&1 &"
#     print(str_cmd)
#     time.sleep(3)
#     os.system(str_cmd)


# str_start_list= [
#     "python run_recbole.py --model=BUIR --dataset=Beauty --learning_rate=0.001 --train_batch_size=256 --gpu_id=9",
#     "python run_recbole.py --model=BUIR --dataset=Yelp --learning_rate=0.001 --train_batch_size=1024 --gpu_id=8",
#     "python run_recbole.py --model=BUIR --dataset=Aminer --learning_rate=0.001 --train_batch_size=256 --gpu_id=7",
#     "python run_recbole.py --model=CLRec --dataset=Beauty --learning_rate=0.001 --train_batch_size=256 --gpu_id=6",
#     "python run_recbole.py --model=CLRec --dataset=Yelp --learning_rate=0.001 --train_batch_size=1024 --gpu_id=5",
    # "python run_recbole.py --model=CLRec --dataset=Aminer --learning_rate=0.001 --train_batch_size=256 --gpu_id=4"]






# BCL+FCL

# nohup python run_recbole.py --model=DirectAU --dataset=Beauty --train_batch_size=256 --weight_decay=1e-5 --encoder=MF --wuser=0.8 --witem=0.1 >1009_best_BCL+FCL_beauty.log
# nohup python run_recbole.py --model=DirectAU --dataset=Yelp --train_batch_size=1024 --weight_decay=1e-6 --encoder=MF --wuser=2.1 --witem=0.2 >1009_best_BCL+FCL_yelp.log


# run_recbole.py --model=DirectAU --dataset=Aminer --learning_rate=0.001 --train_batch_size=256 --weight_decay=1e-6 --gamma=0.5 --encoder=MF --gpu_id=9
# python run_recbole.py --model=DirectAU --dataset=Yelp --learning_rate=1e-3 --weight_decay=1e-6 --gamma=3 --encoder=MF --train_batch_size=1024 --gpu_id=8 >1125_log.log 2>&1

str_start_list= [
    "python run_recbole.py --model=DirectAU --dataset=Beauty --learning_rate=0.001 --train_batch_size=256 --weight_decay=1e-6 --gamma=0.5 --encoder=MF --gpu_id=9",
    "python run_recbole.py --model=DirectAU --dataset=Yelp --learning_rate=0.001 --train_batch_size=1024 --weight_decay=1e-6 --gamma=1 --encoder=MF --gpu_id=8"]



for i_arg in str_start_list:
    model_name=(i_arg[i_arg.find("--model=")+len("--model="):i_arg.find("--dataset=")]).strip()
    dataset_name=(i_arg[i_arg.find("--dataset=")+len("--dataset="):i_arg.find("--learning_rate")]).strip()
    str_cmd = f"nohup {i_arg} >1017_{model_name}_{dataset_name}.log 2>&1 &"
    print(str_cmd)
    time.sleep(3)
    os.system(str_cmd)
