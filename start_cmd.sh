#！/bin/bash

python run_hyper.py --model=DirectAU --dataset=Beauty --config_files='recbole/properties/overall.yaml recbole/properties/model/DirectAU.yaml recbole/properties/dataset/sample.yaml' --params_file=directau_beauty1.hyper --output_file=./hyper/1027_Directau_Beauty1.result > ./hyper/beauty1.log 2>&1
python run_hyper.py --model=DirectAU --dataset=Beauty --config_files='recbole/properties/overall.yaml recbole/properties/model/DirectAU.yaml recbole/properties/dataset/sample.yaml' --params_file=directau_beauty2.hyper --output_file=./hyper/1027_Directau_Beauty2.result > ./hyper/beauty2.log 2>&1

python run_hyper.py --model=DirectAU --dataset=Yelp --config_files='recbole/properties/overall.yaml recbole/properties/model/DirectAU.yaml recbole/properties/dataset/sample.yaml' --params_file=directau_yelp1.hyper --output_file=./hyper/1027_Directau_Yelp1.result > ./hyper/yelp1.log 2>&1
python run_hyper.py --model=DirectAU --dataset=Yelp --config_files='recbole/properties/overall.yaml recbole/properties/model/DirectAU.yaml recbole/properties/dataset/sample.yaml' --params_file=directau_yelp2.hyper --output_file=./hyper/1027_Directau_Yelp2.result > ./hyper/yelp2.log 2>&1

python run_hyper.py --model=DirectAU --dataset=Gowalla --config_files='recbole/properties/overall.yaml recbole/properties/model/DirectAU.yaml recbole/properties/dataset/sample.yaml' --params_file=directau_gowalla1.hyper --output_file=./hyper/1027_Directau_Gowalla1.result > ./hyper/gowalla1.log 2>&1
python run_hyper.py --model=DirectAU --dataset=Gowalla --config_files='recbole/properties/overall.yaml recbole/properties/model/DirectAU.yaml recbole/properties/dataset/sample.yaml' --params_file=directau_gowalla2.hyper --output_file=./hyper/1027_Directau_Gowalla2.result > ./hyper/gowalla2.log 2>&1


# 将angli与uniform分开

python run_hyper.py --model=RauCL --dataset=Beauty --config_files='recbole/properties/overall.yaml recbole/properties/model/RauCL.yaml recbole/properties/dataset/sample.yaml' --params_file=1116raucl_beauty1.hyper --output_file=./hyper/1116_RauCL_Beauty1.result > ./hyper/1116Beauty1.log 2>&1
python run_hyper.py --model=RauCL --dataset=Beauty --config_files='recbole/properties/overall.yaml recbole/properties/model/RauCL.yaml recbole/properties/dataset/sample.yaml' --params_file=1116raucl_beauty2.hyper --output_file=./hyper/1116_RauCL_Beauty2.result > ./hyper/1116Beauty2.log 2>&1

python run_hyper.py --model=RauCL --dataset=Yelp --config_files='recbole/properties/overall.yaml recbole/properties/model/RauCL.yaml recbole/properties/dataset/sample.yaml' --params_file=1116raucl_yelp1.hyper --output_file=./hyper/1116_RauCL_Yelp1.result > ./hyper/1116Yelp1.log 2>&1
python run_hyper.py --model=RauCL --dataset=Yelp --config_files='recbole/properties/overall.yaml recbole/properties/model/RauCL.yaml recbole/properties/dataset/sample.yaml' --params_file=1116raucl_yelp2.hyper --output_file=./hyper/1116_RauCL_Yelp2.result > ./hyper/1116Yelp2.log 2>&1


python run_hyper.py --model=RauCL --dataset=Gowalla --config_files='recbole/properties/overall.yaml recbole/properties/model/RauCL.yaml recbole/properties/dataset/sample.yaml' --params_file=1116raucl_gowalla1.hyper --output_file=./hyper/1116_RauCL_gowalla1.result > ./hyper/1116Gowalla1.log 2>&1


# 方法一
python run_recbole.py --d 0 --alpha 0.7 --gamma 0.5 --std 10 --model=RauCL --dataset=Yelp --learning_rate=1e-3 --weight_decay=1e-6 --encoder=MF --train_batch_size=1024 > ./log_zc/1123Yelp.log 2>&1



# 方法一大范围调参

python run_hyper.py --model=RauCL --dataset=Yelp --config_files='recbole/properties/overall.yaml recbole/properties/model/RauCL.yaml recbole/properties/dataset/sample.yaml' --params_file=1123raucl_yelp1.hyper --output_file=./hyper/1123_RauCL_Yelp1.result > ./hyper/1123Yelp1.log 2>&1
python run_hyper.py --model=RauCL --dataset=Yelp --config_files='recbole/properties/overall.yaml recbole/properties/model/RauCL.yaml recbole/properties/dataset/sample.yaml' --params_file=1123raucl_yelp2.hyper --output_file=./hyper/1123_RauCL_Yelp2.result > ./hyper/1123Yelp2.log 2>&1

python run_hyper.py --model=RauCL --dataset=Beauty --config_files='recbole/properties/overall.yaml recbole/properties/model/RauCL.yaml recbole/properties/dataset/sample.yaml' --params_file=1123raucl_beauty1.hyper --output_file=./hyper/1123_RauCL_Beauty1.result > ./hyper/1123Beauty1.log 2>&1
python run_hyper.py --model=RauCL --dataset=Beauty --config_files='recbole/properties/overall.yaml recbole/properties/model/RauCL.yaml recbole/properties/dataset/sample.yaml' --params_file=1123raucl_beauty2.hyper --output_file=./hyper/1123_RauCL_Beauty2.result > ./hyper/1123Beauty2.log 2>&1
python run_hyper.py --model=RauCL --dataset=Beauty --config_files='recbole/properties/overall.yaml recbole/properties/model/RauCL.yaml recbole/properties/dataset/sample.yaml' --params_file=1123raucl_beauty3.hyper --output_file=./hyper/1123_RauCL_Beauty3.result > ./hyper/1123Beauty3.log 2>&1
python run_hyper.py --model=RauCL --dataset=Beauty --config_files='recbole/properties/overall.yaml recbole/properties/model/RauCL.yaml recbole/properties/dataset/sample.yaml' --params_file=1123raucl_beauty4.hyper --output_file=./hyper/1123_RauCL_Beauty4.result > ./hyper/1123Beauty4.log 2>&1


python run_hyper.py --model=RauCL --dataset=Gowalla --config_files='recbole/properties/overall.yaml recbole/properties/model/RauCL.yaml recbole/properties/dataset/sample.yaml' --params_file=1123raucl_gowalla2.hyper --output_file=./hyper/1123_RauCL_Gowalla2.result > ./hyper/1123Gowalla2.log 2>&1
python run_hyper.py --model=RauCL --dataset=Gowalla --config_files='recbole/properties/overall.yaml recbole/properties/model/RauCL.yaml recbole/properties/dataset/sample.yaml' --params_file=1123raucl_gowalla1.hyper --output_file=./hyper/1123_RauCL_Gowalla1.result > ./hyper/1123Gowalla1.log 2>&1


python run_hyper.py --model=RauCL --dataset=Yelp --config_files='recbole/properties/overall.yaml recbole/properties/model/RauCL.yaml recbole/properties/dataset/sample.yaml' --params_file=1127raucl_yelp1.hyper --output_file=./hyper/1127_RauCL_Yelp1.result > ./hyper/1127Yelp1.log 2>&1

python run_hyper.py --model=RauCL --dataset=Beauty --config_files='recbole/properties/overall.yaml recbole/properties/model/RauCL.yaml recbole/properties/dataset/sample.yaml' --params_file=1127raucl_beauty1.hyper --output_file=./hyper/1127_RauCL_Beauty1.result > ./hyper/1127Beauty1.log 2>&1
python run_hyper.py --model=RauCL --dataset=Beauty --config_files='recbole/properties/overall.yaml recbole/properties/model/RauCL.yaml recbole/properties/dataset/sample.yaml' --params_file=1127raucl_beauty2.hyper --output_file=./hyper/1127_RauCL_Beauty2.result > ./hyper/1127Beauty2.log 2>&1