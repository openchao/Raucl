# @Time   : 2020/7/20
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE
# @Time   : 2020/10/3, 2020/10/1
# @Author : Yupeng Hou, Zihan Lin
# @Email  : houyupeng@ruc.edu.cn, zhlin@ruc.edu.cn

import argparse

from recbole.quick_start import run_recbole
import yaml


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='BPR', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='ml-100k', help='name of datasets')
    parser.add_argument('--config_files', type=str, default=None, help='config files')
    parser.add_argument('--zc', type=float, default=None, help='config files')

    args, _ = parser.parse_known_args()

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    # for zc in [0.01,0.1,0.2,0.5,0.7,0.8,0.9,0.95,1.0]:
    
    # with open('recbole/properties/model/DirectAU.yaml', 'r') as file:
    #     data = yaml.load(file, Loader=yaml.FullLoader)
    #     data['zc'] = args.zc
    # print(data)
    # with open('recbole/properties/model/DirectAU.yaml', 'w') as file:
    #     yaml.dump(data, file)

    # run_recbole(model=args.model, dataset=args.dataset, config_file_list=config_file_list)







# nohup python run_recbole.py --model=DirectAU --dataset=Gowalla --learning_rate=1e-3 --weight_decay=1e-6 --gamma=5 --encoder=MF --train_batch_size=1024 >directau_gowalla.log

# nohup python run_recbole.py --model=DirectAU --dataset=Yelp --learning_rate=1e-3 --weight_decay=1e-6 --gamma=1 --encoder=MF --train_batch_size=1024 >directau_yelp208.log




