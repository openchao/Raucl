import os 


param_list=[1,2,3]
for i_param in param_list:
    try:    
        os.system(f"cp directau.hyper directau_{i_param}.hyper")
    except Exception as e:
        print("文件创建失败")
print("文件创建完毕")






# import argparse
# from recbole.config import Config
# from recbole.data import create_dataset, data_preparation
# from recbole.utils import  get_model, get_trainer
# from os.path import join



# def run_recbole(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True):
#     r""" A fast running api, which includes the complete process of
#     training and testing a model on a specified dataset

#     Args:
#         model (str): model name
#         dataset (str): dataset name
#         config_file_list (list): config files used to modify experiment parameters
#         config_dict (dict): parameters dictionary used to modify experiment parameters
#         saved (bool): whether to save the model
#     """
#     # configurations initialization
#     config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    
#     # dataset filtering
#     dataset = create_dataset(config)

#     # dataset splitting
#     train_data, _, _  = data_preparation(config, dataset)

#     # model loading and initialization
#     model = get_model(config['model'])(config, train_data).to(config['device'])
    
#     # trainer loading and initialization
#     trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

#     # model evaluation
#     saved_model_name = "DirectAU-Sep-29-2023_07-22-57.pth"
#     test_result = trainer.calculate_uniformity(load_best_model=saved,model_file=join("saved/",saved_model_name) ,show_progress=config['show_progress'])


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model', '-m', type=str, default='BPR', help='name of models')
#     parser.add_argument('--dataset', '-d', type=str, default='ml-100k', help='name of datasets')
#     parser.add_argument('--config_files', type=str, default=None, help='config files')

#     args, _ = parser.parse_known_args()

#     config_file_list = args.config_files.strip().split(' ') if args.config_files else None
#     run_recbole(model=args.model, dataset=args.dataset, config_file_list=config_file_list)




























