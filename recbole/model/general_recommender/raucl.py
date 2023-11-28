# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType
import pdb
import pandas as pd

class RauCL(GeneralRecommender):
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(RauCL, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        
        self.encoder_name = config['encoder']
        print(f"this is encoder {self.encoder_name}...")

        # define layers and loss
        if self.encoder_name == 'MF':
            self.encoder = MFEncoder(self.n_users, self.n_items, self.embedding_size)
        elif self.encoder_name == 'LightGCN':
            self.n_layers = config['n_layers']
            self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
            self.norm_adj = self.get_norm_adj_mat().to(self.device)
            self.encoder = LGCNEncoder(self.n_users, self.n_items, self.embedding_size, self.norm_adj, self.n_layers)
        else:
            raise ValueError('Non-implemented Encoder.')

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # parameters initialization
        self.apply(xavier_normal_initialization)

        # DCL
        # self.bn = nn.BatchNorm1d(self.embedding_size, affine=False)

        # layers = []
        # embs = str(self.embedding_size) + '-' + str(self.embedding_size) + '-' + str(self.embedding_size)
        # sizes = [self.embedding_size] + list(map(int, embs.split('-')))
        # for i in range(len(sizes) - 2):
        #     layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
        #     layers.append(nn.BatchNorm1d(sizes[i + 1]))
        #     layers.append(nn.ReLU(inplace=True))
        # layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        # self.projector = nn.Sequential(*layers)# FCL
        # self.BT = config['BT']
        # self.bt = config['bt']
        # MAWU
        # self.w_user = config['wuser']
        # self.w_item = config['witem']
        self.times = config['times']
        # raucl
        # 方法一
        self.gamma = config['gamma'] # 控制alignment与uniform的比例，参数很敏感
        self.d = config['d'] # 控制中心距离的超参
        self.alpha = config['alpha'] #uniformity控制user与item的重要性
        self.std = config['std'] #控制user uniformity正则的超参，这个地方也可以整合成一个超参
        print(f"这是directau 的 RE 版本...gamma{self.gamma}, d{self.d}, alpha{self.alpha}, std{self.std}")

        # 方法二
        # self.d = config['d'] # 控制中心距离的超参
        # self.alpha_u = config['alpha_u'] #uniformity控制user与item的重要性
        # self.alpha_i = config['alpha_i'] #uniformity控制user与item的重要性
        # self.std_u = config['std_u'] #控制user uniformity正则的超参，这个地方也可以整合成一个超参
        # self.std_i = config['std_i'] #控制item uniformity正则的超参
        # self.alpha = config['alpha'] #控制angliment与uniformity重要性的超参


    def get_norm_adj_mat(self):
        # build adj matrix
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def forward(self, user, item):
        user_e, item_e = self.encoder(user, item)
        return F.normalize(user_e, dim=-1), F.normalize(item_e, dim=-1)

    @staticmethod
    def alignment(x, y, alpha=2):
        return (x - y).norm(p=2, dim=1).pow(alpha).mean()

    @staticmethod
    def uniformity(x, t=2):
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()
    
    def uniformity_distributed(self, x, t=2):
        # 方法一直接用d的var
        # d = torch.pdist(x, p=2)
        # return d.pow(2).mul(-t).exp().mean().log()+4*d.var()
        
        # 方法二让d服从指数分布
        d = torch.pdist(x, p=2).pow(2).mul(-t).exp() #[batch, 1]
        return d.mean().log()+self.std*d.std() # 改
    
        # 方法三去除偏远值或者让长尾的item分布的集中一些

    @staticmethod
    def getD(x, t=2):
        # 获取样本之间的距离
        d = torch.pdist(x, p=2).pow(2).mul(-t).exp() #[batch, 1]
        return d
    
    @staticmethod
    def uniformity_std(d, t=2):
        # 将regular uniformity 
        return d.std()
    
    @staticmethod
    def uniformity_d(d, t=2):
        # 将uniformity
        return d.mean().log() 
        # 方法三去除偏远值或者让长尾的item分布的集中一些


    @staticmethod
    def off_diagonal(x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    
    # @staticmethod
    def vicreg_off_diagonal_loss(self,x_a,x_b):
        x_a = x_a - x_a.mean(dim=0)
        x_b = x_b - x_b.mean(dim=0)

        cov_x_a = (x_a @ x_a.T) / (x_a.shape[0]-1)
        cov_x_b = (x_b @ x_b.T) / (x_b.shape[0]-1)
        cov_loss = self.off_diagonal(cov_x_a).pow_(2).sum() + self.off_diagonal(cov_x_b).pow_(2).sum()
        # pdb.set_trace()
        return cov_loss
        

    
    def bt_loss(self, x, y):
        user_e = self.projector(x) 
        item_e = self.projector(y) 
        c = self.bn(user_e).T @ self.bn(item_e)# 
        c.div_(user_e.size()[0])
        # sum the cross-correlation matrix
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().div(self.embedding_size)     #0.9803
        off_diag = self.off_diagonal(c).pow_(2).sum().div(self.embedding_size)      # 0.2578
        return  on_diag + self.bt * off_diag
        # return self.BT * on_diag
        # return self.bt * off_diag

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_e, item_e = self.forward(user, item)
        align = self.alignment(user_e, item_e)  # 1.9909
        # align正则
        center_u = user_e.mean(dim=0)# 聚类中心 [1,64]
        center_i = item_e.mean(dim=0)# 聚类中心 [1,64]
        Ralign = torch.norm(center_u - center_i)

        # 第一个版本
        uniform =  self.gamma * (self.alpha*self.uniformity_distributed(user_e) + (1-self.alpha)*self.uniformity_distributed(item_e))
        return align + uniform + self.d*(Ralign) 
        # 第二个版本，将uniform与正则分开，参数变多了，但是我不会调，所以效果不好
        d_u = self.getD(user_e)
        d_i = self.getD(item_e)
        uniform = self.gamma * (self.alpha_u*self.uniformity_d(d_u)+self.alpha_i*self.uniformity_d(d_i))
        Runiform = self.std_u*self.uniformity_std(d_u)+self.std_i*self.uniformity_std(d_i)

        
        
        # print(align,uniform,align+self.d*(distances))
        # return self.alpha*align + (1-self.alpha)*uniform + self.d*(Ralign) + Runiform
        return align + uniform + self.d*(Ralign) + Runiform
    
        # vicreg_loss = self.vicreg_loss(user_e,item_e)
        # mec_loss = self.MEC_loss(user_e,item_e,self.lamda)
        # return align +Enhanced_uniform
        # return uniform + align + vicreg_loss , vicreg_loss
        # return align + uniform,uniform

    # def calculate_loss(self, interaction):
    #     if self.restore_user_e is not None or self.restore_item_e is not None:
    #         self.restore_user_e, self.restore_item_e = None, None

    #     user = interaction[self.USER_ID]
    #     item = interaction[self.ITEM_ID]

    #     user_e, item_e = self.forward(user, item)
    #     align = self.alignment(user_e, item_e)  # 1.9909
    #     uniform =  self.gamma * (self.uniformity(user_e) + self.uniformity(item_e))

    #     # print(align,uniform)
    #     return align + uniform
    
        # vicreg_loss = self.vicreg_loss(user_e,item_e)
        # mec_loss = self.MEC_loss(user_e,item_e,self.lamda)
        # return align +Enhanced_uniform
        # return uniform + align + vicreg_loss , vicreg_loss
        # return align + uniform,uniform
    
    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_e = self.user_embedding(user)
        item_e = self.item_embedding(item)
        return torch.mul(user_e, item_e).sum(dim=1)



    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.encoder_name == 'LightGCN':
            if self.restore_user_e is None or self.restore_item_e is None:
                self.restore_user_e, self.restore_item_e = self.encoder.get_all_embeddings()
            user_e = self.restore_user_e[user]
            all_item_e = self.restore_item_e
        else:
            user_e = self.encoder.user_embedding(user)
            all_item_e = self.encoder.item_embedding.weight
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)

    def vicreg_loss(self,x,y):
        x = self.projector(x)
        y = self.projector(y)

        # positive pairs 拉近损失
        repr_loss = F.mse_loss(x, y)    #(tensor(0.2232, device='cuda:0', grad_fn=<MseLossBackward0>))

        # x = torch.cat(FullGatherLayer.apply(x), dim=0)
        # y = torch.cat(FullGatherLayer.apply(y), dim=0)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2        # tensor(0.6638)

        cov_x = (x.T @ x) / (x.shape[0] - 1)
        cov_y = (y.T @ y) / (y.shape[0] - 1)
        cov_loss = self.off_diagonal(cov_x).pow_(2).sum().div(
            self.embedding_size
        ) + self.off_diagonal(cov_y).pow_(2).sum().div(self.embedding_size)     # tensor(0.0547)

        # loss = (
        #     self.std_coeff * std_loss
        #     + self.cov_coeff * cov_loss
        # )
        loss = (
            self.sim_coeff * repr_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )
        return loss   

    def MEC_loss(self, p, z, lamda_inv, order=4):

        c = p.T @ z
        power_matrix =  c
        sum_matrix = torch.zeros_like(power_matrix)

        for k in range(1, order+1):
            if k > 1:
                power_matrix = torch.matmul(power_matrix, c)
            if (k + 1) % 2 == 0:
                sum_matrix += power_matrix / k
            else: 
                sum_matrix -= power_matrix / k

        trace = lamda_inv * torch.trace(sum_matrix) #对角线元素求和

        return trace

    def NEW_uniform(self,x,y):

        xx = x.T@x      # 这是一个对称矩阵
        xx_loss = torch.ones_like(xx)-xx
        # xx_loss=xx_loss*8
        
        yy = y.T@y
        yy_loss = torch.ones_like(yy)-yy
        # yy_loss=yy_loss*8

        # pdb.set_trace()
        # 第一版的loss
        # pdb.set_trace()
        # on_loss = torch.trace(xx_loss)/self.embedding_size +torch.trace(yy_loss)/self.embedding_size
        # off_loss = self.off_diagonal(xx_loss).mul(-2).exp().log() + self.off_diagonal(yy_loss).mul(-2).exp().log()

        # 第二版的loss
        # on_loss = torch.trace(xx_loss.exp()) +torch.trace(yy_loss.exp())
        # off_loss = self.off_diagonal(xx_loss).mul(-2).exp().mean().log()  + self.off_diagonal(yy_loss).mul(-2).exp().mean().log()
        # return  self.BT * on_loss + self.bt * off_loss

        # # 第三版整个矩阵求mean().log()  or  加上batchnorm 
        # MF_loss = xx_loss.exp().mean().log()*self.BT+yy_loss.exp().mean().log()*self.bt
        # pdb.set_trace()
        # return MF_loss


class MFEncoder(nn.Module):
    def __init__(self, user_num, item_num, emb_size):
        super(MFEncoder, self).__init__()
        self.user_embedding = nn.Embedding(user_num, emb_size)
        self.item_embedding = nn.Embedding(item_num, emb_size)

    def forward(self, user_id, item_id):
        u_embed = self.user_embedding(user_id)
        i_embed = self.item_embedding(item_id)
        return u_embed, i_embed

    def get_all_embeddings(self):
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        return user_embeddings, item_embeddings


class LGCNEncoder(nn.Module):
    def __init__(self, user_num, item_num, emb_size, norm_adj, n_layers=3):
        super(LGCNEncoder, self).__init__()
        self.n_users = user_num
        self.n_items = item_num
        self.n_layers = n_layers
        self.norm_adj = norm_adj

        self.user_embedding = torch.nn.Embedding(user_num, emb_size)
        self.item_embedding = torch.nn.Embedding(item_num, emb_size)

    def get_ego_embeddings(self):
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def get_all_embeddings(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings

    def forward(self, user_id, item_id):
        user_all_embeddings, item_all_embeddings = self.get_all_embeddings()
        u_embed = user_all_embeddings[user_id]
        i_embed = item_all_embeddings[item_id]
        return u_embed, i_embed
