#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder self-attention layer definition."""

import torch

from torch import nn

class LocalMaxSA(nn.Module):

    def __init__(self,n_feat):
        super().__init__()
        self.n_feat = n_feat
        self.q = nn.Linear(n_feat,n_feat)
        self.k = nn.Linear(n_feat,n_feat)

    def forward(self,feat,hlens):
        n_batchs = feat.size(dim=0)
        n_frames = feat.size(dim=1)
        q = self.q(feat).view(n_batchs,n_frames,self.n_feat)
        k = self.k(feat).view(n_batchs,n_frames,self.n_feat)
        

        sa = torch.matmul(q,k.transpose(-2,-1))
        score = torch.mean(sa,-1)
        sa = torch.softmax(sa,dim=-1) 
        feat = torch.matmul(sa,feat)

        score_left = torch.cat((score[:,0].view(-1,1),score[:,:-1]),1)
        score_right = torch.cat((score[:,1:],score[:,-1].view(-1,1)),1)
        peaks = torch.logical_and(torch.ge(score,score_left),torch.ge(score,score_right))
        peaks = torch.ones([n_batchs,n_frames],dtype=torch.bool)
        hlens_new = torch.sum(peaks,dim=1).int()
        feat_new = torch.zeros(n_batchs,max(hlens_new),self.n_feat,device=feat.device)
        for indx in range(n_batchs):
            feat_new[indx,:hlens_new[indx],:] = feat[indx,peaks[indx],:]
        return feat_new, hlens_new,score

class CIF(nn.Module):

    def __init__(self,n_feat):
        super().__init__()
        self.n_feat = n_feat
        self.Lin = nn.Linear(n_feat,1)
        self.Sig = nn.Sigmoid()

    def forward(self,feat,hlens):
        n_batchs = feat.size(dim=0)
        n_frames = feat.size(dim=1)
        weight = self.Lin(feat)
        weight = self.Sig(weight).view(n_batchs,n_frames)
        
        hlens_new = torch.ceil(torch.sum(weight,dim=1)).int()
        weight_cmat = torch.zeros(n_batchs,n_frames,max(hlens_new),device=feat.device)

        for indx in range(n_batchs):
            fra_new = 0
            weight_acc = 0 
            for fra in range(n_frames):
                if weight_acc+weight[indx,fra] <= 1:
                    weight_cmat[indx,fra,fra_new] = 1
                    weight_acc = weight_acc+weight[indx,fra]
                else:
                    weight_cmat[indx,fra,fra_new] = (1-weight_acc)/weight[indx,fra]
                    fra_new += 1
                    weight_cmat[indx,fra,fra_new] = (weight_acc+weight[indx,fra]-1)/weight[indx,fra]
                    weight_acc = 0 
      
        weight_cmat = weight_cmat.detach()
        weight_mat = weight.unsqueeze(2).repeat(1,1,max(hlens_new))
        weight_mat = weight_mat*weight_cmat
        feat_new = torch.matmul(feat.transpose(-2,-1),weight_mat).transpose(-2,-1)
        return feat_new, hlens_new

class SigMax1(nn.Module):

    def __init__(self,n_feat):
        super().__init__()
        self.n_feat = n_feat
        self.Lin = nn.Linear(n_feat,1)
        self.Sig = nn.Sigmoid()

    def forward(self,feat):
        n_batchs = feat.size(dim=0)
        n_frames = feat.size(dim=1)
        score = self.Lin(feat)
        score = self.Sig(score).view(n_batchs,n_frames)
        
        score_left = torch.cat((score[:,0].view(-1,1),score[:,:-1]),1)
        score_right = torch.cat((score[:,1:],score[:,-1].view(-1,1)),1)
        peaks = torch.logical_and(torch.ge(score,score_left),torch.ge(score,score_right))
        hlens = torch.sum(peaks,dim=1).int()
        feat_new = torch.zeros(n_batchs,max(hlens),self.n_feat,device=feat.device)
        
        for indx in range(n_batchs):
            feat_new[indx,:hlens[indx],:] = feat[indx,peaks[indx],:]

        return feat_new, hlens

class RnnMax(nn.Module):

    def __init__(self,n_feat):
        super().__init__()
        self.n_feat = n_feat
        self.RNN = nn.LSTM(n_feat,int(n_feat/2),num_layers=1,bidirectional=True,batch_first=True)
        self.Lin = nn.Linear(n_feat,1)
        self.Sig = nn.Sigmoid()

    def forward(self,feat):
        n_batchs = feat.size(dim=0)
        n_frames = feat.size(dim=1)
        score,states = self.RNN(feat)
        score = self.Lin(score)
        score = self.Sig(score).view(n_batchs,n_frames)
        
        score_left = torch.cat((score[:,0].view(-1,1),score[:,:-1]),1)
        score_right = torch.cat((score[:,1:],score[:,-1].view(-1,1)),1)
        peaks = torch.logical_and(torch.ge(score,score_left),torch.ge(score,score_right))
        hlens = torch.sum(peaks,dim=1).int()
        feat_new = torch.zeros(n_batchs,max(hlens),self.n_feat,device=feat.device)
        
        for indx in range(n_batchs):
            feat_new[indx,:hlens[indx],:] = feat[indx,peaks[indx],:]

        return feat_new, hlens


class SigMax(nn.Module):

    def __init__(self,n_feat):
        super().__init__()
        self.n_feat = n_feat
        self.Lin = nn.Linear(n_feat,1)
        self.Sig = nn.Sigmoid()

    def forward(self,feat,hlens):
        n_batchs = feat.size(dim=0)
        n_frames = feat.size(dim=1)
        score = self.Lin(feat)
        score = self.Sig(score).view(n_batchs,n_frames)
         
        score_left = torch.cat((score[:,0].view(-1,1),score[:,:-1]),1)
        score_right = torch.cat((score[:,1:],score[:,-1].view(-1,1)),1)
        valleys = torch.logical_and(torch.ge(score,score_left),torch.ge(score,score_right))
      #  valleys = torch.logical_and(torch.ge(score_left,score),torch.ge(score_right,score))
        hlens_new = torch.zeros(n_batchs,dtype=torch.int)
        for indx in range(n_batchs):
            valleys[indx,0] = True
            if n_batchs == 1:
                valleys[indx,hlens-1] = True
                hlens_new[indx] = (torch.sum(valleys[indx,:hlens])).int()
            else:
                valleys[indx,hlens[indx]-1] = True
                hlens_new[indx] = (torch.sum(valleys[indx,:hlens[indx]])).int()
        max_len = torch.max(hlens_new)
       #
        weight_cmat = torch.zeros(n_batchs,n_frames,max_len,device=feat.device)
        for indx in range(n_batchs):
            if n_batchs == 1:
                valley_indx = torch.squeeze(torch.nonzero(valleys[indx,:hlens]))
                vstart = torch.cat((torch.Tensor([0]).to(feat.device),valley_indx[:-1])).int()
                vend = torch.cat((valley_indx[1:],torch.Tensor([hlens-1]).to(feat.device))).int()
            else:
                valley_indx = torch.squeeze(torch.nonzero(valleys[indx,:hlens[indx]]))
                vstart = torch.cat((torch.Tensor([0]).to(feat.device),valley_indx[:-1])).int()
                vend = torch.cat((valley_indx[1:],torch.Tensor([hlens[indx]-1]).to(feat.device))).int()
            for fra_new in range(hlens_new[indx]):                
                weight_cmat[indx,vstart[fra_new]:vend[fra_new],fra_new] = torch.ones(vend[fra_new]-vstart[fra_new]).to(feat.device)/(torch.sum(score[indx,vstart[fra_new]:vend[fra_new]])+1e-10)
       
        weight_cmat = weight_cmat.detach()        
        weight_mat = score.unsqueeze(2).repeat(1,1,max_len)
        weight_mat = weight_mat*weight_cmat
        feat_new = torch.matmul(feat.transpose(-2,-1),weight_mat).transpose(-2,-1)
        return feat_new, hlens_new, score
        
class SigModeOv(nn.Module):

    def __init__(self,n_feat):
        super().__init__()
        self.n_feat = n_feat
        # self.n_heads = 1 
        # self.Lin = nn.Linear(n_feat,self.n_heads)
        # self.Sig = nn.Sigmoid()

        self.linear_sigmoid = torch.nn.Sequential(
            torch.nn.Linear(self.n_feat, 1),
            torch.nn.Sigmoid(),
        )


    def forward(self,input,input_lengths):
        batch, length, dim = input.shape
        
        # scalar importance shape: [B, T, 1]
        scalar_importance = self.linear_sigmoid(input)
        # scalar_importance = torch.mean(scalar_importance,-1)
        # print("scalar: ", scalar_importance.mT)
        
        alpha_h = torch.mul(scalar_importance, input)
        # print("alpha_h: ", alpha_h[1,:,0])

        # find valleys' index
        scalar_before = scalar_importance[:,:-1,:].detach()
        scalar_after = scalar_importance[:,1:,:].detach()
        scalar_before = torch.nn.functional.pad(scalar_before,(0,0,1,0))
        scalar_after = torch.nn.functional.pad(scalar_after,(0,0,0,1))

        mask = ((scalar_importance.lt(scalar_before)) & (scalar_importance.lt(scalar_after))).detach()
        mask = mask.reshape(scalar_importance.shape[0],-1)
        # print("mask: ", mask.shape)
        # print("mask:", mask[0].T)
        mask[:,0] = True
        batch_index = mask.nonzero()[:,0]
        valley_index_start = mask.nonzero()[:,1]
        mask[:,0] = False
        mask[:,-1] = True
        valley_index_end = mask.nonzero()[:,1] + 2
        valley_index_end = torch.where(valley_index_end > length * torch.ones_like(valley_index_end), 
                                       length * torch.ones_like(valley_index_end), valley_index_end)

        
        _,counts = torch.unique(batch_index, return_counts = True)
        max_counts = (torch.max(counts)).item()
        print("")
        print("max_counts: ", max_counts)
        print("counts: ", counts)

        utri_mat1 = torch.tril(torch.ones(max_counts+1,max_counts),-1).to(input.device)
        batch_index_mask = utri_mat1[counts]
        batch_index_mask = batch_index_mask.reshape(-1,1)
        batch_index_mask = batch_index_mask.nonzero()[:, 0]
        # print("batch_index: ", batch_index)
        # print("batch_index_mask: ", batch_index_mask)
        
        # batch_index = batch_index + torch.arange(batch_index.shape[0]).to(batch_index.device)
        
        valleys = torch.zeros(batch * max_counts, 2).type_as(valley_index_start)
        valleys[batch_index_mask] = torch.cat((valley_index_start.unsqueeze(1), valley_index_end.unsqueeze(1)),1)
        print(valleys)
        print("valleys: ", valleys.shape)
        
        # utri_mat = torch.tril(torch.cuda.FloatTensor(length+1,length).fill_(1),-1)
        utri_mat = torch.tril(torch.ones(length+1,length),-1).to(input.device)
        print(utri_mat.shape)
        print("batch, max_counts, length: ", batch, max_counts, length)
        output_mask = (utri_mat[valleys[:,1]]-utri_mat[valleys[:,0]]).reshape(batch, max_counts, length)
        output_mask = output_mask.detach()
        print(output_mask.shape)

        output = torch.bmm(output_mask, alpha_h) / torch.bmm(output_mask, scalar_importance).clamp_(1e-6)
        # print(torch.isnan(output).any())
        output_lengths = (input_lengths / input_lengths[0] * output.shape[1]).type_as(input_lengths)

        return output, output_lengths,  scalar_importance
    
        n_batchs = feat.size(dim=0)
        n_frames = feat.size(dim=1)
        n_feat = feat.size(dim=2)
     
        score_mh = self.Lin(feat)
        score_mh = self.Sig(score_mh).view(n_batchs,n_frames,self.n_heads)
        score = torch.mean(score_mh,-1) 
         
        score_left = torch.cat((score[:,0].view(-1,1),score[:,:-1]),1)
        score_right = torch.cat((score[:,1:],score[:,-1].view(-1,1)),1)
        

      #  valleys = torch.logical_and(torch.ge(score,score_left),torch.ge(score,score_right))
        valleys = torch.logical_and(torch.ge(score_left,score),torch.ge(score_right,score))
        hlens_new = torch.zeros(n_batchs,dtype=torch.int)
        for indx in range(n_batchs):
            valleys[indx,0] = True
            if n_batchs == 1:
                valleys[indx,hlens-1] = True
                hlens_new[indx] = (torch.sum(valleys[indx,:hlens])-1).int()
            else:
                valleys[indx,hlens[indx]-1] = True
                hlens_new[indx] = (torch.sum(valleys[indx,:hlens[indx]])-1).int()
        max_len = torch.max(hlens_new)
       #
        weight_cmat = torch.zeros(n_batchs,n_frames,max_len,device=feat.device)
        for indx in range(n_batchs):
            if n_batchs == 1:
                valley_indx = torch.squeeze(torch.nonzero(valleys[indx,:hlens]))
            else:
                valley_indx = torch.squeeze(torch.nonzero(valleys[indx,:hlens[indx]]))
           # vstart = torch.max(valley_indx[0],valley_indx[:-1]-1).int()
           # vend = torch.min(valley_indx[-1],valley_indx[1:]+2).int()
            vstart = valley_indx[:-1].int()
            vend = torch.min(valley_indx[-1],valley_indx[1:]+2).int()
           # print("vstart {},vend {}".format(vstart,vend))
            for fra_new in range(hlens_new[indx]):                
                weight_cmat[indx,vstart[fra_new]:vend[fra_new],fra_new] = torch.ones(vend[fra_new]-vstart[fra_new]).to(feat.device)/(torch.sum(score[indx,vstart[fra_new]:vend[fra_new]])+1e-10)
       
        weight_cmat = weight_cmat.detach()        
        weight_mat = score.unsqueeze(2).repeat(1,1,max_len)
        weight_mat = weight_mat*weight_cmat
        feat_new = torch.matmul(feat.transpose(-2,-1),weight_mat).transpose(-2,-1)
        return feat_new, hlens_new, score_mh


