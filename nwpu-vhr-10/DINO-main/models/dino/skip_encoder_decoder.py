
# ------------------------------------------------------------------------
# Modified from DINO (https://github.com/IDEA-Research/DINO)
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#  This file includes code from ECA (https://github.com/BangguWu/ECANet)
#  Licensed under the MIT License
# ------------------------------------------------------------------------

import copy
from typing import Optional
import torch
from torch import nn, Tensor
from .utils import _get_activation_fn
from .ops.modules import MSDeformAttn



class eca_layer(nn.Module):
    """Constructs a ECA module.
    (https://github.com/BangguWu/ECANet)
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)
    
class SkipEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,
                 add_channel_attention=False,
                 use_deformable_box_attn=False,
                 box_attn_type='roi_align',
                 ):
        super().__init__()
        # self attention
       
        self.skip_fc1 = nn.Linear(d_model,2*d_model)
        
        block_ = nn.Sequential(
            nn.Conv2d(2*d_model,2*d_model,kernel_size=3, padding=1,groups = 2*d_model), #depth
            nn.Conv2d(2*d_model, 2*d_model,kernel_size=1), # point

        )
                   
        self.depthwise_list = nn.ModuleList([copy.deepcopy(block_) for i in range(4)])
        # self.depthwise_list = nn.ModuleList([copy.deepcopy(nn.Conv2d(2*d_model, 2*d_model, kernel_size=3, padding=1, groups=2*d_model)) for i in range(4)]) # num_level groups 1*d to 2*d
        self.skip_fc2 = nn.Linear(2*d_model,d_model)
        self.skip_eca = eca_layer(9) #3 to 9
        self.gelu = nn.GELU()

        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        # channel attention
        self.add_channel_attention = add_channel_attention
        if add_channel_attention:
            self.activ_channel = _get_activation_fn('dyrelu', d_model=d_model)
            self.norm_channel = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos
    

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, key_padding_mask=None, skip_input=None,skip_layer_id=0):
        # skip-encoder
        # spatial_shapes
            # - src: [bs, sum(hi*wi), 256]
            # - pos: pos embed for src. [bs, sum(hi*wi), 256]
            # - spatial_shapes: h,w of each level [num_level, 2]
            # - level_start_index: [num_level] start point of level in sum(hi*wi).
            # - valid_ratios: [bs, num_level, 2] 0-1

        bs,_,d = src.shape

       
        
        src2 = self.skip_fc1(skip_input)   #bs,n,2d
        src2 = self.gelu(src2)      #gelu+no PE
       
        level,_ = spatial_shapes.shape
        src3 = []
        for i in range(level):
            h,w = spatial_shapes[i]
            index0 = level_start_index[i]
            if i!= 3:
              index1 = level_start_index[i+1]
  
              src_temp = src2[:,index0:index1,:].view(bs,h,w,2*d).permute(0,3,1,2) # bs,hi*wi,2d -> bs,hi,wi,2d
            else:
              src_temp = src2[:,index0:,:].view(bs,h,w,2*d).permute(0,3,1,2)
            src_temp = self.depthwise_list[i](src_temp).permute(0,2,3,1).view(bs,-1,2*d) #bs,hi,wi,2d ->bs,hi*wi,2d
            # src_temp = self.skip_fc2(src_temp).view(bs,h,w,-1)                  #bs,hi*wi,2d ->bs,hi,wi,d (conv,possiblely cast fc!)
            # src_temp = self.skip_eca(src_temp)
            src3.append(src_temp)
        src2 = torch.cat(src3, 1)

        src2 = self.skip_fc2(src2) #bs,n,d
        src2 = self.skip_eca(src2)
        # if skip_layer_id != 1: 
        src2 = skip_input + self.dropout1(src2) # some change
        
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # ffn
        src = self.forward_ffn(src)
        
        # channel attn
        if self.add_channel_attention:
            src = self.norm_channel(src + self.activ_channel(src))

        return src
    
class SkipDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,
                 use_deformable_box_attn=False,
                 box_attn_type='roi_align',
                 key_aware_type=None,
                 decoder_sa_type='ca',
                 module_seq=['sa', 'ca', 'ffn'],
                 
                 ):
        super().__init__()
        self.module_seq = module_seq
        assert sorted(module_seq) == ['ca', 'ffn', 'sa']
        # cross attention
        if use_deformable_box_attn:
            self.cross_attn = MSDeformableBoxAttention(d_model, n_levels, n_heads, n_boxes=n_points, used_func=box_attn_type)
        else:
            self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
          
      

        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn_noise = nn.MultiheadAttention(d_model, n_heads, dropout=dropout) # need modify
        self.self_attn = True
        # skip-attention
        self.skip_fc1 = nn.Linear(d_model,2*d_model)

        self.block_ = nn.Sequential(
            nn.Conv2d(2*d_model,2*d_model,kernel_size=3, padding=1,groups = 2*d_model), #depth
            nn.BatchNorm2d(2*d_model),
            nn.GELU(),
       
            nn.Conv2d(2*d_model, 2*d_model,kernel_size=1), # point
            nn.BatchNorm2d(2*d_model),
            nn.GELU(),

            # nn.Conv2d(2*d_model,2*d_model,kernel_size=3, padding=1,groups = d_model), #depth
            # nn.Conv2d(2*d_model, 2*d_model,kernel_size=1), # point
            # nn.BatchNorm(2*d_model),
            # add norm

            nn.Conv2d(2*d_model,2*d_model,kernel_size=3, padding=1,groups = 2*d_model),
            nn.BatchNorm2d(2*d_model),
            nn.GELU(),
        )
        self.skip_fc2 = nn.Linear(2*d_model,d_model)
        self.skip_eca = eca_layer(9) #3 to 9 to 5 to 3
        self.gelu = nn.GELU()

        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn, batch_dim=1)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.key_aware_type = key_aware_type
        self.key_aware_proj = None
        self.decoder_sa_type = decoder_sa_type
        assert decoder_sa_type in ['sa', 'ca_label', 'ca_content']

        if decoder_sa_type == 'ca_content':
            self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)

    def rm_self_attn_modules(self):
        self.self_attn = None
        self.dropout2 = None
        self.norm2 = None

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_sa(self,
                # for tgt
                tgt: Optional[Tensor],  # nq, bs, d_model
                tgt_query_pos: Optional[Tensor] = None, # pos for query. MLP(Sine(pos))
                tgt_query_sine_embed: Optional[Tensor] = None, # pos for query. Sine(pos)
                tgt_key_padding_mask: Optional[Tensor] = None,
                tgt_reference_points: Optional[Tensor] = None, # nq, bs, 4

                # for memory
                memory: Optional[Tensor] = None, # hw, bs, d_model
                memory_key_padding_mask: Optional[Tensor] = None,
                memory_level_start_index: Optional[Tensor] = None, # num_levels
                memory_spatial_shapes: Optional[Tensor] = None, # bs, num_levels, 2
                memory_pos: Optional[Tensor] = None, # pos for memory

                # sa
                self_attn_mask: Optional[Tensor] = None, # mask used for self-attention
                cross_attn_mask: Optional[Tensor] = None, # mask used for cross-attention
                origin_src = None,
                dn_numbers = None,
            ):
        # self attention
        if self.self_attn is not None:
            if self.decoder_sa_type == 'sa':
                groups_temp = dn_numbers['num_dn_group']
                dn_numbers = dn_numbers['zero_positions']
                if dn_numbers!=0:
                    
                    # tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)[0]
                    tgt_pos = self.with_pos_embed(tgt, tgt_query_pos)
                    # #skip
                    
                    noise_pos = tgt_pos[:dn_numbers] #
                    noise_pos = self.self_attn_noise(query=noise_pos,key=tgt_pos,value=tgt_pos,attn_mask=self_attn_mask[:dn_numbers])[0]

                   
                else:
                    noise_pos = None


                if origin_src != None:
                    tgt_temp = tgt
                    tgt = origin_src
                    #   print()
                no_noise_tgt = tgt[dn_numbers:]
                    # no_noise_tgt_pos = tgt_query_pos[200:]
                    # noise_tgt = tgt[:200]
                    # gelu+noise+no PE+eca kernel = 3
                tgt2 = self.with_pos_embed(no_noise_tgt,tgt_query_pos[dn_numbers:]) # new test

                tgt2 = self.skip_fc1(no_noise_tgt).permute(1,2,0).unsqueeze(3) # nq,bs,d_model*2 -->bs,2d,nq,1
                tgt2 = self.gelu(tgt2)
  
                tgt2 = self.block_(tgt2).squeeze(3).permute(0,2,1) # add residual



                tgt2 = self.skip_eca(self.skip_fc2(tgt2)).permute(1,0,2) #nq,bs,d_model
                tgt = no_noise_tgt + self.dropout2(tgt2)
                if noise_pos!=None:
                    tgt = torch.cat((noise_pos,tgt))
                tgt = self.norm2(tgt)

                tgt = tgt + tgt_temp
              
                

            elif self.decoder_sa_type == 'ca_label':
                bs = tgt.shape[1]
                k = v = self.label_embedding.weight[:, None, :].repeat(1, bs, 1)
                tgt2 = self.self_attn(tgt, k, v, attn_mask=self_attn_mask)[0]
                tgt = tgt + self.dropout2(tgt2)
                tgt = self.norm2(tgt)
            elif self.decoder_sa_type == 'ca_content':
                tgt2 = self.self_attn(self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
                            tgt_reference_points.transpose(0, 1).contiguous(),
                            memory.transpose(0, 1), memory_spatial_shapes, memory_level_start_index, memory_key_padding_mask).transpose(0, 1)
                tgt = tgt + self.dropout2(tgt2)
                tgt = self.norm2(tgt)
            else:
                raise NotImplementedError("Unknown decoder_sa_type {}".format(self.decoder_sa_type))

        return tgt

    def forward_ca(self,
                # for tgt
                tgt: Optional[Tensor],  # nq, bs, d_model
                tgt_query_pos: Optional[Tensor] = None, # pos for query. MLP(Sine(pos))
                tgt_query_sine_embed: Optional[Tensor] = None, # pos for query. Sine(pos)
                tgt_key_padding_mask: Optional[Tensor] = None,
                tgt_reference_points: Optional[Tensor] = None, # nq, bs, 4

                # for memory
                memory: Optional[Tensor] = None, # hw, bs, d_model
                memory_key_padding_mask: Optional[Tensor] = None,
                memory_level_start_index: Optional[Tensor] = None, # num_levels
                memory_spatial_shapes: Optional[Tensor] = None, # bs, num_levels, 2
                memory_pos: Optional[Tensor] = None, # pos for memory

                # sa
                self_attn_mask: Optional[Tensor] = None, # mask used for self-attention
                cross_attn_mask: Optional[Tensor] = None, # mask used for cross-attention
            ):
        # cross attention
        if self.key_aware_type is not None: # default None

            if self.key_aware_type == 'mean':
                tgt = tgt + memory.mean(0, keepdim=True)
            elif self.key_aware_type == 'proj_mean':
                tgt = tgt + self.key_aware_proj(memory).mean(0, keepdim=True)
            else:
                raise NotImplementedError("Unknown key_aware_type: {}".format(self.key_aware_type))
            
        
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
                               tgt_reference_points.transpose(0, 1).contiguous(),
                               memory.transpose(0, 1), memory_spatial_shapes, memory_level_start_index, memory_key_padding_mask).transpose(0, 1)
        # skip_multiscale?


        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        return tgt

    def forward(self,
                # for tgt
                tgt: Optional[Tensor],  # nq, bs, d_model
                tgt_query_pos: Optional[Tensor] = None, # pos for query. MLP(Sine(pos))
                tgt_query_sine_embed: Optional[Tensor] = None, # pos for query. Sine(pos)
                tgt_key_padding_mask: Optional[Tensor] = None,
                tgt_reference_points: Optional[Tensor] = None, # nq, bs, 4

                # for memory
                memory: Optional[Tensor] = None, # hw, bs, d_model
                memory_key_padding_mask: Optional[Tensor] = None,
                memory_level_start_index: Optional[Tensor] = None, # num_levels
                memory_spatial_shapes: Optional[Tensor] = None, # bs, num_levels, 2
                memory_pos: Optional[Tensor] = None, # pos for memory

                # sa
                self_attn_mask: Optional[Tensor] = None, # mask used for self-attention
                cross_attn_mask: Optional[Tensor] = None, # mask used for cross-attention
                origin_src = None,
                dn_numbers = None,
            ):

        for funcname in self.module_seq:
            if funcname == 'ffn':
                tgt = self.forward_ffn(tgt)
            elif funcname == 'ca':
                tgt = self.forward_ca(tgt, tgt_query_pos, tgt_query_sine_embed, \
                    tgt_key_padding_mask, tgt_reference_points, \
                        memory, memory_key_padding_mask, memory_level_start_index, \
                            memory_spatial_shapes, memory_pos, self_attn_mask, cross_attn_mask)
            elif funcname == 'sa':
                tgt = self.forward_sa(tgt, tgt_query_pos, tgt_query_sine_embed, \
                    tgt_key_padding_mask, tgt_reference_points, \
                        memory, memory_key_padding_mask, memory_level_start_index, \
                            memory_spatial_shapes, memory_pos, self_attn_mask, cross_attn_mask,origin_src,dn_numbers=dn_numbers)
            else:
                raise ValueError('unknown funcname {}'.format(funcname))

        return tgt
    
