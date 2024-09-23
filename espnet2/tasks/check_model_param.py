'''
Author: FnoY fangying@westlake.edu.cn
LastEditors: FnoY0723 fangying@westlake.edu.cn
LastEditTime: 2024-07-12 18:21:11
FilePath: /espnet/espnet2/tasks/check_model_param.py
'''
import torch
import torch.nn.functional as F

model_path1 = '/data/home/fangying/espnet/egs2/aishell/asr1/exp_uma_mamba_0506/kernel21_right0_asr_train_asr_uma_mamba_raw_zh_char_sp/valid.cer.ave_10best.pth'
model_path2 = '/data/home/fangying/espnet/egs2/aishell/asr1/exp_uma_mamba_0611/kernel21_right10_asr_train_asr_uma_mamba_raw_zh_char_sp/valid.cer.ave_10best.pth'
# 加载已保存的模型状态字典
state_dict1 = torch.load(model_path1)
state_dict2 = torch.load(model_path2)

# 打印 state_dict 中的所有键以确认模块名称
# print("State dict keys:", state_dict.keys())
uma_linear_weights1 = state_dict1['uma.linear_sigmoid.0.weight']
uma_linear_bias1 = state_dict1['uma.linear_sigmoid.0.bias']
uma_linear_weights2 = state_dict2['uma.linear_sigmoid.0.weight']
uma_linear_bias2 = state_dict2['uma.linear_sigmoid.0.bias']

cos_sim = F.cosine_similarity(uma_linear_weights1, uma_linear_weights2)
print(f"cosine similarity between two models: {cos_sim}")
# print("uma_linear_weights shape:", uma_linear_weights.shape)
# print(f"{uma_linear_weights}")
# print
# print("uma_linear_bias shape:", uma_linear_bias.shape)
# print(f"{uma_linear_bias}")