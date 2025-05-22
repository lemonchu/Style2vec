import torch
import torch.nn.functional as F

def compute_loss(style_vecs, group_size):
    """
    计算交叉熵损失。

    :param style_vecs: 向量序列，由模型生成
    :param group_size: 每组的大小
    :return: 每行的交叉熵平均值，作为最终的 Loss
    """
    # 计算相似度矩阵
    similarity_matrix = torch.matmul(style_vecs, style_vecs.T)

    # 对相似度矩阵的第 i 行做 soft max
    softmax_matrix = F.softmax(similarity_matrix, dim=1)

    # 创建目标矩阵
    target_matrix = torch.zeros_like(softmax_matrix)
    for i in range(0, len(style_vecs), group_size):
        target_matrix[i:i+group_size, i:i+group_size] = 1/group_size

    # 计算交叉熵损失
    loss = F.cross_entropy(softmax_matrix, target_matrix)

    return loss

style_vecs = torch.randn(16, 128)
group_size = 4

# 计算损失
loss = compute_loss(style_vecs, group_size)

# 打印损失
print(loss)