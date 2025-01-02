import torch
import torch.nn as nn
import numpy as np


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)    # 어느 축을 기준으로 softmax할건지 반드시 정해야 함
        # dim=2인 이유는 그래야 결과값이 new word1 = a*word1 + b*word2 + c*word3 + ... 꼴로 나옴
        # 뭔말인지 모르면 계산해봐

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        output = torch.bmm(attn, v)

        return output, attn
