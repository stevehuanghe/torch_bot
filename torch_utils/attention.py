import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SelfAttention(nn.Module):
    """
    Module for relational embedding
    """
    def __init__(self, input_dim, output_dim, r=2):
        super(SelfAttention, self).__init__()

        self.W = nn.Linear(input_dim, int(input_dim/r))
        self.U = nn.Linear(input_dim, int(input_dim/r))
        self.H = nn.Linear(input_dim, int(input_dim/r))

        self.fc0 = nn.Linear(int(input_dim/r), output_dim)
        # self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, O0):
        """
        Forward pass for relational embedding
        :param O0: [N, input_dim] object features
        :return: O1: [N, input_dim] encoded features
                 O2: [N, output_dim] decoded features
        """
        R1 = F.softmax(torch.matmul(self.W(O0), torch.t(self.U(O0))), 1)
        O1 = self.fc0(torch.matmul(R1, self.H(O0)))
        return O1


class SelfAttSentEmbed(nn.Module):
    """
    https://github.com/kaushalshetty/Structured-Self-Attention/blob/master/attention/model.py
    A Structured Self-Attentive Sentence Embedding, ICLR'17
    """
    def __init__(self, input_dim, hid_dim, n_heads):
        super(SelfAttSentEmbed, self).__init__()
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.n_heads = n_heads

        self.linear_first = torch.nn.Linear(input_dim, hid_dim)
        self.linear_first.bias.data.fill_(0)
        self.linear_second = torch.nn.Linear(hid_dim, n_heads)
        self.linear_second.bias.data.fill_(0)

    def forward(self, inputs, reduce=True):
        x = F.tanh(self.linear_first(inputs))
        x = self.linear_second(x)
        x = self.softmax(x, 1)
        attention = x.transpose(1, 2)
        embeddings = attention @ inputs
        if not reduce:
            return embeddings
        avg_embeddings = torch.sum(embeddings, 1) / self.r
        return avg_embeddings


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention authored by Yu-Hsiang Huang.
        modified by He Huang
    """

    def __init__(self, temperature=1, attn_dropout=0.0, dim=-1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=dim)

    def forward(self, query, key, value, mask=None, reduce=True):
        """
        :param query: [batch, hid_dim]
        :param key: [batch, N, hid_dim]
        :param value: [batch, N, hid_dim]
        :param mask: [batch, N]
        :param reduce: bool
        :return:
        """
        attn = torch.bmm(key, query.unsqueeze(2)).squeeze(-1)  # [batch, N]
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = value * attn.unsqueeze(2)
        if not reduce:
            return output, attn
        output = output.sum(1)

        return output, attn

