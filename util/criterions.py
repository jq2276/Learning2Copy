#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions
from torch.nn.modules.loss import _Loss


class NLLLoss(_Loss):
    def __init__(self, weight=None, ignore_index=-100, reduction='mean'):
        super(NLLLoss, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        self.register_buffer('weight', weight)
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target, reduction=True):
        batch_size = input.size(0)
        nll = F.nll_loss(input=input.view(-1, input.size(-1)),
                         target=target.contiguous().view(-1),
                         weight=self.weight,
                         reduction='none')
        nll = nll.view(batch_size, -1).sum(dim=1)
        if reduction:
            if self.reduction == 'mean':
                nll = nll.mean()
            elif self.reduction == 'sum':
                nll = nll.sum()

        return nll


class CopyGeneratorLoss(nn.Module):
    def __init__(self, vocab_size, force_copy, unk_index=0,
                 ignore_index=-100, eps=1e-20):

        super(CopyGeneratorLoss, self).__init__()
        self.force_copy = force_copy
        self.eps = eps
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.unk_index = unk_index

    def forward(self, scores, align, target):

        vocab_probs = scores.gather(1, target.unsqueeze(1)).squeeze(1)  # [b, tgt_len]

        # probability of tokens copied from source
        copy_ix = align.unsqueeze(1)        # + self.vocab_size   # [batch_size, 1, tgt_len]
        copy_tok_probs = scores.gather(1, copy_ix).squeeze(1)   # [b, tgt_len]

        # Set scores for unk to 0 and add eps
        copy_tok_probs[align == self.unk_index] = 0
        copy_tok_probs += self.eps  # to avoid -inf logs

        # find the indices in which you do not use the copy mechanism
        non_copy = align == self.unk_index
        if not self.force_copy:
            non_copy = non_copy | (target != self.unk_index)

        probs = torch.where(
            non_copy, copy_tok_probs + vocab_probs, copy_tok_probs
        )

        loss = -probs.log()  # probs_size = [b, tgt_len]
        # Drop padding.

        loss[target == self.ignore_index] = 0   # [b, tgt_len]
        return loss
