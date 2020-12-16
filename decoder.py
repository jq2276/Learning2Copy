#!/usr/bin/env python

import torch
import torch.nn as nn

from util.attention import Attention
from util.state import DecoderState
from util.misc import sequence_mask


class Decoder(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 embedder=None,
                 num_layers=1,
                 attn_mode=None,
                 memory_size=None,
                 dropout=0.0,
                 device=None):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedder = embedder
        self.num_layers = num_layers
        self.attn_mode = None if attn_mode == 'none' else attn_mode
        self.memory_size = memory_size or hidden_size
        self.dropout = dropout
        self.device = device

        self.rnn_input_size = self.input_size
        self.cue_input_size = self.hidden_size * 2
        self.goal_input_size = self.hidden_size * 2
        self.out_input_size = self.hidden_size

        if self.attn_mode is not None:

            self.utt_attention = Attention(query_size=self.hidden_size,
                                           memory_size=self.memory_size,
                                           mode=self.attn_mode,
                                           project=False,
                                           device=self.device)

            self.cue_attention = Attention(query_size=self.hidden_size,
                                           memory_size=self.memory_size,
                                           mode=self.attn_mode,
                                           project=False,
                                           device=self.device)

            self.goal_attention = Attention(query_size=self.hidden_size,
                                            memory_size=self.memory_size,
                                            mode=self.attn_mode,
                                            project=False,
                                            device=self.device)

            self.high_level_attention = Attention(query_size=self.hidden_size,
                                                  memory_size=self.memory_size,
                                                  mode=self.attn_mode,
                                                  project=False,
                                                  device=self.device)

            self.rnn_input_size += self.memory_size
            self.cue_input_size += self.memory_size
            self.goal_input_size += self.memory_size
            self.out_input_size += self.memory_size

        self.dec_rnn = nn.GRU(input_size=self.rnn_input_size,
                              hidden_size=self.hidden_size,
                              num_layers=self.num_layers,
                              dropout=self.dropout if self.num_layers > 1 else 0,
                              batch_first=True)

        self.p_gen_linear = nn.Linear(self.hidden_size * 2 + self.input_size, 1)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        if self.out_input_size > self.hidden_size:
            self.output_layer = nn.Sequential(
                nn.Dropout(p=self.dropout),
                nn.Linear(self.out_input_size, self.hidden_size),
                nn.Linear(self.hidden_size, self.output_size),
                nn.Softmax(dim=-1),
            )
        else:
            self.output_layer = nn.Sequential(
                nn.Dropout(p=self.dropout),
                nn.Linear(self.out_input_size, self.output_size),
                nn.Softmax(dim=-1),
            )

    def initialize_state(self,
                         utt_hidden,
                         utt_outputs,
                         utt_input_len,
                         cue_hidden,
                         cue_outputs,
                         cue_input_len,
                         goal_hidden,
                         goal_outputs,
                         goal_input_len,
                         pr_attn_dist,
                         po_attn_dist,
                         dec_init_context,
                         feature=None,
                         utt_attn_mask=None,
                         cue_attn_mask=None,
                         goal_attn_mask=None):

        if self.attn_mode is not None:
            assert utt_outputs is not None
            assert cue_outputs is not None

        if utt_input_len is not None and utt_attn_mask is None:
            utt_max_len = utt_outputs.size(1)
            utt_attn_mask = sequence_mask(utt_input_len, utt_max_len).eq(0)

        if cue_input_len is not None and cue_attn_mask is None:
            cue_max_len = cue_outputs.size(2)
            cue_attn_mask = sequence_mask(cue_input_len, cue_max_len).eq(0)

        if goal_input_len is not None and goal_attn_mask is None:
            goal_max_len = goal_outputs.size(1)
            goal_attn_mask = sequence_mask(goal_input_len, goal_max_len).eq(0)

        init_state = DecoderState(
            utt_hidden=utt_hidden,
            utt_outputs=utt_outputs,
            utt_input_len=utt_input_len,
            utt_attn_mask=utt_attn_mask,
            cue_hidden=cue_hidden,
            cue_outputs=cue_outputs,
            cue_input_len=cue_input_len,
            cue_attn_mask=cue_attn_mask,
            goal_hidden=goal_hidden,
            goal_outputs=goal_outputs,
            goal_input_len=goal_input_len,
            goal_attn_mask=goal_attn_mask,
            pr_attn_dist=pr_attn_dist,
            po_attn_dist=po_attn_dist,
            feature=feature,
            dec_init_context=dec_init_context
        )
        return init_state

    def decode(self, input, state, oovs_max,
               valid_src_extend_vocab, valid_cue_extend_vocab, valid_goal_extend_vocab,
               is_training=False):

        # Here we use the last hidden state of utt encoder to initialize the decoder
        dec_hidden = state.utt_hidden
        dec_previous_context = state.dec_init_context

        # Not need to be updated
        cue_enc_hidden = state.cue_hidden  # [sent_num, batch_size, hid_size]
        utt_enc_outputs = state.utt_outputs
        cue_enc_outputs = state.cue_outputs  # [batch_size, cue_sent_num, cue_max_len, hid_size]
        goal_enc_outputs = state.goal_outputs  # [batch_size, goal_max_len, hid_size]

        if state.po_attn_dist is not None:
            kg_attn_dist = state.po_attn_dist
        else:
            kg_attn_dist = state.pr_attn_dist

        utt_attn_mask = state.utt_attn_mask
        goal_attn_mask = state.goal_attn_mask
        cue_attn_mask = state.cue_attn_mask

        cur_batch_size = cue_enc_hidden.size(1)
        cue_sent_num = cue_enc_hidden.size(0)
        cue_max_len = cue_enc_outputs.size(-2)

        # LIST to store the input
        dec_input_list = []    # utt decoder
        out_input_list = []    # output needed input
        p_gen_input_list = []    # p_gen input

        if self.embedder is not None:
            input = self.embedder(input)

        # shape: (batch_size, 1, input_size)
        input = input.unsqueeze(1)
        dec_input_list.append(input)
        dec_input_list.append(dec_previous_context)

        p_gen_input_list.append(input)

        # dec
        dec_input = torch.cat(dec_input_list, dim=-1)
        _, dec_hidden_ori = self.dec_rnn(dec_input, dec_hidden)
        dec_hidden = dec_hidden_ori[-1].unsqueeze(0).clone()

        out_input_list.append(dec_hidden.transpose(0, 1))
        p_gen_input_list.append(dec_hidden.transpose(0, 1))

        # UTT ATTN
        dec_query = dec_hidden[-1].unsqueeze(1)
        utt_weighted_context, utt_attn_dist = self.utt_attention(query=dec_query,
                                                                 memory=utt_enc_outputs,
                                                                 mask=utt_attn_mask)

        # CUE ATTN
        cue_query = dec_query.repeat(1, cue_sent_num, 1).view(-1, self.hidden_size).unsqueeze(1)
        cue_attn_memory = cue_enc_outputs.view(-1, cue_max_len, self.hidden_size)
        cue_attn_mask = cue_attn_mask.view(-1, cue_max_len)

        cue_weighted_context, cue_attn = self.cue_attention(query=cue_query,
                                                            memory=cue_attn_memory,
                                                            mask=cue_attn_mask)

        cue_weighted_context = cue_weighted_context.view(cur_batch_size, cue_sent_num, -1)
        cue_weighted_context = kg_attn_dist.unsqueeze(2).mul(cue_weighted_context)
        cue_weighted_context = cue_weighted_context.sum(dim=1)

        # GOAL ATTN
        goal_weighted_context, goal_attn_dist = self.goal_attention(query=dec_query,
                                                                    memory=goal_enc_outputs,
                                                                    mask=goal_attn_mask)

        # high_level attn
        context_merge = torch.cat([utt_weighted_context,
                                   cue_weighted_context.unsqueeze(1),
                                   goal_weighted_context], dim=1)     # [B, 3, 64]

        context_merge, ct_w = self.high_level_attention(query=dec_query,
                                                        memory=context_merge)

        out_input_list.append(context_merge)
        p_gen_input_list.append(context_merge)

        w_utt = ct_w[..., 0].clone()   # [batch_size, 1]
        w_cue = ct_w[..., 1].clone()
        w_goal = ct_w[..., 2].clone()

        kg_attn_dist_rep = kg_attn_dist.view(-1).repeat(cue_max_len, 1).transpose(0, 1)
        cue_attn_dist = kg_attn_dist_rep.mul(cue_attn.squeeze(1)).view(cur_batch_size, cue_sent_num, cue_max_len)

        utt_dist_ = w_utt * utt_attn_dist.squeeze(1)
        cue_dist_ = w_cue * cue_attn_dist.contiguous().view(cur_batch_size, -1)
        goal_dist_ = w_goal * goal_attn_dist.squeeze(1)

        p_gen_input = torch.cat(p_gen_input_list, dim=-1)   # [batch_size, 1, hid_size *2 + emb_size]
        p_gen = torch.sigmoid(self.p_gen_linear(p_gen_input).squeeze(-1))  # [batch_size, 1]

        out_input = torch.cat(out_input_list, dim=-1).squeeze(1)   # [batch_size, hid_size * 2]
        vocab_dist = self.output_layer(out_input)    # [batch_size, vocab_size]
        vocab_dist = p_gen * vocab_dist

        if oovs_max > 0:
            extra_zeros = torch.zeros([cur_batch_size, oovs_max], device=self.device)
            vocab_dist = torch.cat([vocab_dist, extra_zeros], dim=-1)

        extend_vocab_merge = torch.cat([valid_src_extend_vocab,
                                        valid_cue_extend_vocab.contiguous().view(cur_batch_size, -1),
                                        valid_goal_extend_vocab],
                                       dim=-1)

        attn_dist_merge = torch.cat([utt_dist_, cue_dist_, goal_dist_], dim=-1)
        attn_dist_merge = (1 - p_gen) * attn_dist_merge
        final_dist = vocab_dist.scatter_add(1, extend_vocab_merge, attn_dist_merge)
        state.dec_init_context = context_merge
        state.utt_hidden = dec_hidden_ori

        return final_dist, state

    def forward(self, inputs, state, is_training):

        inputs, lengths, src_extend_vocab, cue_extend_vocab, goal_extend_vocab, merge_oovs_str = inputs

        oovs_max_len = max([len(i) for i in merge_oovs_str])
        src_extend_vocab = src_extend_vocab[0][..., 1:-1]
        cue_extend_vocab, cue_lengths = cue_extend_vocab[0][..., 1:-1], cue_extend_vocab[1] - 2
        goal_extend_vocab = goal_extend_vocab[0][..., 1:-1]
        # cue_extend_vocab.size() == [batch_size, cue_sent_num, cue_sent_max_len]
        # cue_len_list == [batch_size, cue_sent_num]

        batch_size, max_len = inputs.size()
        out_inputs = inputs.new_zeros(
            size=(batch_size, max_len, self.output_size + oovs_max_len),
            dtype=torch.float)

        # sort by lengths
        sorted_lengths, indices = lengths.sort(descending=True)
        inputs = inputs.index_select(0, indices)   # inputs.size == [batch_size, max_dec_step]

        state = state.index_select(indices)
        src_entend_vocab = src_extend_vocab.index_select(0, indices)
        cue_extend_vocab = cue_extend_vocab.index_select(0, indices)
        goal_extend_vocab = goal_extend_vocab.index_select(0, indices)

        # number of valid input (i.e. not padding index) in each time step
        num_valid_list = sequence_mask(sorted_lengths).int().sum(dim=0)

        # num_valid_list.size() == [max_seq_len]
        for i, num_valid in enumerate(num_valid_list):
            # inputs.size == [batch_size, seq_len]
            dec_input = inputs[:num_valid, i]
            valid_state = state.slice_select(num_valid)
            valid_src_entend_vocab = src_entend_vocab[:num_valid]
            valid_cue_extend_vocab = cue_extend_vocab[:num_valid]
            valid_goal_extend_vocab = goal_extend_vocab[:num_valid]

            out_input, valid_state = self.decode(
                input=dec_input,
                state=valid_state,
                oovs_max=oovs_max_len,
                valid_src_extend_vocab=valid_src_entend_vocab,
                valid_cue_extend_vocab=valid_cue_extend_vocab,
                valid_goal_extend_vocab=valid_goal_extend_vocab,
                is_training=is_training,
            )

            state.utt_hidden[:, :num_valid] = valid_state.utt_hidden
            state.dec_init_context[:num_valid, ...] = valid_state.dec_init_context
            out_inputs[:num_valid, i] = out_input

        # Resort
        _, inv_indices = indices.sort()
        state = state.index_select(inv_indices)
        out_inputs = out_inputs.index_select(0, inv_indices)
        return out_inputs, state
