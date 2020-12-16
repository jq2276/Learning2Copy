#!/usr/bin/env python

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F

from base_model import BaseModel
from util.embedder import Embedder
from encoder import RNNEncoder
from decoder import Decoder
from util.criterions import NLLLoss, CopyGeneratorLoss
from util.misc import Pack
from evaluation.metrics import accuracy, perplexity
from util.attention import Attention


class GOKC(BaseModel):

    def __init__(self, src_vocab_size, tgt_vocab_size, cue_vocab_size, goal_vocab_size, embed_size, hidden_size,
                 padding_idx=None, num_layers=1, bidirectional=True, attn_mode="mlp", with_bridge=False,
                 tie_embedding=False, dropout=0.0, use_gpu=False, use_bow=False, use_kd=False,
                 use_posterior=False, device=None, unk_idx=None, force_copy=True, stage=None):
        super().__init__()

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.cue_vocab_size = cue_vocab_size
        self.goal_vocab_size = goal_vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.attn_mode = attn_mode
        self.with_bridge = with_bridge
        self.tie_embedding = tie_embedding
        self.dropout = dropout
        self.use_gpu = use_gpu
        self.use_bow = use_bow
        self.use_kd = use_kd
        self.use_posterior = use_posterior
        self.baseline = 0
        self.device = device if device >= 0 else "cpu"
        self.unk_idx = unk_idx
        self.force_copy = force_copy
        self.stage = stage

        # the utterance embedding
        enc_embedder = Embedder(num_embeddings=self.src_vocab_size,
                                embedding_dim=self.embed_size, padding_idx=self.padding_idx)

        self.utt_encoder = RNNEncoder(input_size=self.embed_size, hidden_size=self.hidden_size,
                                      embedder=enc_embedder, num_layers=self.num_layers,
                                      bidirectional=self.bidirectional, dropout=self.dropout)

        if self.with_bridge:
            self.utt_bridge = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.Tanh())
            self.goal_bridge = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.Tanh())

        # self.prior_query_mlp = nn.Sequential(nn.Linear(self.hidden_size * 2, self.hidden_size), nn.Tanh())
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size * 2, 1)

        if self.tie_embedding:
            # share the same embedding with utt encoder
            assert self.src_vocab_size == self.tgt_vocab_size == self.cue_vocab_size == self.goal_vocab_size
            self.dec_embedder = enc_embedder
            knowledge_embedder = enc_embedder
            goal_embedder = enc_embedder
        else:
            self.dec_embedder = Embedder(num_embeddings=self.tgt_vocab_size,
                                         embedding_dim=self.embed_size,
                                         padding_idx=self.padding_idx)
            knowledge_embedder = Embedder(num_embeddings=self.cue_vocab_size,
                                          embedding_dim=self.embed_size,
                                          padding_idx=self.padding_idx)
            goal_embedder = Embedder(num_embeddings=self.goal_vocab_size,
                                     embedding_dim=self.embed_size,
                                     padding_idx=self.padding_idx)

        self.knowledge_encoder = RNNEncoder(input_size=self.embed_size,
                                            hidden_size=self.hidden_size,
                                            embedder=knowledge_embedder,
                                            num_layers=self.num_layers,
                                            bidirectional=self.bidirectional,
                                            dropout=self.dropout)

        self.goal_encoder = RNNEncoder(input_size=self.embed_size,
                                       hidden_size=self.hidden_size,
                                       embedder=goal_embedder,
                                       num_layers=self.num_layers,
                                       bidirectional=self.bidirectional,
                                       dropout=self.dropout)

        self.prior_attention = Attention(query_size=self.hidden_size,
                                         memory_size=self.hidden_size,
                                         hidden_size=self.hidden_size,
                                         mode="dot",
                                         device=self.device)

        self.posterior_attention = Attention(query_size=self.hidden_size,
                                             memory_size=self.hidden_size,
                                             hidden_size=self.hidden_size,
                                             mode="dot",
                                             device=self.device)

        self.decoder = Decoder(input_size=self.embed_size, hidden_size=self.hidden_size,
                               output_size=self.tgt_vocab_size, embedder=self.dec_embedder,
                               num_layers=self.num_layers, attn_mode=self.attn_mode,
                               memory_size=self.hidden_size, dropout=self.dropout,
                               device=self.device)

        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        if self.use_bow:
            self.bow_output_layer = nn.Sequential(
                    nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
                    nn.Tanh(),
                    nn.Linear(in_features=self.hidden_size, out_features=self.tgt_vocab_size),
                    nn.LogSoftmax(dim=-1))

        if self.use_kd:
            self.knowledge_dropout = nn.Dropout(self.dropout)

        if self.padding_idx is not None:
            self.weight = torch.ones(self.tgt_vocab_size)
            self.weight[self.padding_idx] = 0
        else:
            self.weight = None

        self.nll_loss = NLLLoss(weight=self.weight, ignore_index=self.padding_idx,
                                reduction='mean')
        self.copy_gen_loss = CopyGeneratorLoss(vocab_size=self.tgt_vocab_size,
                                               force_copy=self.force_copy,
                                               unk_index=self.unk_idx,
                                               ignore_index=self.padding_idx)

        self.kl_loss = torch.nn.KLDivLoss(reduction="mean")

        if self.use_gpu:
            self.cuda()
            self.weight = self.weight.cuda()

    def encode(self, inputs, is_training=False):
        """
        encode
        """
        outputs = Pack()
        # utt encoding info.
        utt_inputs = _, utt_lengths = inputs.src[0][:, 1:-1], inputs.src[1] - 2
        utt_enc_outputs, utt_enc_hidden = self.utt_encoder(utt_inputs)
        if self.with_bridge:
            utt_enc_hidden = self.utt_bridge(utt_enc_hidden)

        # goal encoding info.
        goal_inputs = _, goal_lengths = inputs.goal[0][:, 1:-1], inputs.goal[1] - 2
        # goal_enc_hidden.size == [1, batch_size, hidden_size]
        goal_enc_outputs, goal_enc_hidden = self.goal_encoder(goal_inputs)
        if self.with_bridge:
            goal_enc_hidden = self.goal_bridge(goal_enc_hidden)

        # knowledge
        batch_size, sent_num, sent = inputs.cue[0].size()
        tmp_len = inputs.cue[1]   # [batch_size, sent_num]
        tmp_len[tmp_len > 0] -= 2

        cue_inputs = inputs.cue[0].view(-1, sent)[:, 1:-1], tmp_len.view(-1)
        # cue_enc_hidden.size() == [1, batch_size * sent_num, hidden_size]
        # cue_enc_outputs.size() == [batch_size * sent_num, sent_len, hidden_size]
        cue_enc_outputs, cue_enc_hidden = self.knowledge_encoder(cue_inputs)

        # cue_enc_hidden[-1].size() == [batch_size * sent_num, hidden_size]
        # [batch_size, sent_num, hidden_size]
        cue_enc_outputs = cue_enc_outputs.view(batch_size, sent_num, cue_enc_outputs.size(-2), -1)
        cue_outputs = cue_enc_hidden[-1].view(batch_size, sent_num, -1)

        # Attention
        p_U = self.tanh(self.fc1(utt_enc_hidden[-1].unsqueeze(0)))
        p_G = self.tanh(self.fc2(goal_enc_hidden[-1].unsqueeze(0)))
        k = self.sigmoid(self.fc3(torch.cat([p_U, p_G], dim=-1)))
        prior_query = self.tanh(k * utt_enc_hidden + (1 - k) * goal_enc_hidden)
        weighted_cue, cue_attn = self.prior_attention(query=prior_query[-1].unsqueeze(1),
                                                      memory=self.tanh(cue_outputs),
                                                      mask=inputs.cue[1].eq(0))

        prior_attn = cue_attn.squeeze(1)
        outputs.add(prior_attn=prior_attn)

        posterior_attn = None
        if self.use_posterior:
            tgt_enc_inputs = inputs.tgt[0][:, 1:-1], inputs.tgt[1] - 2
            _, tgt_enc_hidden = self.knowledge_encoder(tgt_enc_inputs)

            posterior_weighted_cue, posterior_attn = self.posterior_attention(
                query=tgt_enc_hidden[-1].unsqueeze(1),
                memory=self.tanh(cue_outputs),
                mask=inputs.cue[1].eq(0))

            posterior_attn = posterior_attn.squeeze(1)
            outputs.add(posterior_attn=posterior_attn)

            knowledge = posterior_weighted_cue
            if self.use_kd:
                knowledge = self.knowledge_dropout(knowledge)

            if self.use_bow:
                bow_logits = self.bow_output_layer(knowledge)
                outputs.add(bow_logits=bow_logits)

        # Initialize the context vector of decoder
        dec_init_context = torch.zeros(size=[batch_size, 1, self.hidden_size],
                                       dtype=torch.float,
                                       device=self.device)

        dec_init_state = self.decoder.initialize_state(
            utt_hidden=utt_enc_hidden,
            utt_outputs=utt_enc_outputs if self.attn_mode else None,
            utt_input_len=utt_lengths if self.attn_mode else None,
            cue_hidden=cue_outputs.transpose(0, 1),
            cue_outputs=cue_enc_outputs if self.attn_mode else None,
            cue_input_len=tmp_len if self.attn_mode else None,
            goal_hidden=goal_enc_hidden,
            goal_outputs=goal_enc_outputs,
            goal_input_len=goal_lengths,
            pr_attn_dist=prior_attn,
            po_attn_dist=posterior_attn,
            dec_init_context=dec_init_context
        )
        return outputs, dec_init_state

    def decode(self, input, state, oovs_max, src_extend_vocab, cue_extend_vocab, goal_extend_vocab):

        output, dec_state = self.decoder.decode(input=input,
                                                 state=state,
                                                 oovs_max=oovs_max,
                                                 valid_src_extend_vocab=src_extend_vocab,
                                                 valid_cue_extend_vocab=cue_extend_vocab,
                                                 valid_goal_extend_vocab=goal_extend_vocab)

        return output, dec_state

    def forward(self, enc_inputs, dec_inputs, is_training=False):

        outputs, dec_init_state = self.encode(
                enc_inputs, is_training=is_training)

        log_probs, _ = self.decoder(dec_inputs, dec_init_state, is_training=is_training)
        outputs.add(logits=log_probs)
        return outputs

    def collect_metrics(self, outputs, oovs_target, no_extend_target, epoch=-1):

        num_samples = no_extend_target.size(0)
        metrics = Pack(num_samples=num_samples)
        loss = 0
        logits = outputs.logits   # [batch_size, dec_seq_len, vocab_size]
        nll_loss_ori = self.copy_gen_loss(scores=logits.transpose(1, 2).contiguous(),
                                          align=oovs_target,
                                          target=no_extend_target)   # [batch_size, tgt_len]

        nll_loss = torch.mean(torch.sum(nll_loss_ori, dim=-1))

        num_words = no_extend_target.ne(self.padding_idx).sum()  # .item()
        ppl = nll_loss_ori.sum() / num_words
        ppl = ppl.exp()
        acc = accuracy(logits, no_extend_target, padding_idx=self.padding_idx)
        metrics.add(nll=(nll_loss, num_words), acc=acc, ppl=ppl)

        if self.use_posterior:
            kl_loss = self.kl_loss(torch.log(outputs.prior_attn + 1e-20),
                                   outputs.posterior_attn.detach())

            metrics.add(kl=kl_loss)

            if self.stage == 1:
                loss += nll_loss
                loss += kl_loss

            if self.use_bow:
                bow_logits = outputs.bow_logits   # size = [batch_size, 1, vocab_size]
                bow_logits = bow_logits.repeat(1, no_extend_target.size(-1), 1)
                bow = self.nll_loss(bow_logits, no_extend_target)
                loss += bow
                metrics.add(bow=bow)

        else:
            loss += nll_loss

        metrics.add(loss=loss)
        return metrics

    def iterate(self, inputs, optimizer=None, grad_clip=None, is_training=False, epoch=-1):
        enc_inputs = inputs
        dec_inputs = (inputs.tgt[0][:, :-1],
                      inputs.tgt[1] - 1,
                      inputs.src_extend_vocab,
                      inputs.cue_extend_vocab,
                      inputs.goal_extend_vocab,
                      inputs.merge_oovs_str)

        outputs = self.forward(enc_inputs, dec_inputs, is_training=is_training)

        oovs_target = inputs.tgt_oovs_vocab[0][:, 1:]
        no_extend_target = inputs.tgt[0][:, 1:]
        metrics = self.collect_metrics(outputs, oovs_target, no_extend_target, epoch=epoch)

        loss = metrics.loss
        if torch.isnan(loss):
            raise ValueError("nan loss encountered")

        if is_training:
            assert optimizer is not None
            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                clip_grad_norm_(parameters=self.parameters(),
                                max_norm=grad_clip)
            optimizer.step()
        return metrics
