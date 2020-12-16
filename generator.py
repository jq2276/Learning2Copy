#!/usr/bin/env python


import torch

from util.misc import sequence_mask
from evaluation.metrics import perplexity
from util.criterions import CopyGeneratorLoss


class TopKGenerator(object):

    def __init__(self,
                 model,
                 src_field,
                 tgt_field,
                 cue_field=None,
                 max_length=10,
                 ignore_unk=True,
                 length_average=True,
                 use_gpu=False,
                 dec_embedder=None,
                 device=None,
                 tgt_vocab_size=None,
                 force_copy=True):
        self.model = model.cuda() if use_gpu else model
        self.src_field = src_field
        self.tgt_field = tgt_field
        self.cue_field = cue_field
        self.k = 3
        self.max_length = max_length
        self.ignore_unk = ignore_unk
        self.length_average = length_average
        self.use_gpu = use_gpu
        self.PAD = tgt_field.stoi[tgt_field.pad_token]
        self.UNK = tgt_field.stoi[tgt_field.unk_token]
        self.BOS = tgt_field.stoi[tgt_field.bos_token]
        self.EOS = tgt_field.stoi[tgt_field.eos_token]
        self.V = self.tgt_field.vocab_size
        self.dec_embedder = dec_embedder
        self.device = device if device >= 0 else "cpu"
        self.tgt_vocab_size = tgt_vocab_size
        self.force_copy = force_copy
        self.copy_gen_loss = CopyGeneratorLoss(vocab_size=self.tgt_vocab_size,
                                               force_copy=self.force_copy,
                                               unk_index=self.UNK,
                                               ignore_index=self.PAD)

    def sub_oovs(self, input_var):
        for idx, ele in enumerate(input_var):
            if ele >= self.dec_embedder.num_embeddings:
                input_var[idx] = self.UNK
        return input_var

    def forward(self, inputs):
        oovs_max = inputs.merge_oovs_str
        oovs_max_len = max([len(i) for i in oovs_max])
        src_extend_vocab = inputs.src_extend_vocab[0][:, 1:-1]
        cue_extend_vocab = inputs.cue_extend_vocab[0][..., 1:-1]
        goal_extend_vocab = inputs.goal_extend_vocab[0][..., 1:-1]

        self.model.eval()

        with torch.no_grad():
            enc_outputs, dec_state = self.model.encode(inputs)
            preds, lens, _ = self.decode(dec_state, oovs_max_len,
                                                           src_extend_vocab, cue_extend_vocab, goal_extend_vocab)

        return enc_outputs, preds, lens

    @staticmethod
    def inflate_tensor(X, times):
        """
        inflate X from shape (batch_size, ...) to shape (batch_size*times, ...)
        for first decoding of beam search
        """
        sizes = X.size()

        if X.dim() == 1:
            X = X.unsqueeze(1)

        repeat_times = [1] * X.dim()
        repeat_times[1] = times
        X = X.repeat(*repeat_times).view(-1, *sizes[1:])
        return X

    def decode(self, dec_state, oovs_max_len=None,
               src_extend_vocab=None, cue_extend_vocab=None, goal_extend_vocab=None):

        long_tensor_type = torch.cuda.LongTensor if self.use_gpu else torch.LongTensor
        b = dec_state.get_batch_size()
        self.pos_index = (long_tensor_type(range(b)) * self.k).view(-1, 1)

        # Inflate the initial hidden states to be of size: (b*k, H)

        dec_state = dec_state.inflate(self.k)
        src_extend_vocab = self.inflate_tensor(src_extend_vocab, self.k)
        cue_extend_vocab = self.inflate_tensor(cue_extend_vocab, self.k)
        goal_extend_vocab = self.inflate_tensor(goal_extend_vocab, self.k)

        # Initialize the scores; for the first step,
        # ignore the inflated copies to avoid duplicate entries in the top k
        sequence_scores = long_tensor_type(b * self.k).float()
        sequence_scores.fill_(-float('inf'))
        sequence_scores.index_fill_(0, long_tensor_type(
            [i * self.k for i in range(b)]), 0.0)

        # Initialize the input vector
        input_var = long_tensor_type([self.BOS] * b * self.k)

        # Store decisions for backtracking
        stored_scores = list()
        stored_predecessors = list()
        stored_emitted_symbols = list()

        for t in range(1, self.max_length+1):
            input_var = self.sub_oovs(input_var)
            output, dec_state = self.model.decode(
                input=input_var,
                state=dec_state,
                oovs_max=oovs_max_len,
                src_extend_vocab=src_extend_vocab,
                cue_extend_vocab=cue_extend_vocab,
                goal_extend_vocab=goal_extend_vocab
            )

            softmax_output = (output+1e-20).log().squeeze(1)
            self.V = softmax_output.size(-1)

            # To get the full sequence scores for the new candidates, add the
            # local scores for t_i to the predecessor scores for t_(i-1)
            sequence_scores = sequence_scores.unsqueeze(1).repeat(1, self.V)
            if self.length_average and t > 1:
                sequence_scores = sequence_scores * \
                    (1 - 1/t) + softmax_output / t
            else:
                sequence_scores += softmax_output

            scores, candidates = sequence_scores.view(
                b, -1).topk(self.k, dim=1)

            # Reshape input = (b*k, 1) and sequence_scores = (b*k)
            input_var = (candidates % self.V)
            sequence_scores = scores.view(b * self.k)

            input_var = input_var.view(b * self.k)

            # Update fields for next timestep
            predecessors = (
                candidates // self.V + self.pos_index.expand_as(candidates)).view(b * self.k)

            dec_state = dec_state.index_select(predecessors)

            # Update sequence scores and erase scores for end-of-sentence symbol so that they aren't expanded
            stored_scores.append(sequence_scores.clone())
            eos_indices = input_var.data.eq(self.EOS)
            if eos_indices.nonzero().dim() > 0:
                sequence_scores.data.masked_fill_(eos_indices, -float('inf'))

            if self.ignore_unk:
                # Erase scores for UNK symbol so that they aren't expanded
                unk_indices = input_var.data.eq(self.UNK)
                if unk_indices.nonzero().dim() > 0:
                    sequence_scores.data.masked_fill_(
                        unk_indices, -float('inf'))

            # Cache results for backtracking
            stored_predecessors.append(predecessors)
            stored_emitted_symbols.append(input_var)

        predicts, scores, lengths = self._backtrack(
            stored_predecessors, stored_emitted_symbols, stored_scores, b)

        predicts = predicts[:, :1]
        scores = scores[:, :1]
        lengths = long_tensor_type(lengths)[:, :1]
        mask = sequence_mask(lengths, max_len=self.max_length).eq(0)
        predicts[mask] = self.PAD

        return predicts, lengths, scores

    def _backtrack(self, predecessors, symbols, scores, b):
        p = list()
        l = [[self.max_length] * self.k for _ in range(b)]

        # the last step output of the beams are not sorted
        # thus they are sorted here
        sorted_score, sorted_idx = scores[-1].view(
            b, self.k).topk(self.k, dim=1)

        # initialize the sequence scores with the sorted last step beam scores
        s = sorted_score.clone()

        # the number of EOS found in the backward loop below for each batch
        batch_eos_found = [0] * b

        t = self.max_length - 1
        # initialize the back pointer with the sorted order of the last step beams.
        # add self.pos_index for indexing variable with b*k as the first dimension.
        t_predecessors = (
            sorted_idx + self.pos_index.expand_as(sorted_idx)).view(b * self.k)

        while t >= 0:
            # Re-order the variables with the back pointer
            current_symbol = symbols[t].index_select(0, t_predecessors)
            # Re-order the back pointer of the previous step with the back pointer of the current step
            t_predecessors = predecessors[t].index_select(0, t_predecessors)

            # This tricky block handles dropped sequences that see EOS earlier.
            # The basic idea is summarized below:
            #
            #   Terms:
            #       Ended sequences = sequences that see EOS early and dropped
            #       Survived sequences = sequences in the last step of the beams
            #
            #       Although the ended sequences are dropped during decoding,
            #   their generated symbols and complete backtracking information are still
            #   in the backtracking variables.
            #   For each batch, everytime we see an EOS in the backtracking process,
            #       1. If there is survived sequences in the return variables, replace
            #       the one with the lowest survived sequence score with the new ended
            #       sequences
            #       2. Otherwise, replace the ended sequence with the lowest sequence
            #       score with the new ended sequence
            #
            eos_indices = symbols[t].data.eq(self.EOS).nonzero()
            if eos_indices.dim() > 0:
                for i in range(eos_indices.size(0)-1, -1, -1):
                    # Indices of the EOS symbol for both variables
                    # with b*k as the first dimension, and b, k for
                    # the first two dimensions
                    idx = eos_indices[i]
                    b_idx = idx[0].item() // self.k
                    # The indices of the replacing position
                    # according to the replacement strategy noted above
                    res_k_idx = self.k - (batch_eos_found[b_idx] % self.k) - 1
                    batch_eos_found[b_idx] += 1
                    res_idx = b_idx * self.k + res_k_idx

                    # Replace the old information in return variables
                    # with the new ended sequence information
                    t_predecessors[res_idx] = predecessors[t][idx[0]]
                    current_symbol[res_idx] = symbols[t][idx[0]]
                    s[b_idx, res_k_idx] = scores[t][idx[0]]
                    l[b_idx][res_k_idx] = t + 1

            # record the back tracked results
            p.append(current_symbol)

            t -= 1

        # Sort and re-order again as the added ended sequences may change
        # the order (very unlikely)
        s, re_sorted_idx = s.topk(self.k)
        for b_idx in range(b):
            l[b_idx] = [l[b_idx][k_idx.item()]
                        for k_idx in re_sorted_idx[b_idx, :]]

        re_sorted_idx = (
            re_sorted_idx + self.pos_index.expand_as(re_sorted_idx)).view(b * self.k)

        # Reverse the sequences and re-order at the same time
        # It is reversed because the backtracking happens in reverse time order
        predicts = torch.stack(p[::-1]).t()
        predicts = predicts[re_sorted_idx].contiguous().view(
            b, self.k, -1).data
        # p = [step.index_select(0, re_sorted_idx).view(b, self.k).data for step in reversed(p)]
        scores = s.data
        lengths = l

        # if self.k == 1:
        #     lengths = [_l[0] for _l in lengths]

        return predicts, scores, lengths

    def generate(self, batch_iter):
        results = []
        preds_all = []
        batch_cnt = 0
        for batch in batch_iter:
            enc_outputs, preds, _, = self.forward(inputs=batch)
            src = batch.src[0]
            tgt = batch.tgt[0]

            batch_oovs_str = batch.merge_oovs_str
            if batch_oovs_str is None:
                raise NotImplementedError
            src = self.src_field.denumericalize(src)
            tgt = self.tgt_field.denumericalize(tgt)
            preds = self.tgt_field.denumericalize(preds, batch_oovs_str)

            if 'cue' in batch:
                cue = self.cue_field.denumericalize(batch.cue[0].data)
                enc_outputs.add(cue=cue)

            temp_pred = []
            for sen in preds:
                preds = sen.split(" ")
                while "<unk>" in preds:
                    preds.remove("<unk>")
                preds = " ".join(preds)
                temp_pred.append(preds)

            preds_all += temp_pred
            enc_outputs.add(src=src, tgt=tgt, preds=temp_pred)
            result_batch = enc_outputs.flatten()
            results += result_batch
            batch_cnt += 1
        return results, preds_all
