#!/usr/bin/env python

import os
import torch

from tqdm import tqdm
from util.field import tokenize
from util.field import TextField
# from util.field import NumberField
from util.dataset import Dataset


class Corpus(object):
    """
    Corpus
    """
    def __init__(self,
                 data_dir,
                 data_prefix,
                 min_freq=0,
                 max_vocab_size=None):
        self.data_dir = data_dir
        self.data_prefix = data_prefix
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size

        prepared_data_file = data_prefix + "_" + str(max_vocab_size) + ".data.pt"
        prepared_vocab_file = data_prefix + "_" + str(max_vocab_size) + ".vocab.pt"

        self.prepared_data_file = os.path.join(data_dir, prepared_data_file)
        self.prepared_vocab_file = os.path.join(data_dir, prepared_vocab_file)
        self.fields = {}

        self.filter_pred = None
        self.data = None

    def load(self):
        """
        load
        """
        if not (os.path.exists(self.prepared_data_file) and
                os.path.exists(self.prepared_vocab_file)):
            self.build()
        self.load_vocab(self.prepared_vocab_file)
        self.load_data(self.prepared_data_file)
        self.padding_idx = self.TGT.stoi[self.TGT.pad_token]
        self.unk_idx = self.TGT.stoi[self.TGT.unk_token]

    def reload(self, data_type='test'):
        data_file = os.path.join(self.data_dir, self.data_prefix + "." + data_type)
        data_raw = self.read_data(data_file, data_type="test")
        data_examples = self.build_examples(data_raw)
        self.data[data_type] = Dataset(data_examples)

        print("Number of examples:",
              " ".join("{}-{}".format(k.upper(), len(v)) for k, v in self.data.items()))

    def load_data(self, prepared_data_file=None):
        """
        load_data
        """
        prepared_data_file = prepared_data_file or self.prepared_data_file
        print("Loading prepared data from {} ...".format(prepared_data_file))
        data = torch.load(prepared_data_file)
        self.data = {"train": Dataset(data['train']),
                     "valid": Dataset(data["valid"]),
                     "test": Dataset(data["test"])}
        print("Number of examples:",
              " ".join("{}-{}".format(k.upper(), len(v)) for k, v in self.data.items()))

    def load_vocab(self, prepared_vocab_file):
        """
        load_vocab
        """
        prepared_vocab_file = prepared_vocab_file or self.prepared_vocab_file
        print("Loading prepared vocab from {} ...".format(prepared_vocab_file))
        vocab_dict = torch.load(prepared_vocab_file)

        for name, vocab in vocab_dict.items():
            if name in self.fields:
                self.fields[name].load_vocab(vocab)
        print("Vocabulary size of fields:",
              " ".join("{}-{}".format(name.upper(), field.vocab_size) 
                for name, field in self.fields.items() 
                    if isinstance(field, TextField)))

    def read_data(self, data_file, data_type=None):
        """
        Returns
        -------
        data: ``List[Dict]``
        """
        raise NotImplementedError

    def build_vocab(self, data):
        """
        Args
        ----
        data: ``List[Dict]``
        """
        # data仅仅是训练数据
        field_data_dict = {}
        for name in data[0].keys():
            field = self.fields.get(name)
            if isinstance(field, TextField):
                xs = [x[name] for x in data]
                if field not in field_data_dict:
                    field_data_dict[field] = xs
                else:
                    field_data_dict[field] += xs

        vocab_dict = {}
        for name, field in self.fields.items():
            if field in field_data_dict:
                print("Building vocabulary of field {} ...".format(name.upper()))
                if field.vocab_size == 0:
                    field.build_vocab(field_data_dict[field],
                                      min_freq=self.min_freq,
                                      max_size=self.max_vocab_size)
                vocab_dict[name] = field.dump_vocab()
        return vocab_dict

    def find_oovs(self, strings, name):
        oovs = []
        strings = tokenize(strings)
        unk_id = self.fields[name].stoi[self.fields[name].unk_token]
        for w in strings:
            i = self.fields[name].stoi.get(w, unk_id)
            if (i == unk_id) and (w not in oovs):
                oovs.append(w)
        return oovs

    def get_idx_from_extend_vocab(self, strings, name, oovs_list):
        ids = []
        strings = tokenize(strings)
        unk_id = self.fields[name].stoi[self.fields[name].unk_token]
        bos_token = self.fields[name].stoi[self.fields[name].bos_token] if self.fields[name].bos_token else None
        eos_token = self.fields[name].stoi[self.fields[name].eos_token] if self.fields[name].eos_token else None
        for w in strings:
            i = self.fields[name].stoi.get(w, unk_id)
            if i == unk_id:
                if w in oovs_list:
                    w_oov_idx = oovs_list.index(w)
                    ids.append(self.fields[name].vocab_size + w_oov_idx)
                else:

                    ids.append(i)
            else:
                ids.append(i)
        if bos_token is not None:
            ids.insert(0, bos_token)
        if eos_token is not None:
            ids.append(eos_token)
        assert len(ids) - 2 == len(strings)   # remove the length of <bos> and <eos>
        return ids

    def get_idx_from_oovs_vocab(self, strings, name, oovs_list):
        ids = []
        strings = tokenize(strings)
        unk_id = self.fields[name].stoi[self.fields[name].unk_token]
        bos_token = self.fields[name].stoi[self.fields[name].bos_token] if self.fields[name].bos_token else None
        eos_token = self.fields[name].stoi[self.fields[name].eos_token] if self.fields[name].eos_token else None
        for w in strings:
            if w in oovs_list:
                w_oov_idx = oovs_list.index(w)
                w_oov_idx = w_oov_idx + self.fields[name].vocab_size
                ids.append(w_oov_idx)
            else:
                ids.append(unk_id)

        if bos_token is not None:
            ids.insert(0, bos_token)
        if eos_token is not None:
            ids.append(eos_token)
        assert len(ids) - 2 == len(strings)
        return ids

    def build_examples(self, data):
        examples = []
        for raw_data in tqdm(data):
            example = dict()
            example["src"] = self.fields["src"].numericalize(raw_data["src"])
            example["tgt"] = self.fields["tgt"].numericalize(raw_data["tgt"])
            example["cue"] = self.fields["cue"].numericalize(raw_data["cue"])
            example["goal"] = self.fields["goal"].numericalize(raw_data["goal"])

            # build oovs vocab
            src_oovs = self.find_oovs(raw_data["src"], "src")
            cue_oovs = [self.find_oovs(i, "cue") for i in raw_data["cue"]]
            cue_oovs_merge = [j for i in cue_oovs for j in i]
            goal_oovs = self.find_oovs(raw_data["goal"], "goal")

            merge_oovs = src_oovs + cue_oovs_merge + goal_oovs
            merge_oovs = sorted(set(merge_oovs), key=merge_oovs.index)
            example["merge_oovs_str"] = merge_oovs
            example["src_extend_vocab"] = self.get_idx_from_extend_vocab(raw_data["src"], "src", merge_oovs)
            example["cue_extend_vocab"] = [self.get_idx_from_extend_vocab(s, "cue", merge_oovs)
                                           for s in raw_data["cue"]]
            example["goal_extend_vocab"] = self.get_idx_from_extend_vocab(raw_data["goal"], "goal", merge_oovs)
            example["tgt_oovs_vocab"] = self.get_idx_from_oovs_vocab(raw_data["tgt"], "tgt", merge_oovs)

            examples.append(example)
        return examples

    def build(self):
        """
        build
        """
        print("Start to build corpus!")
        train_file = os.path.join(self.data_dir, self.data_prefix + ".train")
        valid_file = os.path.join(self.data_dir, self.data_prefix + ".dev")
        test_file = os.path.join(self.data_dir, self.data_prefix + ".test")

        print("Reading data ...")

        train_raw = self.read_data(train_file, data_type="train")
        valid_raw = self.read_data(valid_file, data_type="valid")
        test_raw = self.read_data(test_file, data_type="test")
        vocab = self.build_vocab(train_raw)
        print("Building TRAIN examples ...")
        train_data = self.build_examples(train_raw)
        print("Building VALID examples ...")
        valid_data = self.build_examples(valid_raw)
        print("Building TEST examples ...")
        test_data = self.build_examples(test_raw)
        # <<<

        data = {"train": train_data,
                "valid": valid_data,
                "test": test_data}

        print("Saving prepared vocab ...")
        torch.save(vocab, self.prepared_vocab_file)
        print("Saved prepared vocab to '{}'".format(self.prepared_vocab_file))

        print("Saving prepared data ...")
        torch.save(data, self.prepared_data_file)
        print("Saved prepared data to '{}'".format(self.prepared_data_file))

    def create_batches(self, batch_size, data_type="train",
                       shuffle=False, device=None):
        """
        create_batches
        """
        try:
            data = self.data[data_type]
            data_loader = data.create_batches(batch_size, shuffle, device)
            return data_loader
        except KeyError:
            raise KeyError("Unsported data type: {}!".format(data_type))

    def transform(self, data_file, batch_size,
                  data_type="test", shuffle=False, device=None):
        """
        Transform raw text from data_file to Dataset and create data loader.
        """
        raw_data = self.read_data(data_file, data_type=data_type)
        examples = self.build_examples(raw_data)
        data = Dataset(examples)
        data_loader = data.create_batches(batch_size, shuffle, device)
        return data_loader


class KnowledgeCorpus(Corpus):
    """
    KnowledgeCorpus
    """
    def __init__(self,
                 data_dir,
                 data_prefix,
                 min_freq=0,
                 max_vocab_size=None,
                 min_len=0,
                 max_len=100,
                 embed_file=None,
                 share_vocab=False):
        super(KnowledgeCorpus, self).__init__(data_dir=data_dir,
                                              data_prefix=data_prefix,
                                              min_freq=min_freq,
                                              max_vocab_size=max_vocab_size)
        self.min_len = min_len
        self.max_len = max_len
        self.share_vocab = share_vocab
        self.SRC = TextField(tokenize_fn=tokenize,
                             embed_file=embed_file)
        if self.share_vocab:
            self.TGT = self.SRC
            self.CUE = self.SRC
            self.GOAL = self.SRC
        else:
            self.TGT = TextField(tokenize_fn=tokenize,
                                 embed_file=embed_file)
            self.CUE = TextField(tokenize_fn=tokenize,
                                 embed_file=embed_file)
            self.GOAL = TextField(tokenize_fn=tokenize,
                                  embed_file=embed_file)

        self.fields = {'src': self.SRC, 'tgt': self.TGT, 'cue': self.CUE, 'goal': self.GOAL}

        def src_filter_pred(src):
            return min_len <= len(self.SRC.tokenize_fn(src)) <= max_len

        def tgt_filter_pred(tgt):
            return min_len <= len(self.TGT.tokenize_fn(tgt)) <= max_len

        self.filter_pred = lambda ex: src_filter_pred(ex['src']) and tgt_filter_pred(ex['tgt'])

    def read_data(self, data_file, data_type="train"):
        data = []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                src, tgt, knowledge, goal = line.strip().split('\t')[:4]
                filter_knowledge = []
                for sent in knowledge.split('\1'):
                    filter_knowledge.append(' '.join(sent.split()[:self.max_len]))
                data.append({'src': src, 'tgt': tgt, 'cue': filter_knowledge, 'goal': goal})

        filtered_num = len(data)
        # if self.filter_pred is not None:
        #     data = [ex for ex in data if self.filter_pred(ex)]
        filtered_num -= len(data)
        print(
            "Read {} {} examples ({} filtered)".format(len(data), data_type.upper(), filtered_num))
        return data
