#!/usr/bin/env python

from __future__ import print_function

import json
import collections
import sys

P_COVER = ['代表作', '主演', '好友', '导演', '搭档', '丈夫', '妻子', '家人']


class CreateLabel:
    def __init__(self):
        self.all_dict = {}
        self.count_p = {}
        self.all_p_set = set()
        self.all_o_set = set()

    def add2dict(self, _p, _o):
        if (_p not in self.all_p_set) and (_o not in self.all_o_set):
            self.all_dict["<{}_#>".format(_p)] = _o
            self.count_p[_p] = 0
            self.all_p_set.add(_p)
            self.all_o_set.add(_o)
        elif (_p in self.all_p_set) and (_o not in self.all_o_set):
            self.count_p[_p] += 1
            self.all_dict["<{}_{}>".format(_p, chr(ord("#")+self.count_p[_p]))] = _o
            self.all_o_set.add(_o)
        elif (_p not in self.all_p_set) and (_o in self.all_o_set):
            self.all_dict["<{}_#>".format(_p)] = _o
            self.count_p[_p] = 0
            self.all_p_set.add(_p)
        else:
            pass


class AddKgLabel:
    def __init__(self, kg):
        self.kg = kg
        self.create_label = CreateLabel()

    def add_kg(self):
        for each_idx in range(self.kg.__len__()):
            if self.kg[each_idx][1] in P_COVER:
                self.create_label.add2dict(self.kg[each_idx][1], self.kg[each_idx][2])
        return self.create_label.all_dict


def data_resplit(data):
    # remove the redundant whitespace
    new_list_all = list()
    if isinstance(data, str):
        return " ".join(data.strip().split())
    elif isinstance(data, list):
        new_list = list()
        for d in data:   # [["", "", ""], ["", "", ""], ...] --> ["", "", ""] --> ""
            resplit_str = data_resplit(d)
            new_list.append(resplit_str)
        new_list_all.extend(new_list)
    else:
        raise NotImplementedError("Data type out of range.")
    return new_list_all


def preprocessing_for_one_conversation(text,
                                       topic_generalization=False):

    # The data preprocessing we are strictly following the baseline (https://github.com/PaddlePaddle/models/
    # tree/develop/PaddleNLP/Research/ACL2019-DuConv)

    conversation = json.loads(text.strip(), encoding="utf-8", \
                              object_pairs_hook=collections.OrderedDict)

    goal = data_resplit(conversation["goal"])
    knowledge = data_resplit(conversation["knowledge"])
    history = data_resplit(conversation["history"] if conversation["history"] != [] else ["nan"])
    response = data_resplit(conversation["response"] if "response" in conversation else "null")

    # building the knowledge dict
    kg_dict = \
        AddKgLabel(kg=knowledge).add_kg()

    # building the goal dict
    topic_a = goal[0][1]
    topic_b = goal[0][2]
    for i, [s, p, o] in enumerate(knowledge):
        if u"领域" == p:
            if topic_a == s:
                domain_a = o
            elif topic_b == s:
                domain_b = o

    topic_dict = {}
    if u"电影" == domain_a:
        topic_dict["<video_topic_a>"] = topic_a
    else:
        topic_dict["<person_topic_a>"] = topic_a

    if u"电影" == domain_b:
        topic_dict["<video_topic_b>"] = topic_b
    else:
        topic_dict["<person_topic_b>"] = topic_b

    topic_dict.update(kg_dict)   # dict merge

    goal = ' '.join([' '.join(spo) for spo in goal])
    knowledge_str1 = ' '.join([' '.join(spo) for spo in knowledge])
    knowledge_str2 = '\1'.join([' '.join(spo) for spo in knowledge])
    history_str = ' '.join(history)
    src = goal + " " + knowledge_str1 + " : " + history_str
    model_text = '\t'.join([src, response, knowledge_str2, goal])
    if topic_generalization:
        topic_list = sorted(topic_dict.items(), key=lambda item: len(item[1]), reverse=True)
        for key, value in topic_list:
            model_text = model_text.replace(value, key)

    return model_text, topic_dict


def convert_conversation_corpus_to_model_text(corpus_file, text_file, topic_file, \
                                              topic_generalization=True):
    """
    convert_conversation_corpus_to_model_text
    """
    fout_text = open(text_file, 'w', encoding="utf-8")
    fout_topic = open(topic_file, 'w', encoding="utf-8")
    with open(corpus_file, 'r', encoding="utf-8") as f:
        for i, line in enumerate(f):
            model_text, topic_dict = preprocessing_for_one_conversation(
                line.strip(), topic_generalization=topic_generalization)

            topic_dict = json.dumps(topic_dict, ensure_ascii=False)

            fout_text.write(model_text + "\n")
            fout_topic.write(topic_dict + "\n")

    fout_text.close()
    fout_topic.close()


def main():
    """
    main
    """

    convert_conversation_corpus_to_model_text(sys.argv[1],
                                              sys.argv[2],
                                              sys.argv[3],
                                              int(sys.argv[4]) > 0)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited from the program ealier!")
