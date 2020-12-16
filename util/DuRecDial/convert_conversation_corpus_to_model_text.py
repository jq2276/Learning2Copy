#!/usr/bin/env python

import sys
import json
import collections


def preprocessing_for_one_conversation(text,
                                       topic_generalization=False):

    conversation = json.loads(text.strip(), encoding="utf-8",
                              object_pairs_hook=collections.OrderedDict)
    goal = conversation["goal"]
    kg = conversation["kg"]
    history = conversation["history"] if conversation["history"] != [] else ["[START]"]
    response = conversation["response"] if "response" in conversation else "[NAN]"
    situation = conversation["situation"]
    user_profile = conversation["user_profile"]

    entity_dict = dict()
    entity_dict["<PRF_NAME>"] = user_profile["姓名"]
    entity_dict["<PRF_RES>"] = user_profile["居住地"]

    # obtain the background knowledge
    kg_facts = '\1'.join([' '.join(spo) for spo in kg])

    background_kg = kg_facts + "\1" + situation

    history_str = " ".join(history)
    goal = " ".join(goal)

    # input of the model
    src = history_str
    knw = background_kg
    response = response

    model_text = '\t'.join([src, response, knw, goal])

    if topic_generalization:
        entity_list = sorted(entity_dict.items(), key=lambda item: len(item[1]), reverse=True)
        for key, value in entity_list:
            model_text = model_text.replace(value, key)
    return model_text, entity_dict


def convert_conversation_corpus_to_model_text(corpus_file, text_file, topic_file, \
                                              topic_generalization=False):
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
                                              sys.argv[4])


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited from the program ealier!")
