#!/usr/bin/env python

import json
import collections
import sys
import re

GOAL_IDX = ["[1]", "[2]", "[3]", "[4]", "[5]", "[6]", "[7]", "[8]", "[9]"]


def extract_goal_type(goal_list):
    goal_type_list = []
    for e in goal_list[1:]:
        e = e.strip()
        if e != "(":
            goal_type_list.append(e)
        else:
            break
    return " ".join(goal_type_list)


def extract_goal_ent(_goal):

    entity_pattern = re.compile(r"[『](.+?)[』]")
    entity_list = re.findall(entity_pattern, string=_goal)
    entity_list = [" ".join(i.strip().split()) for i in entity_list]
    entity_list = sorted(set(entity_list), key=entity_list.index)
    return entity_list


def goals_format(no_fomt_goal):
    goal_index = "[?]"
    no_fomt_goal_list = no_fomt_goal.strip().split()
    if no_fomt_goal_list[0] in GOAL_IDX:
        goal_index = no_fomt_goal_list[0]
    goal_type = extract_goal_type(no_fomt_goal_list)
    goal_ent = " ".join(extract_goal_ent(no_fomt_goal))
    goal_str = goal_index + " " + goal_type + " " + goal_ent
    return goal_str


def bot_prior_f(_goals_0):
    goal_pattern = re.compile(r"\(.+\)")
    goal_content = re.findall(pattern=goal_pattern, string=_goals_0)
    goal_content = goal_content[0][1:-1].strip()
    find_trigger = goal_content.split()[0]
    if find_trigger == "User" or find_trigger == "用户":
        return False  # not cold start
    elif find_trigger == "Bot":
        return True
    else:
        raise ValueError("The trigger can only be the User or Bot!")


def convert_session_to_sample(session_file, sample_file):
    """
    convert_session_to_sample
    """
    fout = open(sample_file, 'w', encoding="utf-8")
    with open(session_file, 'r', encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                session = json.loads(line.strip(), encoding="utf-8",
                                     object_pairs_hook=collections.OrderedDict)
            except json.decoder.JSONDecodeError:
                continue
            goals = session["goal"].split("-->")
            bot_prior = bot_prior_f(goals[0])
            goals_ft = [goals_format(i) for i in goals]

            if bot_prior:
                n = 0
            else:
                n = 1
            conversation = session["conversation"]
            for j in range(n, len(conversation), 2):
                sample = collections.OrderedDict()
                sample["kg"] = session["knowledge"]
                sample["user_profile"] = session["user_profile"]
                sample["situation"] = session["situation"]
                sample["history"] = conversation[:j]
                sample["response"] = conversation[j]
                sample["goal"] = goals_ft
                sample = json.dumps(sample, ensure_ascii=False)
                fout.write(sample + "\n")

    fout.close()


def main():
    """
    main
    """

    convert_session_to_sample(sys.argv[1],
                              sys.argv[2])


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited from the program ealier!")
