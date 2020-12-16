#!/usr/bin/env python

from __future__ import print_function
import json
import collections
import sys


def convert_session_to_sample(session_file, sample_file):
    fout = open(sample_file, 'w', encoding="utf-8")
    with open(session_file, 'r', encoding="utf-8") as f:
        for i, line in enumerate(f):
            session = json.loads(line.strip(), encoding="utf-8", \
                                      object_pairs_hook=collections.OrderedDict)
            try:
                conversation = session["conversation"]
                for j in range(0, len(conversation), 2):
                    sample = collections.OrderedDict()
                    sample["goal"] = session["goal"]
                    sample["knowledge"] = session["knowledge"]
                    sample["history"] = conversation[:j]
                    sample["response"] = conversation[j]
                    sample = json.dumps(sample, ensure_ascii=False)
                    fout.write(sample + "\n")
            except KeyError:
                history = session["history"]
                sample = collections.OrderedDict()
                sample["goal"] = session["goal"]
                sample["knowledge"] = session["knowledge"]
                sample["history"] = history
                sample["response"] = session["response"]
                sample = json.dumps(sample, ensure_ascii=False)
                fout.write(sample + "\n")

    fout.close()


def main():
    """
    main
    """

    convert_session_to_sample(sys.argv[1], sys.argv[2])

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited from the program ealier!")
