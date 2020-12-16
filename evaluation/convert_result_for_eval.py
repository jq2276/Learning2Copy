#!/usr/bin/env python

import sys
import json
import collections


def convert_result_for_eval(sample_file, result_file, output_file):
    """
    convert_result_for_eval
    """
    sample_list = [line.strip() for line in open(sample_file, 'r', encoding="utf-8")]
    result_list = [line.strip() for line in open(result_file, 'r', encoding="utf-8")]

    assert len(sample_list) == len(result_list), "The sample_list length must be equal to the result_list length"
    fout = open(output_file, 'w', encoding="utf-8")
    for i, sample in enumerate(sample_list):
        sample = json.loads(sample, encoding="utf-8", object_pairs_hook=collections.OrderedDict)
        response = sample["response"]
        fout.write(result_list[i] + "\t" + response + "\n")

    fout.close()


def main():
    """
    main
    """
    convert_result_for_eval(sys.argv[1],
                            sys.argv[2],
                            sys.argv[3])


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited from the program ealier!")
