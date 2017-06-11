import argparse
import os
import subprocess

import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input_file")
    parser.add_argument("output_file")
    parser.add_argument("model_dir")

    args = parser.parse_args()

    # check bio file
    if not os.path.exists(args.input_file):
        raise IOError("input file not found.")
    # check model dir
    if not os.path.exists(args.model_dir) or \
            not os.path.isdir(args.model_dir):
        raise IOError("model dir not found")

    # validate bio
    utils.bio_validator(args.input_file)

    # executing tagger
    tagger_exe = os.path.join(os.path.dirname(os.path.abspath("__file__")),
                              "../bin/seqLabelingTagger")
    cmd = [
        tagger_exe,
        args.input_file,
        args.output_file,
        args.model_dir
    ]
    print('=>executing tagger...')
    print(' '.join(cmd))

    subprocess.call(cmd)




