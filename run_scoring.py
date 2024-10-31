import argparse
import json
import os
from tifascore import tifa_score_benchmark


parser = argparse.ArgumentParser()
parser.add_argument("--imgs_file", required=True)
parser.add_argument("--output_file", required=True)
parser.add_argument("--element_type")
args = parser.parse_args()

assert os.path.exists(args.imgs_file)

results = tifa_score_benchmark("mplug-large", "/home/rachithp/code/tifa/tifa_v1.0/tifa_v1.0_question_answers.json", args.imgs_file, args.element_type)

# save the results
with open(args.output_file, "w") as f:
    json.dump(results, f, indent=4)
