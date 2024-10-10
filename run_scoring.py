import argparse
import json
from tifascore import tifa_score_benchmark


parser = argparse.ArgumentParser()
parser.add_argument("--imgs_file", required=True)
args = parser.parse_args()


# generate json 
results = tifa_score_benchmark("mplug-large", "D:/github/tifa/tifa_v1.0/tifa_v1.0_question_answers.json", args.imgs_file)

# save the results
with open("D:/github/tifa/image_generation/sd2/results.json", "w") as f:
    json.dump(results, f, indent=4)
