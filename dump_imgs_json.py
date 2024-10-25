from tifascore import tifa_score_benchmark
import json
import os
import posixpath

def parse_text_inputs(text_inputs_file):

    assert os.path.exists(text_inputs_file)

    with open(text_inputs_file, 'r') as f:
        data = json.load(f)

    return data

folder_name = "sd21_base_fp16_ov"

imgs_path = posixpath.join("D:/github/tifa/image_generation", folder_name)
text_inputs_file = os.path.realpath(os.path.join(os.path.dirname(__file__), "tifa_v1.0", "tifa_v1.0_text_inputs.json"))

data = parse_text_inputs(text_inputs_file)

imgs = {}
for entry in data:
    imgs[entry["id"]] = posixpath.join(imgs_path, f'{entry["id"]}.png')

imgs_json = posixpath.join("D:/github/tifa/image_generation", folder_name, "imgs.json")

with open(imgs_json, 'w') as f:
    json.dump(imgs, f)
