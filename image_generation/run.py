import argparse
from datetime import datetime
from diffusers import DPMSolverMultistepScheduler
import json
import os
import torch


def parse_text_inputs(text_inputs_file):
    import json
    assert os.path.exists(text_inputs_file)

    with open(text_inputs_file, 'r') as f:
        data = json.load(f)
    return data


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="ov")
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", default="stabilityai/stable-diffusion-2-1")
    parser.add_argument("--cache_dir")
    args = parser.parse_args()

    if args.cache_dir:
        assert os.path.exists(args.cache_dir)
    
    assert args.device in ["cpu", "gpu", "npu"]
    os.makedirs(args.output_dir, exist_ok=True)

    text_inputs_file = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "tifa_v1.0", "tifa_v1.0_text_inputs.json"))
    text_inputs = parse_text_inputs(text_inputs_file)

    print("Loading the models and setting up parameters at ", datetime.now())

    if args.backend == "ov":
        from optimum.intel import OVStableDiffusionPipeline
        model_id = args.model
        pipe = OVStableDiffusionPipeline.from_pretrained(model_id, compile=False).to(args.device)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    elif args.backend == "pt":
        assert args.cache_dir
        from diffusers import StableDiffusionPipeline
        model_id = args.model
        pipe = StableDiffusionPipeline.from_pretrained(model_id, cache_dir=args.cache_dir, torch_dtype=torch.float16, compile=False)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        if args.device == "gpu":
            pipe = pipe.to("cuda")
    else:
        assert False

    print("Loaded the models, starting generation now ", datetime.now())

    # if run stopped in-between, to reduce redundancy disregard images already generated
    files = os.listdir(args.output_dir)

    id_to_img_paths = {}
    for text_input in text_inputs:
        id = text_input["id"]
        prompt = text_input["caption"]
        image_file_name = f"{id}.png"
        image_file_path = os.path.join(args.output_dir, image_file_name)

        if image_file_name in files:
            print(f"{id}.png exists, so skipping it...")
            continue

        image = pipe(prompt).images[0]
        image.save(image_file_path)

        id_to_img_paths[id] = image_file_path

    # store the id_to_imgs dict on disk
    # this is required by tifa_scoring
    imgs_json = os.path.join(args.output_dir, "imgs.json")
    with open(imgs_json, 'a') as f:
        json.dump(id_to_img_paths, f)

    print("Succesfully wrote unique id to corresponding image path values to the file ", imgs_json)
    print("Completed run at ", datetime.now())
