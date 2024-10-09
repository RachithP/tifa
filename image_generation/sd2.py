import argparse
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from optimum.intel import OVStableDiffusionPipeline
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

    if args.backend == "ov":
        model_id = args.model
        pipe = OVStableDiffusionPipeline.from_pretrained(model_id, compile=False).to(args.device)
    elif args.backend == "pt":
        model_id = args.model
        pipe = StableDiffusionPipeline.from_pretrained(model_id, cache_dir=args.cache_dir, torch_dtype=torch.float16, compile=False)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        if args.device == "gpu":
            pipe = pipe.to("cuda")
    else:
        assert False

    for text_input in text_inputs:
        id = text_input["id"]
        prompt = text_input["caption"]
        image = pipe(prompt).images[0]
        image.save(os.path.join(args.output_dir, f"{id}.png"))
