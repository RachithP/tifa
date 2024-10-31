import argparse
import logging
import os
import subprocess


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--log_file_name", required=True)
    parser.add_argument("--model", default="stabilityai/stable-diffusion-2-1")
    parser.add_argument("--weight_format", required=True)
    parser.add_argument("--cache_dir")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cmd = [
        "optimum-cli",
        "export",
        "openvino",
        "--model",
        args.model,
        "--weight-format",
        args.weight_format,
        args.output_dir]

    log.info(f"Executing: {' '.join(cmd)}")

    with open(os.path.join(args.output_dir, f"{args.log_file_name}.txt"), "w") as f:
        subprocess.run(cmd, check=True, stdout=f, stderr=f)
