## Image generation setup

### Export Model to OpenVino IR format
- Use optimum-cli to export models to OpenVino IR format.
- Documentation for such operations are available here - https://huggingface.co/docs/optimum/main/intel/openvino/export
- I've created a simple wrapper python script https://github.com/RachithP/tifa/blob/main/ov-quant.py that can be run in lieu of a command.

### Generate Images
- Use the script - https://github.com/RachithP/tifa/blob/main/image_generation/run.py, to generate images.
- Example command: `python run.py --backend ov --device gpu --output_dir <path_to_your_output_dir>"
- Currently the script doesn't accept any tweakable params for the model during inference.

## Evaluation

### TIFA Setup

Depending on the OS, Tifa package installation requirements change.
- On Windows, TIFA requires python 3.8, and pip 24.0
- On Linux (cluster), python=3.9 and pip=24.0 worked. However, numpy and opencv-python versions needed to be tweaked.
  - opencv-contrib-python==4.5.5.62
  - numpy==1.26.4
- Example command to create the python environment: `mamba create -n tifa python=3.8 pip=24.0`

### Run Evaluation

- After generating images using the script `image_generation/run.py` mentioned above, we can run TIFA evaluation on them using https://github.com/RachithP/tifa/blob/main/run_scoring.py.
- Example command: `python run_scoring.py --imgs_file <path_to_imgs.json_generated_during_image_generation> --output_file <path_to_a_json_output_file>`

