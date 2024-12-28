# An example of how to convert a given API workflow into its own Replicate model
# Replace predict.py with this file when building your own workflow

import os
import mimetypes
import json
import shutil
from typing import List
from cog import BasePredictor, Input, Path
from comfyui import ComfyUI
from cog_model_helpers import optimise_images
from cog_model_helpers import seed as seed_helper

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"
ALL_DIRECTORIES = [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]

mimetypes.add_type("image/webp", ".webp")

# Save your example JSON to the same directory as predict.py
api_json_file = "workflow_api.json"

# Force HF offline
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

class Predictor(BasePredictor):
    def setup(self):
        print("\n=== 开始初始化 ===")
        print("1. 创建ComfyUI实例...")
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        
        print("2. 启动服务器...")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

        print("3. 加载工作流配置...")
        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())
            print(f"工作流节点数量: {len(workflow)}")
        
        print("4. 开始处理权重...")
        self.comfyUI.handle_weights(workflow, weights_to_download=[])

    def filename_with_extension(self, input_file, prefix):
        extension = os.path.splitext(input_file.name)[1]
        return f"{prefix}{extension}"

    def handle_input_file(
        self,
        input_file: Path,
        filename: str = "image.png",
    ):
        shutil.copy(input_file, os.path.join(INPUT_DIR, filename))

    # Update nodes in the JSON workflow to modify your workflow based on the given inputs
    def update_workflow(self, workflow, **kwargs):
        # Below is an example showing how to get the node you need and update the inputs

        # positive_prompt = workflow["6"]["inputs"]
        # positive_prompt["text"] = kwargs["prompt"]

        # negative_prompt = workflow["7"]["inputs"]
        # negative_prompt["text"] = f"nsfw, {kwargs['negative_prompt']}"

        # sampler = workflow["3"]["inputs"]
        # sampler["seed"] = kwargs["seed"]
        image = workflow["12"]["inputs"]
        image["image"] = kwargs["image_filename"]
        pass

    def predict(
        self,
        image: Path = Input(
            description="An input image",
            default=None,
        ),
        output_format: str = optimise_images.predict_output_format(),
        output_quality: int = optimise_images.predict_output_quality(),
        seed: int = seed_helper.predict_seed(),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        self.comfyUI.cleanup(ALL_DIRECTORIES)

        # Make sure to set the seeds in your workflow
        seed = seed_helper.generate(seed)

        image_filename = self.filename_with_extension(image, "image")
        self.handle_input_file(image, image_filename)

        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())

        self.update_workflow(
            workflow,
            image_filename=image_filename,
            seed=seed,
        )

        wf = self.comfyUI.load_workflow(workflow)
        self.comfyUI.connect()
        self.comfyUI.run_workflow(wf)

        return optimise_images.optimise_image_files(
            output_format, output_quality, self.comfyUI.get_files(OUTPUT_DIR)
        )
