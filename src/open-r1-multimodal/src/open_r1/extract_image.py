import requests
import time
import argparse
import json
import base64
import os
from tqdm import tqdm
from PIL import Image


if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--result_directory", type=str, required=True)
    args = args.parse_args()

    EACs = []

    results_directory = args.result_directory
    with open(os.path.join(results_directory, "results.json"), "r", encoding='utf-8') as f:
        results = json.load(f)

    os.makedirs(os.path.join(results_directory, "input"), exist_ok=True)
    os.makedirs(os.path.join(results_directory, "generated"), exist_ok=True)
    os.makedirs(os.path.join(results_directory, "gt"), exist_ok=True)

    for result in tqdm(results):
        combined_image = Image.open(os.path.join(results_directory, f"sample_{result['sample_id']:04d}.png"))

        combined_image.crop((0, 0, 512, 512)).save(os.path.join(results_directory, "input", f"sample_{result['sample_id']:04d}.png"))
        combined_image.crop((512, 0, 1024, 512)).save(os.path.join(results_directory, "generated", f"sample_{result['sample_id']:04d}.png"))
        combined_image.crop((1024, 0, 1536, 512)).save(os.path.join(results_directory, "gt", f"sample_{result['sample_id']:04d}.png"))