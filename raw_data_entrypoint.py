import argparse
import logging
import os

from run_inference import DeeplabInference
from run_train import DeeplabTraining


logging.basicConfig(level=logging.INFO, filename='/output/deeplab.log', filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Process some images.")

parser.add_argument("--work_format_training", action="store_true", help="Flag for training mode")
parser.add_argument("--demo_mode", action="store_true", help="Flag for demo mode")

# Парсим аргументы
args = parser.parse_args()
WORK_FORMAT_TRAINING = args.work_format_training
DEMO_MODE = args.demo_mode
IMAGE_PATH = '/images'
MASK_PATH = '/masks'
MODEL_PATH = "/weights"
OUTPUT_PATH = '/output'

files_in_weights = [
    os.path.join(MODEL_PATH, f)
    for f in os.listdir(MODEL_PATH)
    if (
        f.split('.')[-1] == 'pt'
    )
]
if files_in_weights:
    model_file = files_in_weights[0]
else:
    if not WORK_FORMAT_TRAINING:
        logger.error("No weights file were found")
        exit(-1)

    model_file = None

if WORK_FORMAT_TRAINING:
    deeplab = DeeplabTraining()
    deeplab.run(image_path=IMAGE_PATH, mask_path=MASK_PATH, model_path=model_file, output_weights=f'{OUTPUT_PATH}/deeplab_weights.pt')
else:
    deeplab = DeeplabInference(model_path=model_file, demo_mode=DEMO_MODE, counter=None)
    deeplab.run(image_directory=IMAGE_PATH, mask_directory=f'/{OUTPUT_PATH}/{MASK_PATH}', polygon_file=f'{OUTPUT_PATH}/polygons.txt')
