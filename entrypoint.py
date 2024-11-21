import argparse
import base64
import json
import os
import time
import cv2
import logging

from container_status import ContainerStatus as CS
from run_inference import DeeplabInference
from run_train import DeeplabTraining
from progress_counter import ProgressCounter

run_time = time.time()
logging.basicConfig(level=logging.INFO, filename='/output/deeplab.log', filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Process some images.")

parser.add_argument('--input_data', type=str, help='Path to the input image')
parser.add_argument("--work_format_training", action="store_true", help="Flag for training mode")
parser.add_argument("--demo_mode", action="store_true", help="Flag for demo mode")
parser.add_argument("--host_web", type=str, help="url host with web")
parser.add_argument("--min_width", type=str, help='Path to the input image')
parser.add_argument("--min_height", type=str, help='Path to the input image')

# Парсим аргументы
args = parser.parse_args()

# Получаем значения аргументов

OUTPUT_PATH = "/output"
WORK_FORMAT_TRAINING = args.work_format_training
INPUT_DATA_ARG = args.input_data
MODEL_PATH = "/weights"

DEMO_MODE = args.demo_mode
INPUT_DATA = "/input_data"

TEMP_ROOT = OUTPUT_PATH if DEMO_MODE else "temp_image_root"

IMAGE_TEMP_PATH = f"{TEMP_ROOT}/bboxes"
MASK_TEMP_PATH = f"{TEMP_ROOT}/masks"

cs = CS(args.host_web, logger)
cs.post_start()
# Необязательные параметры на ограничения изображений
if args.min_width:
    min_width = int(args.min_width)
else:
    min_width = 5
if args.min_height:
    min_height = int(args.min_height)
else:
    min_height = 5
if INPUT_DATA_ARG:
    json_input_arg = json.loads(INPUT_DATA_ARG.replace("'", "\""))
    PROCESS_FREQ = int(json_input_arg.get("frame_frequency", 1))
else:
    PROCESS_FREQ = 1


def check_video_extension(video_path):
    valid_extensions = {"avi", "mp4", "m4v", "mov", "mpg", "mpeg", "wmv"}
    ext = video_path.split('.')[-1].lower()
    return ext in valid_extensions


def get_image_name(video, frame, person):
    return f"{video.split('.')[0].split('/')[-1]}_frame_{frame}_person_{person}.png"


def prepare_image_dir(file_path, prepared_data, out_path, mask_out_path=None):
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(mask_out_path, exist_ok=True)
    cap = cv2.VideoCapture(file_path)
    frame_count = 0
    success = True

    while success:
        success, frame = cap.read()

        if not success:
            break

        # Проверяем, есть ли данный кадр в prepared_data
        if frame_count in prepared_data:
            frame_data = prepared_data[frame_count]

            for person_id, bbox in frame_data.items():
                x, y, width, height = bbox['x'], bbox['y'], bbox['width'], bbox['height']
                person_image = frame[y:y + height, x:x + width]

                # Сохранение изображения человека
                image_name = get_image_name(file_path, frame_count, person_id)
                person_image_path = os.path.join(out_path, image_name)

                cv2.imwrite(person_image_path, person_image)
                if WORK_FORMAT_TRAINING:
                    mask = bbox['mask']
                    image_data = base64.b64decode(mask)
                    with open(os.path.join(mask_out_path, image_name), 'wb') as output_file:
                        output_file.write(image_data)
        frame_count += 1
    cap.release()


processed_frames = {"small": 0, 'ok': 0}


def frame_process_condition(num, markup_path):
    if ((PROCESS_FREQ < 2 or int(num) % PROCESS_FREQ == 1) and markup_path['x'] > 0 and markup_path['y'] > 0
            and markup_path['width'] > min_width and markup_path['height'] > min_height):
        if markup_path['width'] < 50 or markup_path['height'] < 100:
            processed_frames["small"] += 1
        else:
            processed_frames['ok'] += 1
        return True


def get_img_str(file_name, mask_dir):
    with open(f'{mask_dir}/{file_name}', 'rb') as image_file:
        # Читаем содержимое изображения
        image_data = image_file.read()
        # Кодируем содержимое в Base64
        base64_encoded_data = base64.b64encode(image_data)
        # Преобразуем закодированные данные в строку
        base64_image_str = base64_encoded_data.decode('utf-8')
    return base64_image_str


def labels_to_dict(labels_file):
    res = dict()
    with open(labels_file, 'r', encoding='utf-8') as labels:
        return json.load(labels)


def prepare_output(input_data, polygons_file, mask_dir):
    labels = labels_to_dict(polygons_file)
    for item in input_data['files']:
        video_path = item['file_name']
        file_chains = item['file_chains']

        for chain in file_chains:
            chain_name = chain['chain_name']
            for frame in chain['chain_markups']:
                # думаю, надо красить каждый 10 фрейм
                frame_num = str(frame['markup_frame'])
                if frame_process_condition(frame_num, frame["markup_path"]):
                    file_name = get_image_name(video_path, frame_num, chain_name)
                    frame['markup_path']['mask'] = get_img_str(file_name, mask_dir)
                    frame['markup_path']['polygons'] = labels[file_name]
    return input_data


def verify_file_name(postfix_name, common_name):
    _, postfix_tail = os.path.split(postfix_name)
    tail1 = '.'.join(postfix_tail.split('.')[0:-1])
    _, tail2 = os.path.split(common_name)
    logger.info(f"comparing {tail1} and {tail2}")
    return tail1 == tail2


files_in_directory = [
    os.path.join(INPUT_DATA, f)
    for f in os.listdir(INPUT_DATA)
    if (
        os.path.isfile(os.path.join(INPUT_DATA, f))
    )
]
files_in_directory = [
    file for file in files_in_directory if check_video_extension('.'.join(file.split('.')[0:-1]))
]

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
        cs.post_progress({"stage": "1 из 2", "progress": 0, "statistics": {"test_error": "No weights file were found"}})
        exit(-1)
    model_file = None

directories = list()
total_images = 0
start_time = time.time()

count = 0

for file in files_in_directory:
    cs.post_progress({"stage": "1 из 2", "progress": round(100*(count/len(files_in_directory)), 2)})
    with open(file, 'r', encoding='utf-8') as input_json:
        json_data = json.load(input_json)

    video_path = None
    _, dir_postfix = os.path.split(file)
    dir_postfix = dir_postfix.split('.')[0]
    im_dir = f'{IMAGE_TEMP_PATH}_{dir_postfix}'
    mask_dir = f'{MASK_TEMP_PATH}_{dir_postfix}'
    for item in json_data['files']:
        prepared_data = dict()
        video_path = item.get('file_name', "no_video")
        if not os.path.isfile(f'/input/{video_path}'):
            logger.warning(f"File name {video_path} doesn't exist - it is skipped")
            continue
        if not verify_file_name(file, video_path):
            logger.warning(f"File name {file} doesn't correspond to file_name key in json {video_path} - it is skipped")
            continue
        # if not check_video_extension(video_path):
        #     continue
        file_chains = item['file_chains']
        for chain in file_chains:
            chain_name = chain['chain_name']
            for frame in chain['chain_markups']:
                frame_num = frame['markup_frame']
                if frame_process_condition(frame_num, frame["markup_path"]):
                    # {"1frame":{"1pers": "bounder box", ...}, "11frame": {...}...}
                    if frame_num not in prepared_data.keys():
                        prepared_data[frame_num] = dict()
                    prepared_data[frame_num][chain_name] = frame["markup_path"]
        if not prepared_data:
            continue
        prepare_image_dir(f'/input/{video_path}', prepared_data, im_dir, mask_dir)
    logger.info(f'Data preparation took {time.time() - start_time} seconds')
    logger.info(f'Amount of small images (with height < 100 or width < 50): {processed_frames["small"]} '
                f'Amount of other images: {processed_frames["ok"]}. Total: {processed_frames["small"] + processed_frames["ok"]}')
    if not video_path:
        logger.warning(f"For {file} correspondent videos")
        continue
    if processed_frames["small"] + processed_frames["ok"] == 0:
        logger.warning(f"For {file} no bounder boxes were found")
        continue
    directories.append((json_data, im_dir, mask_dir, video_path, file, processed_frames["small"] + processed_frames["ok"]))
    total_images += (processed_frames["small"] + processed_frames["ok"])
    processed_frames["small"] = 0
    processed_frames["ok"] = 0

cs.post_progress({"stage": "1 из 2", "progress": 100})
counter = ProgressCounter(total=total_images, processed=0, cs=cs, logger=logger)
deeplab = DeeplabTraining() if WORK_FORMAT_TRAINING else DeeplabInference(model_path=model_file, demo_mode=DEMO_MODE, counter=counter)
cs.post_progress({"stage": "2 из 2", "progress": 0})

for json_data, image_directory, mask_directory, vp, f, frame_amount in directories:
    if WORK_FORMAT_TRAINING:
        start_time = time.time()
        output_weights = f'{OUTPUT_PATH}/deeplab_weights.pt'
        deeplab.run(image_path=image_directory, mask_path=mask_directory, model_path=model_file, output_weights=output_weights)
        # command = f'python run_train.py --image_path {image_directory} --mask_path {mask_directory} --output_weights {output_weights} --host_web {args.host_web}'
        # if model_file:
        #     command += f' --model_path {model_file}'
        # cs.post_progress({"stage": "1 из 2", "progress": 100})
        # os.system(command)
        counter.report_status(report_amount=frame_amount, out_file=output_weights)
        logger.info(f'Deeplab training took {time.time() - start_time} seconds')
        model_file = output_weights
    else:
        start_time = time.time()
        polygon_file = f'{OUTPUT_PATH}/{vp}_labels.txt' if DEMO_MODE else f'{vp}_labels.txt'
        deeplab.run(image_directory=image_directory, mask_directory=mask_directory, polygon_file=polygon_file)
        # command = f'python run_inference.py --image_path {image_directory} --output_path {mask_directory} --model_path {model_file} --host_web {args.host_web} --class_info_path {polygon_file}'
        # if DEMO_MODE:
        #     command += f' --demo_mode'
        # os.system(command)
        logger.info(f'Deeplab inference took {time.time() - start_time} seconds')

        result = prepare_output(json_data, polygon_file, mask_directory)
        with open(f"{OUTPUT_PATH}/{os.path.basename(f)}", "w") as outfile:
            json.dump(result, outfile, ensure_ascii=False)
        counter.report_status(report_amount=frame_amount % 1000, out_file=f"{OUTPUT_PATH}/{os.path.basename(f)}")
cs.post_progress({"stage": "2 из 2", "progress": 100})
cs.post_end()
logger.info(
    f'The whole process took {time.time() - run_time} seconds, work_format_training mode = {WORK_FORMAT_TRAINING}')
