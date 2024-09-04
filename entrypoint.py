import argparse
import base64
import json
import os
import time
import cv2
import logging
from container_status import ContainerStatus as CS

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

output_path = "/output"
work_format_training = args.work_format_training
model_path = "/models/deeplab_weights.pt"
with open(os.path.join('/input', args.input_data), 'r', encoding='utf-8') as input_json:
    input_data = json.load(input_json)
demo_mode = args.demo_mode

temp_root = "/output" if demo_mode else "temp_image_root"

image_temp_path = f"{temp_root}/bboxes"
mask_temp_path = f"{temp_root}/masks"

cs = CS(args.host_web)

# Необязательные параметры на ограничения изображений
if args.min_width:
    min_width = int(args.min_width)
else:
    min_width = 5
if args.min_height:
    min_height = int(args.min_height)
else:
    min_height = 5
PROCESS_FREQ = 10


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
        frame_count += 1

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
                if work_format_training:
                    mask = bbox['mask']
                    image_data = base64.b64decode(mask)
                    with open(os.path.join(mask_out_path, image_name), 'wb') as output_file:
                        output_file.write(image_data)

    cap.release()

processed_frames = {"small": 0, 'ok': 0}

def frame_process_condition(num, markup_path):
    if (int(num) % PROCESS_FREQ == 1 and markup_path['x'] > 0 and markup_path['y'] > 0
            and markup_path['width'] > min_width and markup_path['height'] > min_height):
        if markup_path['width'] < 50 or markup_path['height'] < 100:
            processed_frames["small"] += 1
        else:
            processed_frames['ok'] += 1
        return True

cs.post_start()
start_time = time.time()
for item in input_data['files']:
    prepared_data = dict()
    video_path = item['file_name']
    file_chains = item['file_chains']
    for chain in file_chains:
        chain_id = chain['chain_id']
        for frame in chain['chain_markups']:
            frame_num = frame['markup_frame']
            if frame_process_condition(frame_num, frame["markup_path"]):
                # {"1frame":{"1pers": "bounder box", ...}, "11frame": {...}...}
                if frame_num not in prepared_data.keys():
                    prepared_data[frame_num] = dict()
                prepared_data[frame_num][chain_id] = frame["markup_path"]
    prepare_image_dir(f'/input/{video_path}', prepared_data, image_temp_path, mask_temp_path)
logger.info(f'Data preparation took {time.time() - start_time} seconds')
logger.info(f'Amount of small images (with height < 100 or width < 50): {processed_frames["small"]} '
            f'Amount of other images: {processed_frames["ok"]}. Total: {processed_frames["small"] + processed_frames["ok"]}')


def get_img_str(file_name):
    with open(f'{mask_temp_path}/{file_name}', 'rb') as image_file:
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
        for line in labels:
            # Удалить пробельные символы в начале и в конце строки и разбить строку по пробелам
            parts = line.strip().split()

            # Первая часть - это название файла, а оставшиеся части - метки классов
            image_name = parts[0]
            class_labels = " ".join(parts[1:])

            # Записать данные в словарь
            res[image_name] = class_labels
    return res


def prepare_output():
    labels = labels_to_dict(f'{output_path}/labels.txt')
    for item in input_data['files']:
        video_path = item['file_name']
        file_chains = item['file_chains']

        for chain in file_chains:
            chain_id = chain['chain_id']
            for frame in chain['chain_markups']:
                # думаю, надо красить каждый 10 фрейм
                frame_num = str(frame['markup_frame'])
                if frame_process_condition(frame_num, frame["markup_path"]):
                    file_name = get_image_name(video_path, frame_num, chain_id)
                    frame['markup_path']['mask'] = get_img_str(file_name)
                    frame['markup_path']['labels'] = labels[file_name]
    return input_data


if work_format_training:
    start_time = time.time()
    output_weights = f'{output_path}/deeplab_weights.pt'
    command = f'python run_train.py --image_path {image_temp_path} --mask_path {mask_temp_path} --output_weights {output_weights}'
    if model_path:
        command += f' --model_path {model_path}'
    os.system(command)
    logger.info(f'Deeplab training took {time.time() - start_time} seconds')
else:
    start_time = time.time()
    command = f'python run_inference.py --image_path {image_temp_path} --output_path {mask_temp_path} --model_path {model_path}'
    if model_path:
        command += f' --model_path {model_path}'
    command += f' --class_info_path {output_path}/labels.txt'
    if demo_mode:
        command += f' --demo_mode'
    os.system(command)
    logger.info(f'Deeplab inference took {time.time() - start_time} seconds')

if not work_format_training:
    result = prepare_output()
    with open(f"{output_path}/deeplab.json", "w") as outfile:
        json.dump(result, outfile, ensure_ascii=False)
logger.info(f'The whole process took {time.time() - run_time} seconds, work_format_training mode = {work_format_training}')