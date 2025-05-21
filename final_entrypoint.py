import argparse
import base64
import json
import os
import pickle
import time
import cv2
import logging
import numpy as np
from PIL import Image

from container_status import ContainerStatus as CS
from run_inference import DeeplabInference
from run_train import DeeplabTraining
from progress_counter import ProgressCounter
import visualization

color_mapping = {
    0: (0, 0, 0),  # Background
    1: (128, 0, 0),  # Jacket/Coat| "Shirt/Blouse", "vest"
    2: (0, 128, 0),  # Shirt/Blouse| "top, t-shirt, sweatshirt", "Sweater"
    3: (128, 128, 0),  # Sweater/Sweatshirt/Hoodie| "cardigan", "jacket", "coat", "cape"
    4: (0, 0, 128),  # Dress/Romper| "pants"
    5: (128, 0, 128),  # Pants/Jeans/Leggings| "shorts", "skirt"
    6: (0, 128, 128),  # Shorts| "dress", "jumpsuit"
    7: (128, 128, 128),  # Skirt| "shoe"
    8: (64, 0, 0),  # Shoes| "bag, wallet", "umbrella", "hat", "headband, head covering, hair accessory"
    # 9: (192, 0, 0),  # Vest|
    # 10: (64, 128, 0),  # Boots
    # 11: (192, 128, 0),  # Bodysuit/T-shirt/Top
    # 12: (64, 0, 128),  # Bag/Purse
    # 13: (192, 0, 128),  # Hat
    # 14: (64, 128, 128),  # Scarf/Tie
    # 15: (192, 128, 128),  # Gloves
    # 16: (0, 64, 0),  # Blazer/Suit
    # 17: (128, 64, 0),  # Underwear/Swim
}

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

WORK_FORMAT_TRAINING = args.work_format_training
INPUT_DATA_ARG = args.input_data
MODEL_PATH = "/weights"

DEMO_MODE = args.demo_mode

#volumes
INPUT_DATA = "/input_data"
INPUT = "/input_videos"
OUTPUT_PATH = "/output"

TEMP_ROOT = OUTPUT_PATH if DEMO_MODE else "temp_image_root"
STAGE1, STAGE2, STAGE3 = "Подготовка данных", "Обучение", "Обработка"

IMAGE_TEMP_PATH = f"{TEMP_ROOT}/bboxes"
MASK_TEMP_PATH = f"{TEMP_ROOT}/masks"

cs = CS(args.host_web, logger)
cs.post_start()
MAX_STAGE = 3 if WORK_FORMAT_TRAINING else 2
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
    try:
        json_input_arg = json.loads(INPUT_DATA_ARG.replace("'", "\""))
    except json.decoder.JSONDecodeError:
        cs.post_error({"msg": "Wrong input_data argument format", "details": f"input_data is following {INPUT_DATA_ARG}"})
        logger.info(f"Wrong input_data argument format {INPUT_DATA_ARG}")
        exit(-1)
    PROCESS_FREQ = int(json_input_arg.get("frame_frequency", 1))
    WEIGHTS_FILENAME = json_input_arg.get("weights", None)
    h, w = int(json_input_arg.get("tr_h", 224)), int(json_input_arg.get("tr_w", 224))
    epoches = int(json_input_arg.get("epoches", 50))
    pixel_hist_step = json_input_arg.get("pixel_hist_step", None)
    pixel_hist_step = float(pixel_hist_step) if pixel_hist_step else None

    SOLO_FILE = eval(json_input_arg.get("solo_file", "False"))
    MAKE_VISUALIZATION = eval(json_input_arg.get("visualize", "False"))
    EVAL_MODE = eval(json_input_arg.get("eval_mode", "False"))

    approx_eps = json_input_arg.get("approx_eps", None)
    approx_eps = float(approx_eps) if approx_eps else None

    if not WEIGHTS_FILENAME:
        if not WORK_FORMAT_TRAINING:
            logger.error("No weights file were specified for inference")
            cs.post_progress(
                {"stage": STAGE1, "progress": 0,
                 "statistics": {"test_error": "No weights file were specified for inference"}})
            exit(-1)
        model_file = f'{MODEL_PATH}/deeplab_weights.pt'
    elif not os.path.isfile(f'{MODEL_PATH}/{WEIGHTS_FILENAME}'):
        logger.error("No weights file were found")
        cs.post_progress(
            {"stage": STAGE1, "progress": 0, "statistics": {"test_error": "No weights file were found"}})
        exit(-1)
    else:
        model_file = f'{MODEL_PATH}/{WEIGHTS_FILENAME}'
else:
    if not WORK_FORMAT_TRAINING:
        logger.error("No weights file were specified for inference")
        cs.post_progress(
            {"stage": STAGE1, "progress": 0,
             "statistics": {"test_error": "No weights file were specified for inference"}})
        exit(-1)
    PROCESS_FREQ = 1
    model_file = f'{MODEL_PATH}/deeplab_weights.pt'
    h, w = 224, 224
    epoches = 50
    pixel_hist_step = None
    approx_eps = None
    SOLO_FILE = False
    MAKE_VISUALIZATION = False
    EVAL_MODE = False


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
                x, y, width, height = int(bbox['x']), int(bbox['y']), int(bbox['width']), int(bbox['height'])
                person_image = frame[y:y + height, x:x + width]

                # Сохранение изображения человека
                image_name = get_image_name(file_path, frame_count, person_id)
                person_image_path = os.path.join(out_path, image_name)

                cv2.imwrite(person_image_path, person_image)
                if WORK_FORMAT_TRAINING:
                    polygons = bbox['polygons']
                    mask = np.zeros((height, width), dtype=np.uint8)
                    # logger.info(f"shape = {mask.shape} width height = {(width, height)}")
                    for class_id, poly_list in polygons.items():
                        # logger.info(f'for {image_name} class_id = {class_id}, poly_list = {poly_list}')
                        for poly in poly_list:
                            # Преобразуем список координат в массив точек формы (-1, 1, 2)
                            poly_shifted = [(px - x, py - y) for px, py in zip(poly[::2], poly[1::2])]
                            points = np.array(poly_shifted, dtype=np.int32).reshape(-1, 2)
                            # points = points[:, [1, 0]]
                            # Заполняем область полигона на маске
                            cv2.fillPoly(mask, [points], int(class_id))

                    # Сохраняем маску, восстановленную по полигонам
                    polygonized_pred_img = Image.fromarray(mask)
                    # polygonized_pred_img = polygonized_pred_img.resize((width, height), resample=Image.NEAREST)
                    polygonized_pred_img.save(os.path.join(mask_out_path, image_name))
                    # cv2.imwrite(os.path.join(mask_out_path, image_name), mask)

                    # with open(os.path.join(mask_out_path, image_name), 'wb') as output_file:
                    #     output_file.write(image_data)
        frame_count += 1
    cap.release()


processed_frames = {"small": 0, 'ok': 0}


def frame_process_condition(num, markup_path):
    if num == True:
        return True
    if ((PROCESS_FREQ < 2 or int(num) % PROCESS_FREQ == 1) and markup_path['x'] > 0 and markup_path['y'] > 0
            and markup_path['width'] > min_width and markup_path['height'] > min_height):
        if markup_path['width'] < 224 or markup_path['height'] < 224:
            processed_frames["small"] += 1
        else:
            processed_frames['ok'] += 1
        return True


def labels_to_dict(labels_file):
    res = dict()
    with open(labels_file, 'r', encoding='utf-8') as labels:
        return json.load(labels)


def resize_polygons(polygons, orig_width, orig_height, orig_x, orig_y):
    """
    Масштабирует полигоны обратно к размерам исходного баундер-бокса.

    :param polygons: Список полигонов, каждый из которых представлен списком координат [x1, y1, x2, y2, ...].
    :param orig_width: Исходная ширина баундер-бокса.
    :param orig_height: Исходная высота баундер-бокса.
    :param target_size: Размер, к которому были приведены полигоны (по умолчанию 224x224).
    :return: Список масштабированных полигонов.
    """
    scale_x = float(orig_width) / float(w)
    scale_y = float(orig_height) / float(h)

    resized_polygons = []
    for polygon in polygons:
        resized_polygon = []
        for i in range(0, len(polygon), 2):
            x = int(orig_x) + int(polygon[i] * scale_x)
            y = int(orig_y) + int(polygon[i + 1] * scale_y)
            resized_polygon.extend([x, y])
        resized_polygons.append(resized_polygon)

    return resized_polygons


def prepare_output(input_data, polygons_file=None):
    if not polygons_file:
        input_data.pop('datasets', None)
        for item in input_data['files']:
            item.pop('file_subset', None)
            file_chains = item.get('file_chains', None)
            if file_chains:
                for chain in file_chains:
                    chain.pop('chain_id', None)
                    chain.pop('chain_dataset_id', None)
                    chain.pop('chain_markups', None)
        return input_data

    vector_list = list()
    vector_chain_list = list()
    output_data = {'files': []}
    labels = labels_to_dict(polygons_file)
    input_data.pop('datasets', None)
    for item in input_data['files']:
        video_path = item['file_name']
        file_idx = len(output_data['files'])
        output_data['files'].append({'file_name': video_path, 'file_id': item['file_id'], 'file_chains': list()})
        output_chains, temp_vect_dict, scores = dict(), dict(), dict()
        file_chains = item['file_chains']
        for chain in file_chains:
            chain_name = chain['chain_name']
            chain_markups = chain.pop('chain_markups', None)
            # if not SOLO_FILE:
            #     vector_chain_list.append()
            if not chain_markups:
                logger.warning(f'No chain_markups for chain {chain_name}')
                continue
            chain['chain_markups'] = list()
            for frame in chain_markups:
                frame_num = str(frame['markup_frame'])
                file_name = get_image_name(video_path, frame_num, chain_name)
                if frame_process_condition(frame_num, frame['markup_path']):
                    for cls, p in labels[file_name].items():
                        resized_polygons = resize_polygons(p['polygons'], frame['markup_path']['width'],
                                                           frame['markup_path']['height'], frame['markup_path']['x'],
                                                           frame['markup_path']['y'])
                        vct = np.array(p['markup_vector'])
                        # logger.info(vct)
                        vector_list.append(vct)
                        if not output_chains.get(f'{chain_name}_{cls}', None):
                            output_chains[f'{chain_name}_{cls}'] = {'chain_vector': None, 'chain_confidence': 0,
                                                                    'chain_markups': list()}
                            temp_vect_dict[f'{chain_name}_{cls}'], scores[f'{chain_name}_{cls}'] = list(), list()
                        output_chains[f'{chain_name}_{cls}']['chain_markups'].append({
                            'markup_parent_id': frame['markup_id'],
                            'markup_frame': frame['markup_frame'],
                            'markup_time': frame['markup_time'],
                            'markup_vector': len(vector_list) - 1,
                            'markup_confidence': round(p['score'], 6),
                            'markup_path': {'class': cls, 'x': frame['markup_path']['x'],
                                            'y': frame['markup_path']['y'], 'width': frame['markup_path']['width'],
                                            'height': frame['markup_path']['height'], 'polygons': resized_polygons}
                        })
                        temp_vect_dict[f'{chain_name}_{cls}'].append(vct)
                        scores[f'{chain_name}_{cls}'].append(p['score'])
            # logger.info(np.mean(np.array(temp_vect_list), axis=0))

        for k, v in temp_vect_dict.items():
            vector_chain_list.append(np.mean(np.array(v), axis=0))
            output_chains[k]['chain_vector'] = len(vector_chain_list) - 1
            score_val = scores[k]
            sum = 0
            for val in score_val:
                sum += val
            output_chains[k]['chain_confidence'] = 0 if len(score_val) == 0 else round(sum / len(score_val), 6)
        for chain_name, chain_val in output_chains.items():
            output_data['files'][file_idx]['file_chains'].append(
                {'chain_name': chain_name, 'chain_vector': chain_val['chain_vector'],
                 'chain_confidence': chain_val['chain_confidence'], 'chain_markups': chain_val['chain_markups']})

    return output_data, vector_chain_list, vector_list


def verify_file_name(postfix_name, common_name):
    _, postfix_tail = os.path.split(postfix_name)
    tail1 = '.'.join(postfix_tail.split('.')[0:-1])
    _, tail2 = os.path.split(common_name)
    logger.info(f"comparing {tail1} and {tail2}")
    return tail1 == f'IN_{tail2}'


#Temparal for training format
def verify_additional_file_name(postfix_name, common_name):
    _, postfix_tail = os.path.split(postfix_name)
    tail1 = '.'.join(postfix_tail.split('.')[0:-1])
    _, tail2 = os.path.split(common_name)
    logger.info(f"comparing {tail1} and {tail2}")
    return tail1 == f'{tail2}'


# For now the method is used to get markup_path from bytetracker results
def get_bbox_data(frame_data, id_bbox):
    if not id_bbox:
        raise Exception("No bbox_data is found")
    return id_bbox[frame_data['markup_parent_id']]


def prepare_bbox_info(json_file):
    bbox_res = dict()
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for i in data['files']:
        chains = i['file_chains']
        for ch in chains:
            for fr in ch['chain_markups']:
                bbox_res[fr['markup_id']] = fr['markup_path']
    return bbox_res


def delete_files_in_directory(directory_path):
    try:
        files = os.listdir(directory_path)
        for file in files:
            file_path = os.path.join(directory_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print("All files deleted successfully.")
    except OSError:
        print("Error occurred while deleting files.")


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

directories = list()
total_images = 0
start_time = time.time()
json_data = None

count = 0


def save_empty_file(data, f):
    result = prepare_output(data)
    spl = os.path.basename(f).split('_')
    output_file_name = '_'.join(spl[1:len(spl)]) if os.path.basename(f).startswith('IN_') else os.path.basename(
        f)
    spl = output_file_name.split('.')
    outp_without_ext = '.'.join(spl[0:len(spl) - 1])
    output_file_name = "OUT_" + output_file_name
    result['files'][0]['file_name'] = outp_without_ext

    with open(f"{OUTPUT_PATH}/{output_file_name}", "w") as outfile:
        json.dump(result, outfile, ensure_ascii=False)

    empty_vectors, empty_vector_chains = list(), list()

    with open(f"{OUTPUT_PATH}/{outp_without_ext}_chains_vectors.pkl", "wb") as pickle_file:
        pickle.dump(empty_vector_chains, pickle_file)
    with open(f"{OUTPUT_PATH}/{outp_without_ext}_markups_vectors.pkl", "wb") as pickle_file:
        pickle.dump(empty_vectors, pickle_file)
    cs.post_progress(
        {"stage": STAGE3, "progress": 100,
         "statistics": {"out_file": output_file_name, "chains_count": 0, "markups_count": 0}})

#Only needed for training mode to get bboxes
bbox_info, empty_files = None, list()
for file in files_in_directory:
    cs.post_progress({"stage": STAGE1, "progress": round(100 * (count / len(files_in_directory)), 2)})
    with open(file, 'r', encoding='utf-8') as input_json:
        try:
            json_data = json.load(input_json)
        except json.decoder.JSONDecodeError:
            cs.post_error({"msg": "Error when loading json file", "details": f"File: {file}"})
            logger.info(f"Error when loading json file {file}")
            continue

    video_path = None
    _, dir_postfix = os.path.split(file)
    dir_postfix = dir_postfix.split('.')[0]
    im_dir = f'{IMAGE_TEMP_PATH}_{dir_postfix}'
    mask_dir = f'{MASK_TEMP_PATH}_{dir_postfix}'

    for item in json_data['files']:
        prepared_data = dict()
        prepared_data_train = dict()
        video_path = item.get('file_name', 'no_video')
        _, video_path = os.path.split(video_path)
        if not (os.path.isfile(f'{INPUT}/{video_path}') or os.path.islink(f'{INPUT}/{video_path}')):
            cs.post_error({"msg": "Video file hasn't been found", "details": f"File name: {INPUT}/{video_path}"})
            logger.warning(f"File name {INPUT}/{video_path} doesn't exist - it is skipped")
            break
        if not verify_file_name(file, video_path):
            cs.post_error({"msg": "Invalid file name",
                           "details": f"File name {file} doesn't correspond to file_name key in json {INPUT}/{video_path}"})
            logger.warning(f"File name {file} doesn't correspond to file_name key in json {video_path}")
            break
        # if not check_video_extension(video_path):
        #     continue

        # To convert time to frame_num
        cap = cv2.VideoCapture(f'{INPUT}/{video_path}')
        fps = cap.get(cv2.CAP_PROP_FPS)

        file_chains = item.get('file_chains', None)
        if not file_chains:
            # cs.post_error({"msg": "file_chains key has not been found",
            #                "details": f"File name {file}"})
            continue
        for chain in file_chains:
            chain_name = chain.get('chain_name', None)
            if chain_name is None:
                cs.post_error({"msg": "Chain name is not specified",
                               "details": f"File name {file}"})
                continue
            if WORK_FORMAT_TRAINING:
                for frame in chain['chain_markups']:
                    frame_num = frame.get('markup_frame', None)
                    if frame_num is None:
                        m_time = frame.get('markup_time', None)
                        if m_time is None:
                            cs.post_error({"msg": "Nor markup_frame, nor markup_time are set",
                                           "details": f"File name {file}, chain_name {chain_name}"})
                            continue
                        frame['markup_frame'] = round(float(frame['markup_time']) * float(fps))
                        frame_num = frame['markup_frame']
                    frame["markup_path"]["x"], frame["markup_path"]["y"] = round(frame["markup_path"]["x"]), round(
                        frame["markup_path"]["y"])
                    frame["markup_path"]["width"], frame["markup_path"]["height"] = round(
                        frame["markup_path"]["width"]), round(frame["markup_path"]["height"])

                    bbox_data = {"x": frame["markup_path"]["x"], "y": frame["markup_path"]["y"],
                                 "width": frame["markup_path"]["width"], "height": frame["markup_path"]["height"]}
                    if frame_process_condition(frame_num, bbox_data):
                        if frame_num not in prepared_data_train.keys():
                            prepared_data_train[frame_num] = dict()
                        if chain_name not in prepared_data_train[frame_num].keys():
                            prepared_data_train[frame_num][chain_name] = bbox_data
                            prepared_data_train[frame_num][chain_name]['polygons'] = dict()
                        prepared_data_train[frame_num][chain_name]['polygons'][frame['markup_path']['class']] = \
                            frame["markup_path"]['polygons']

            else:
                for frame in chain['chain_markups']:
                    frame_num = frame.get('markup_frame', None)
                    if frame_num is None:
                        m_time = frame.get('markup_time', None)
                        if m_time is None:
                            cs.post_error({"msg": "Nor markup_frame, nor markup_time are set",
                                           "details": f"File name {file}, chain_name {chain_name}"})
                            continue
                        frame['markup_frame'] = round(float(frame['markup_time']) * float(fps))
                        frame_num = frame['markup_frame']
                    frame_time = frame['markup_time']
                    bbox_data = frame['markup_path']
                    if frame_process_condition(frame_num, bbox_data):
                        if frame_num not in prepared_data.keys():
                            prepared_data[frame_num] = dict()
                        prepared_data[frame_num][chain_name] = bbox_data
        cap.release()
    prepared_data = prepared_data_train if WORK_FORMAT_TRAINING else prepared_data
    if not prepared_data:
        empty_files.append((json_data, file))
        continue
    prepare_image_dir(f'{INPUT}/{video_path}', prepared_data, im_dir, mask_dir)
    logger.info(f'Data preparation took {time.time() - start_time} seconds')
    logger.info(f'Total images: {processed_frames["small"] + processed_frames["ok"]}')
    if not video_path:
        logger.warning(f"For {file} correspondent videos")
        empty_files.append((json_data, file))
        continue
    if processed_frames["small"] + processed_frames["ok"] == 0:
        logger.warning(f"For {file} no bounder boxes were found")
        empty_files.append((json_data, file))
        continue
    directories.append(
        (json_data, im_dir, mask_dir, video_path, file, processed_frames["small"] + processed_frames["ok"]))
    total_images += (processed_frames["small"] + processed_frames["ok"])
    processed_frames["small"] = 0
    processed_frames["ok"] = 0
    count += 1

cs.post_progress({"stage": STAGE1, "progress": 100})
if not directories:
    logger.info("There no data to process")
    for params in empty_files:
        save_empty_file(*params)
    cs.post_end()
    exit(0)
if WORK_FORMAT_TRAINING:
    cs.post_progress({"stage": STAGE2, "progress": 0})
    counter = ProgressCounter(total=total_images, processed=0, cs=cs, logger=logger, stage="Обучение")
    for json_data, image_directory, mask_directory, vp, f, frame_amount in directories:
        deeplab = DeeplabTraining()
        start_time = time.time()
        deeplab.run(image_path=image_directory, mask_path=mask_directory, model_path=model_file,
                    output_weights=model_file, h=h, w=w, epoches=epoches, eval_mode=EVAL_MODE)
        counter.report_status(report_amount=frame_amount)
        logger.info(f'Deeplab training took {time.time() - start_time} seconds')
    cs.post_progress({"stage": STAGE2, "progress": 100})

cs.post_progress({"stage": STAGE3, "progress": 0})
counter = ProgressCounter(total=total_images, processed=0, cs=cs, logger=logger, stage=STAGE3)
for json_data, image_directory, mask_directory, vp, f, frame_amount in directories:
    start_time = time.time()
    delete_files_in_directory(mask_directory)
    deeplab = DeeplabInference(model_path=model_file, demo_mode=DEMO_MODE, counter=counter)
    polygon_file = f'{OUTPUT_PATH}/{vp}_labels.txt' if DEMO_MODE else f'{vp}_labels.txt'
    deeplab.run(image_directory=image_directory, mask_directory=mask_directory, polygon_file=polygon_file, h=h, w=w,
                pixel_hist_step=pixel_hist_step, approx_eps=approx_eps)
    logger.info(f'Deeplab inference took {time.time() - start_time} seconds')

    result, vector_chains, vectors = prepare_output(json_data, polygon_file)
    spl = os.path.basename(f).split('_')
    output_file_name = '_'.join(spl[1:len(spl)]) if os.path.basename(f).startswith('IN_') else os.path.basename(f)

    spl = output_file_name.split('.')
    outp_without_ext = '.'.join(spl[0:len(spl) - 1])
    with open(f"{OUTPUT_PATH}/{outp_without_ext}_chains_vectors.pkl", "wb") as pickle_file:
        pickle.dump(vector_chains, pickle_file)
    with open(f"{OUTPUT_PATH}/{outp_without_ext}_markups_vectors.pkl", "wb") as pickle_file:
        pickle.dump(vectors, pickle_file)

    output_file_name = "OUT_" + output_file_name
    result['files'][0]['file_name'] = outp_without_ext

    with open(f"{OUTPUT_PATH}/{output_file_name}", "w") as outfile:
        json.dump(result, outfile, ensure_ascii=False)
    # visualize if required
    if MAKE_VISUALIZATION:
        visualization.visualize_masks(f'{INPUT}/{vp}', result, OUTPUT_PATH)
    counter.report_status(report_amount=frame_amount % 1000, out_file=f"{output_file_name}",
                          chains_count=len(vector_chains), markups_count=len(vectors))
for params in empty_files:
    save_empty_file(*params)
cs.post_progress({"stage": STAGE3, "progress": 100})
cs.post_end()
logger.info(
    f'The whole process took {time.time() - run_time} seconds, work_format_training mode = {WORK_FORMAT_TRAINING}')
