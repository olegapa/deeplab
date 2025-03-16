import os

import cv2
import numpy as np

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


def get_fourcc(video_filename):
    ext = os.path.splitext(video_filename)[-1].lower()  # Получаем расширение файла

    fourcc_dict = {
        '.avi': 'XVID',
        '.mp4': 'mp4v',
        '.m4v': 'mp4v',
        '.mov': 'avc1',
        '.mpg': 'MPEG',
        '.mpeg': 'MPEG',
        '.wmv': 'WMV2'
    }

    return cv2.VideoWriter_fourcc(*fourcc_dict.get(ext, 'mp4v'))  # По умолчанию mp4v

def visualize_masks(video, json_data, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for item in json_data['files']:
        cap = cv2.VideoCapture(video)

        if not cap.isOpened():
            print(f"Не удалось открыть видео: {video}")
            continue

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        output_video_path = os.path.join(output_dir, f"masked_{os.path.basename(video)}")

        fourcc = get_fourcc(video)
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        frames = []
        success, frame = cap.read()
        while success:
            frames.append(frame)
            success, frame = cap.read()

        cap.release()

        for chain in item['file_chains']:
            for markups in chain['chain_markups']:
                cls = int(markups['markup_path']['class'])
                if cls == 0:
                    continue

                frame_num = int(markups['markup_frame'])
                polygons = markups['markup_path']['polygons']
                x = int(markups['markup_path']['x'])
                y = int(markups['markup_path']['y'])
                color = color_mapping.get(cls, (255, 255, 255))

                if frame_num < len(frames):
                    overlay = frames[frame_num].copy()
                    mask = np.zeros_like(overlay, dtype=np.uint8)

                    for polygon in polygons:
                        polygon_shifted = [(px + x, py + y) for px, py in zip(polygon[::2], polygon[1::2])]
                        polygon_shifted = np.array(polygon_shifted, np.int32).reshape((-1, 1, 2))
                        cv2.fillPoly(mask, [polygon_shifted], color)

                    frames[frame_num] = cv2.addWeighted(frames[frame_num], 1, mask, 0.5, 0)
        for frame in frames:
            out.write(frame)

        out.release()
        print(f"Видео сохранено: {output_video_path}")
