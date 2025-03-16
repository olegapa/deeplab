import argparse
import bisect
import json, os, torch, cv2, numpy as np, albumentations as A
import logging
import numpy as np

from PIL import Image, ImageOps
from matplotlib import pyplot as plt
from glob import glob
from PIL import ImageFile
from torch.utils.data import random_split, Dataset, DataLoader
from albumentations.pytorch import ToTensorV2
import random
from torchvision import transforms as tfs
import segmentation_models_pytorch as smp, time
from tqdm import tqdm
from torch.nn import functional as F

# from container_status import ContainerStatus as CS
from progress_counter import ProgressCounter

ImageFile.LOAD_TRUNCATED_IMAGES = True
logging.basicConfig(level=logging.INFO, filename='/output/deeplab.log', format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# parser = argparse.ArgumentParser(description="Process some images.")

# Добавляем аргументы
# parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
# parser.add_argument('--mask_path', type=str, help='Path to the mask image')
# parser.add_argument('--output_path', type=str, required=True, help='Path to the input image')
# parser.add_argument('--model_path', type=str, required=True, help='Path to the input image')
# parser.add_argument('--class_info_path', help='Path to the output class info file')
# parser.add_argument("--demo_mode", action="store_true", help="Flag for demo mode")
# parser.add_argument("--host_web", type=str, help="url host with web")

# Парсим аргументы
# args = parser.parse_args()

# Получаем значения аргументов
# image_path = args.image_path
# mask_path = args.mask_path
# output_path = args.output_path
# model_path = args.model_path
# class_info_path = args.class_info_path
# demo_mode = args.demo_mode
# cs = CS(args.host_web, logger)

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

N_CLS = 9


class CustomSegmentationDataset(Dataset):
    def __init__(self, image_path, transformations=None):
        self.im_paths = sorted(glob(f"{image_path}/*"))
        self.transformations = transformations
        self.n_cls = N_CLS  # количество новых классов

    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, idx):
        im, size = self.get_im_gt(self.im_paths[idx])
        if self.transformations:
            im = self.apply_transformations(im)
        return im, size, self.im_paths[idx]

    def get_im_gt(self, im_path):
        return self.read_im(im_path)

    def read_im(self, im_path):
        im = Image.open(im_path).convert("RGB")
        size = im.size
        im = np.array(im)
        return im, size

    def apply_transformations(self, im):
        transformed = self.transformations(image=im)
        return transformed["image"]


class DeeplabInference:
    def __init__(self, image_path=None,
                 output_path=None,
                 model_path=None,
                 class_info_path=None,
                 demo_mode=False,
                 counter=None):
        self.image_path = image_path
        self.output_path = output_path
        self.model_path = model_path
        self.class_info_path = class_info_path
        self.demo_mode = demo_mode
        self.counter = counter

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = torch.load(f"{self.model_path}", weights_only=False)
        self.model.eval()
        self.model = self.model.to(self.device)
    def get_dls(self, transformations, split=[0, 1], ns=4, pixel_hist_step=None):
        assert sum(split) == 1., "Sum of the split must be exactly 1"

        ds = CustomSegmentationDataset(transformations=transformations, image_path=self.image_path)
        n_cls = ds.n_cls

        val_len = int(len(ds) * split[0])
        test_len = len(ds) - (val_len)

        # Data split
        val_ds, test_ds = torch.utils.data.random_split(ds, [val_len, test_len])

        print(f"There are {len(val_ds)} number of images in the validation set")
        print(f"There are {len(test_ds)} number of images in the test set\n")

        test_d = DataLoader(dataset=test_ds, batch_size=1, shuffle=False, num_workers=ns)

        return test_d, n_cls

    def inference(self, dl, model, device, pixel_hist_step=None, approx_eps=None):
        # torch.set_printoptions(threshold=10000, edgeitems=30, precision=1, linewidth=1000, sci_mode=False)
        # np.set_printoptions(threshold=10000, edgeitems=15, precision=1, linewidth=1000, suppress=True)
        torch.set_printoptions(precision=6)
        os.makedirs(self.output_path, exist_ok=True)
        if self.demo_mode:
            polygonized_output_path = f"{self.output_path}_polygonized_mask"
            os.makedirs(polygonized_output_path, exist_ok=True)

        labels_dict = {}
        polygons_dict = {}
        total = len(dl)
        if pixel_hist_step:
            hist_data = dict()
            for label, color in color_mapping.items():
                hist_data[label] = dict()
                for i in np.arange(0, 1,  pixel_hist_step):
                    hist_data[label][f"{i:.2f}"] = 0
        for idx, (im, orig_size, save_name) in enumerate(dl):
            if idx % 1000 == 0 and self.counter and idx != 0:
                self.counter.report_status(1000)
            # if idx % 1000 == 0:
                # cs.post_progress('{:.2%}'.format(idx / total))
            im = im.to(device)

            # Get predicted mask
            with torch.no_grad():
                (pred, class_out, vector) = model(im.to(device))
                vector = vector.squeeze()
                # encoder_res = model.encoder(im.to(device))
                # logger.info(f"encoder shape = {len(encoder_res)}")
                # for i in range(len(encoder_res)):
                #     logger.info(f"{i} item has shape {encoder_res[i].shape}")
                logger.info(f"class_outp.shape = {class_out.shape}\n class_out = {class_out}")
                logger.info(f"class_outp.shape = {vector.shape}\n class_out = {vector}")
                probs = torch.softmax(pred, dim=1)
                pred_for_score = torch.argmax(pred, dim=1)
                pred = pred_for_score.squeeze(0).cpu().numpy()
                # print(model.classification_head)

            # Convert prediction to an image and save
            pred_img = Image.fromarray(pred.astype(np.uint8))
            pred_img = pred_img.resize(orig_size, resample=Image.NEAREST)
            pred_img.save(os.path.join(self.output_path, save_name[0].split('/')[-1]))

            if self.demo_mode:
                polygonized_pred = np.zeros((*pred.shape, 3), dtype=np.uint8)

            detected_classes = set()
            polygons_per_image = {}

            for label, color in color_mapping.items():
                # TODO: Delete it after training file with bboxes will appear
                if int(label) == 0:
                    continue
                mask = (pred == label).astype(np.uint8)
                score_mask = (pred_for_score == label)

                if np.any(mask):
                    # logger.info(f"found label {label}")
                    class_probs = probs[:, label, :, :]
                    class_sum = class_probs[score_mask].sum().item()  # сумма вероятностей
                    num_pixels = mask.sum().item()  # Количество пикселей класса
                    score = class_sum / num_pixels if num_pixels > 0 else 0.0
                    if pixel_hist_step:
                        # logger.info(f"class_probs[score_mask] = {class_probs[score_mask]}\nsize = {len(class_probs[score_mask])}")
                        for e in class_probs[score_mask]:
                            # logger.info(f"value = {e} gets idx = {idx}")
                            # logger.info(f"value = {e}")
                            if e >= float(0.999):
                                e = 0.99
                            # logger.info(f"goes lable={label} {(e - e%pixel_hist_step):.2f}'")
                            hist_data[label][f"{(e - e%pixel_hist_step):.2f}"] += 1

                    detected_classes.add(label)

                    # Находим контуры (полигоны) на маске
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    # approximate countours
                    if approx_eps:
                        approximated_contours = list()
                        for c in contours:
                            peri = cv2.arcLength(c, True)
                            approx = cv2.approxPolyDP(c, approx_eps * peri, True)
                            approximated_contours.append(approx)
                        contours = approximated_contours
                    polygons_per_image[label] = {"score": score, "polygons": [contour.squeeze().flatten().tolist() for contour in contours], "markup_vector": vector[label].tolist()}

                    if self.demo_mode:
                        # Рисуем полигоны на пустой маске
                        cv2.drawContours(polygonized_pred, contours, -1, color, thickness=cv2.FILLED)

            polygons_dict[save_name[0].split('/')[-1]] = polygons_per_image

            if self.demo_mode:
                restored_mask = np.zeros_like(polygonized_pred, dtype=np.uint8)

                for label, polygons in polygons_per_image.items():
                    for polygon in polygons["polygons"]:
                        # Преобразуем плоский список координат обратно в массив точек
                        points = np.array(polygon, dtype=np.int32).reshape(-1, 2)
                        # Рисуем полигон на маске
                        cv2.fillPoly(restored_mask, [points], color_mapping[label])

                # Сохраняем маску, восстановленную по полигонам
                polygonized_pred_img = Image.fromarray(restored_mask)
                polygonized_pred_img = polygonized_pred_img.resize(orig_size, resample=Image.NEAREST)
                polygonized_pred_img.save(os.path.join(polygonized_output_path, save_name[0].split('/')[-1]))

            labels_dict[save_name[0].split('/')[-1]] = ' '.join(map(str, detected_classes))

        # Сохранение полигонов
        with open(self.class_info_path, 'a', encoding='utf-8') as polygon_file:
            json.dump(polygons_dict, polygon_file, ensure_ascii=False)  # Сохраняем полигоны в файл

        print(f"Saved all predicted masks and polygons in {self.output_path}")
        if pixel_hist_step:
            with open(f"{self.class_info_path}_histogram", 'a', encoding='utf-8') as hist_file:
                json.dump(hist_data, hist_file, ensure_ascii=False)  # Сохраняем полигоны в файл

    def run(self, image_directory, mask_directory, polygon_file, h=None, w=None, pixel_hist_step=None, approx_eps=None):
        self.image_path, self.output_path, self.class_info_path = image_directory, mask_directory, polygon_file
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        im_h = h if h else 224
        im_w = w if w else 224
        trans = A.Compose([A.Resize(im_h, im_w), A.augmentations.transforms.Normalize(mean=mean, std=std),
                           ToTensorV2(transpose_mask=True)])

        test_dl, n_cls = self.get_dls(transformations=trans, split=[0, 1])

        self.inference(test_dl, model=self.model, device=self.device, pixel_hist_step=pixel_hist_step, approx_eps=approx_eps)
