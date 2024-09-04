import argparse
import json, os, torch, cv2, numpy as np, albumentations as A
import logging

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

from container_status import ContainerStatus as CS

ImageFile.LOAD_TRUNCATED_IMAGES = True
logging.basicConfig(level=logging.INFO, filename='/output/deeplab.log', format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Process some images.")

# Добавляем аргументы
parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
parser.add_argument('--mask_path', type=str, help='Path to the mask image')
parser.add_argument('--output_path', type=str, required=True, help='Path to the input image')
parser.add_argument('--model_path', type=str, required=True, help='Path to the input image')
parser.add_argument('--class_info_path', help='Path to the output class info file')
parser.add_argument("--demo_mode", action="store_true", help="Flag for demo mode")

# Парсим аргументы
args = parser.parse_args()

# Получаем значения аргументов
image_path = args.image_path
mask_path = args.mask_path
output_path = args.output_path
model_path = args.model_path
class_info_path = args.class_info_path
demo_mode = args.demo_mode
cs = CS(args.host_web, logger)

color_mapping = {
    0: (0, 0, 0),         # Background
    1: (128, 0, 0),       # Jacket/Coat
    2: (0, 128, 0),       # Shirt/Blouse
    3: (128, 128, 0),     # Sweater/Sweatshirt/Hoodie
    4: (0, 0, 128),       # Dress/Romper
    5: (128, 0, 128),     # Pants/Jeans/Leggings
    6: (0, 128, 128),     # Shorts
    7: (128, 128, 128),   # Skirt
    8: (64, 0, 0),        # Shoes
    9: (192, 0, 0),       # Vest
    10: (64, 128, 0),     # Boots
    11: (192, 128, 0),    # Bodysuit/T-shirt/Top
    12: (64, 0, 128),     # Bag/Purse
    13: (192, 0, 128),    # Hat
    14: (64, 128, 128),   # Scarf/Tie
    15: (192, 128, 128),  # Gloves
    16: (0, 64, 0),       # Blazer/Suit
    17: (128, 64, 0),     # Underwear/Swim
    18: (0, 192, 0)       # Socks/Stockings
}

class CustomSegmentationDataset(Dataset):
    def __init__(self, transformations=None):
        self.im_paths = sorted(glob(f"{image_path}/*"))
        self.transformations = transformations
        self.n_cls = 19 # количество новых классов

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


def get_dls(transformations, split=[0, 1], ns=4):
    assert sum(split) == 1., "Sum of the split must be exactly 1"

    ds = CustomSegmentationDataset(transformations=transformations)
    n_cls = ds.n_cls

    val_len = int(len(ds) * split[0])
    test_len = len(ds) - (val_len)

    # Data split
    val_ds, test_ds = torch.utils.data.random_split(ds, [val_len, test_len])

    print(f"There are {len(val_ds)} number of images in the validation set")
    print(f"There are {len(test_ds)} number of images in the test set\n")

    test_d = DataLoader(dataset=test_ds, batch_size=1, shuffle=False, num_workers=ns)

    return test_d, n_cls


mean, std, im_h, im_w = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 256, 256
trans = A.Compose([A.Resize(im_h, im_w), A.augmentations.transforms.Normalize(mean=mean, std=std),
                   ToTensorV2(transpose_mask=True)])
device = "cuda" if torch.cuda.is_available() else "cpu"

test_dl, n_cls = get_dls(transformations=trans, split=[0, 1])


def inference(dl, model, device):
    os.makedirs(output_path, exist_ok=True)
    total = len(dl)
    if demo_mode:
        colored_output_path = f"{output_path}_colored"
        os.makedirs(colored_output_path, exist_ok=True)
    labels_dict = {}
    for idx, (im, orig_size, save_name) in enumerate(dl):
        if idx%1000 == 0:
            cs.post_progress('{:.2%}'.format(idx/total))
        im = im.to(device)
        # Get predicted mask
        with torch.no_grad():
            pred = torch.argmax(model(im.to(device)), dim=1).squeeze(0).cpu().numpy()
        # Convert prediction to an image and save
        pred_img = Image.fromarray(pred.astype(np.uint8))
        pred_img = pred_img.resize(orig_size, resample=Image.NEAREST)
        pred_img.save(os.path.join(output_path, save_name[0].split('/')[-1]))
        if demo_mode:
            colored_pred = np.zeros((*pred.shape, 3), dtype=np.uint8)
        detected_classes = set()
        for label, color in color_mapping.items():
            mask = pred == label
            if np.any(mask):
                detected_classes.add(label)
                if demo_mode:
                    colored_pred[mask] = color
        if demo_mode:
            colored_pred_img = Image.fromarray(colored_pred)
            colored_pred_img = colored_pred_img.resize(orig_size, resample=Image.NEAREST)
            colored_pred_img.save(os.path.join(colored_output_path, save_name[0].split('/')[-1]))
        labels_dict[save_name[0].split('/')[-1]] = ' '.join(map(str, detected_classes))
    with open(class_info_path, 'a') as class_info_file:
        for img_name in sorted(labels_dict):
            class_info_file.write(f"{img_name} {labels_dict[img_name]}\n")

    print(f"Saved all predicted masks in {output_path}")


model = torch.load(f"{model_path}")
model.eval()
model = model.to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=3e-4)

inference(test_dl, model=model, device=device)
