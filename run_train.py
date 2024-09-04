import argparse
import json, os, torch, cv2, numpy as np, albumentations as A
from PIL import Image
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
import logging

from container_status import ContainerStatus as CS

# Настройка логгера
logging.basicConfig(level=logging.INFO, filename='/output/training.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser(description="Process some images.")

# Добавляем аргументы
parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
parser.add_argument('--mask_path', type=str, help='Path to the mask image')
parser.add_argument('--model_path', type=str, help='Path to the input image')
parser.add_argument('--output_weights', type=str, help='Path to the input image')

# Парсим аргументы
args = parser.parse_args()

# Получаем значения аргументов
image_path = args.image_path
mask_path = args.mask_path
model_path = args.model_path
output_weights = args.output_weights
cs = CS(args.host_web, logger)


class CustomSegmentationDataset(Dataset):
    def __init__(self, transformations=None):
        self.im_paths = sorted(glob(f"{image_path}/*"))
        self.gt_paths = sorted(glob(f"{mask_path}/*"))
        self.transformations = transformations
        self.n_cls = 19  # количество новых классов

        assert len(self.im_paths) == len(self.gt_paths)

    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, idx):

        im, gt = self.get_im_gt(self.im_paths[idx], self.gt_paths[idx])
        if self.transformations:
            im, gt = self.apply_transformations(im, gt)
        return im, gt

    def get_im_gt(self, im_path, gt_path=None):
        return self.read_im(im_path, gt_path)

    def read_im(self, im_path, gt_path):
        im = np.array(Image.open(im_path).convert("RGB"))
        gt = np.array(Image.open(gt_path).convert("L"))
        return im, gt

    def apply_transformations(self, im, gt):
        transformed = self.transformations(image=im, mask=gt)
        return transformed["image"], transformed["mask"]


def get_dls(transformations, bs, split=[0.95, 0.05], ns=4):
    assert sum(split) == 1., "Sum of the split must be exactly 1"

    ds = CustomSegmentationDataset(transformations=transformations)
    n_cls = ds.n_cls

    tr_len = int(len(ds) * split[0])
    val_len = len(ds) - tr_len

    # Data split
    tr_ds, val_ds = torch.utils.data.random_split(ds, [tr_len, val_len])

    logger.info(f"\nThere are {len(tr_ds)} number of images in the train set")
    logger.info(f"There are {len(val_ds)} number of images in the validation set")

    # Get dataloaders
    tr_dl = DataLoader(dataset=tr_ds, batch_size=bs, shuffle=True, num_workers=ns)
    val_dl = DataLoader(dataset=val_ds, batch_size=bs, shuffle=False, num_workers=ns)

    return tr_dl, val_dl, n_cls


class Metrics:

    def __init__(self, pred, gt, loss_fn, eps=1e-10, n_cls=19):

        self.pred, self.gt = torch.argmax(pred, dim=1), gt.squeeze(1)  # (batch, width, height)
        self.loss_fn, self.eps, self.n_cls, self.pred_ = loss_fn, eps, n_cls, pred

    def to_contiguous(self, inp):
        return inp.contiguous().view(-1)

    def PA(self):

        with torch.no_grad():
            match = torch.eq(self.pred, self.gt).int()

        return float(match.sum()) / float(match.numel())

    def mIoU(self):

        with torch.no_grad():

            pred, gt = self.to_contiguous(self.pred), self.to_contiguous(self.gt)

            iou_per_class = []
            logger.info(self.n_cls)
            for c in range(self.n_cls):
                match_pred = pred == c
                match_gt = gt == c

                if match_gt.long().sum().item() == 0:
                    iou_per_class.append(np.nan)

                else:

                    intersect = torch.logical_and(match_pred, match_gt).sum().float().item()
                    union = torch.logical_or(match_pred, match_gt).sum().float().item()

                    iou = (intersect + self.eps) / (union + self.eps)
                    iou_per_class.append(iou)

            return np.nanmean(iou_per_class)

    def loss(self):
        return self.loss_fn(self.pred_, self.gt.long())


def tic_toc(start_time=None):
    return time.time() if start_time == None else time.time() - start_time


def train(model, tr_dl, val_dl, loss_fn, opt, device, epochs, save_prefix, threshold=0.005, save_path="saved_models"):
    tr_loss, tr_pa, tr_iou = [], [], []
    val_loss, val_pa, val_iou = [], [], []
    tr_len, val_len = len(tr_dl), len(val_dl)
    best_loss, decrease, not_improve, early_stop_threshold = np.inf, 1, 0, 5
    os.makedirs(save_path, exist_ok=True)

    model.to(device)
    train_start = tic_toc()
    logger.info("Start training process...")

    for epoch in range(1, epochs + 1):
        tic = tic_toc()
        tr_loss_, tr_iou_, tr_pa_ = 0, 0, 0

        model.train()
        logger.info(f"Epoch {epoch} train process is started...")
        for idx, batch in enumerate(tqdm(tr_dl)):
            ims, gts = batch
            ims, gts = ims.to(device), gts.to(device)

            preds = model(ims)

            met = Metrics(preds, gts, loss_fn, n_cls=n_cls)
            loss_ = met.loss()

            tr_iou_ += met.mIoU()

            tr_pa_ += met.PA()
            tr_loss_ += loss_.item()

            loss_.backward()
            opt.step()
            opt.zero_grad()

        logger.info(f"Epoch {epoch} validation process is started...")
        model.eval()
        val_loss_, val_iou_, val_pa_ = 0, 0, 0

        with torch.no_grad():
            for idx, batch in enumerate(tqdm(val_dl)):
                ims, gts = batch
                ims, gts = ims.to(device), gts.to(device)

                preds = model(ims)

                met = Metrics(preds, gts, loss_fn, n_cls=n_cls)

                val_loss_ += met.loss().item()
                val_iou_ += met.mIoU()
                val_pa_ += met.PA()

        logger.info(f"Epoch {epoch} train process is completed.")

        tr_loss_ /= tr_len
        tr_iou_ /= tr_len
        tr_pa_ /= tr_len

        val_loss_ /= val_len
        val_iou_ /= val_len
        val_pa_ /= val_len

        logger.info("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        logger.info(f"\nEpoch {epoch} train process results: \n")
        logger.info(f"Train Time         -> {tic_toc(tic):.3f} secs")
        logger.info(f"Train Loss         -> {tr_loss_:.3f}")
        logger.info(f"Train PA           -> {tr_pa_:.3f}")
        logger.info(f"Train IoU          -> {tr_iou_:.3f}")
        logger.info(f"Validation Loss    -> {val_loss_:.3f}")
        logger.info(f"Validation PA      -> {val_pa_:.3f}")
        logger.info(f"Validation IoU     -> {val_iou_:.3f}\n")

        tr_loss.append(tr_loss_)
        tr_iou.append(tr_iou_)
        tr_pa.append(tr_pa_)

        val_loss.append(val_loss_)
        val_iou.append(val_iou_)
        val_pa.append(val_pa_)

        if best_loss > (val_loss_ + threshold):
            logger.info(f"Loss decreased from {best_loss:.3f} to {val_loss_:.3f}!")
            best_loss = val_loss_
            decrease += 1
            if decrease % 2 == 0:
                logger.info("Saving the model with the best loss value...")
                os.makedirs('/'.join(model_path.split('/')[0:-1]), exist_ok=True)
                torch.save(model, f"{output_weights}")


        else:
            not_improve += 1
            best_loss = val_loss_
            logger.info(f"Loss did not decrease for {not_improve} epoch(s)!")
            if not_improve == early_stop_threshold:
                logger.info(
                    f"Stopping training process becuase loss value did not decrease for {early_stop_threshold} epochs!")
                break
        logger.info("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

    logger.info(f"Train process is completed in {(tic_toc(train_start)) / 60:.3f} minutes.")

    return {"tr_loss": tr_loss, "tr_iou": tr_iou, "tr_pa": tr_pa,
            "val_loss": val_loss, "val_iou": val_iou, "val_pa": val_pa}


def load_model(n_cls):
    if model_path:
        if os.path.exists(model_path):
            return torch.load(f"{model_path}")
    return smp.DeepLabV3Plus(classes=n_cls)


mean, std, im_h, im_w = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 256, 256
trans = A.Compose([A.Resize(im_h, im_w), A.augmentations.transforms.Normalize(mean=mean, std=std),
                   ToTensorV2(transpose_mask=True)])
device = "cuda" if torch.cuda.is_available() else "cpu"

tr_dl, val_dl, n_cls = get_dls(transformations=trans, split=[0.95, 0.05], bs=32)

model = load_model(n_cls)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=3e-4)

history = train(model=model, tr_dl=tr_dl, val_dl=val_dl,
                loss_fn=loss_fn, opt=optimizer, device=device,
                epochs=50, save_prefix="clothing")
