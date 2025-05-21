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
from custom_segmentation import CustomDeepLabV3Plus

# from container_status import ContainerStatus as CS

# Настройка логгера
logging.basicConfig(level=logging.INFO, filename='/output/training.log', filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ImageFile.LOAD_TRUNCATED_IMAGES = True

N_CLS = 9


def get_class_labels_from_masks(masks, num_classes=9):
    """
    Функция, извлекающая информацию о наличии классов в изображении.
    :param masks: (B, H, W) — маски сегментации
    :param num_classes: общее количество классов
    :return: (B, num_classes) — бинарные метки наличия классов
    """
    batch_size = masks.shape[0]
    class_labels = torch.zeros((batch_size, num_classes), device=masks.device)

    for i in range(batch_size):
        unique_classes = torch.unique(masks[i])  # Выделяем классы, присутствующие на изображении
        class_labels[i, unique_classes.long()] = 1  # Заполняем матрицу наличия классов

    return class_labels


class CustomSegmentationDataset(Dataset):
    def __init__(self, image_path, mask_path, transformations=None):
        self.im_paths = sorted(glob(f"{image_path}/*"))
        self.gt_paths = sorted(glob(f"{mask_path}/*"))
        self.transformations = transformations
        self.n_cls = N_CLS  # количество новых классов

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


class Metrics:

    def __init__(self, pred, gt, loss_fn, eps=1e-10, n_cls=N_CLS):

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
            # print(f'pred = {pred}')
            # logger.info(f'pred = {pred}')
            iou_per_class = []
            # logger.info(f'n_cls= {self.n_cls}')
            # print(self.n_cls)
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


class DeeplabTraining:

    def __init__(self, image_path=None,
                 mask_path=None,
                 model_path=None,
                 output_weights=None,
                 n_cls=N_CLS):
        self.image_path = image_path
        self.mask_path = mask_path
        self.model_path = model_path
        self.output_weights = output_weights
        self.n_cls = n_cls

    def get_dls(self, transformations, bs, split=[0.85, 0.15], ns=4):
        assert sum(split) == 1., "Sum of the split must be exactly 1"

        ds = CustomSegmentationDataset(transformations=transformations, image_path=self.image_path,
                                       mask_path=self.mask_path)
        n_cls = ds.n_cls

        tr_len = int(len(ds) * split[0])
        val_len = len(ds) - tr_len

        # Data split
        tr_ds, val_ds = torch.utils.data.random_split(ds, [tr_len, val_len])

        logger.info(f"\nThere are {len(tr_ds)} number of images in the train set")
        logger.info(f"There are {len(val_ds)} number of images in the validation set")

        # Get dataloaders
        tr_dl = DataLoader(dataset=tr_ds, batch_size=bs, shuffle=True, num_workers=ns, drop_last=True) if split[
                                                                                                              0] > 0 else None
        val_dl = DataLoader(dataset=val_ds, batch_size=bs, shuffle=False, num_workers=ns, drop_last=True)

        return tr_dl, val_dl, n_cls

    def tic_toc(self, start_time=None):
        return time.time() if start_time == None else time.time() - start_time

    def train(self, model, tr_dl, val_dl, segm_loss_fn, class_loss_fn, opt, device, epochs, save_prefix,
              threshold=0.005,
              save_path="saved_models"):
        tr_loss, tr_pa, tr_iou = [], [], []
        val_loss, val_pa, val_iou = [], [], []
        if tr_dl is not None:
            tr_len = len(tr_dl)
        else:
            tr_len = 0
        val_len = len(val_dl)
        best_loss, decrease, not_improve, early_stop_threshold = np.inf, 1, 0, 5
        os.makedirs(save_path, exist_ok=True)

        model.to(device)
        train_start = self.tic_toc()
        logger.info("Start training process...")

        for epoch in range(1, epochs + 1):
            tic = self.tic_toc()
            tr_loss_, tr_iou_, tr_pa_ = 0, 0, 0
            if tr_dl is not None:
                model.train()
                logger.info(f"Epoch {epoch} train process is started...")
                for idx, batch in enumerate(tqdm(tr_dl)):
                    ims, gts = batch
                    ims, gts = ims.to(device), gts.to(device)

                    (preds, class_probs, _) = model(ims)

                    met = Metrics(preds, gts, segm_loss_fn, n_cls=self.n_cls)

                    # Лосс для сегментации
                    loss_ = met.loss()

                    # Генерация class_labels из маски сегментации
                    class_labels = get_class_labels_from_masks(gts, num_classes=N_CLS)

                    # Лосс для classification head
                    classification_loss = class_loss_fn(class_probs, class_labels)

                    #loss_ += 0.5 * classification_loss

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

                    (preds, class_probs, _) = model(ims)

                    met = Metrics(preds, gts, segm_loss_fn, n_cls=self.n_cls)
                    # Генерация class_labels из маски сегментации
                    class_labels = get_class_labels_from_masks(gts, num_classes=N_CLS)

                    # Лосс для classification head
                    classification_loss = class_loss_fn(class_probs, class_labels)

                    val_loss_ += met.loss().item() #+ 0.5 * classification_loss.item()
                    val_iou_ += met.mIoU()
                    val_pa_ += met.PA()
                    # logger.info(f"val_iou = {val_iou_}\tval_pa_ = {val_pa_} ")

            logger.info(f"Epoch {epoch} train process is completed.")

            logger.info("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            if tr_dl is not None:
                tr_loss_ /= tr_len
                tr_iou_ /= tr_len
                tr_pa_ /= tr_len
                logger.info(f"\nEpoch {epoch} train process results: \n")
                logger.info(f"Train Time         -> {self.tic_toc(tic):.3f} secs")
                logger.info(f"Train Loss         -> {tr_loss_:.3f}")
                logger.info(f"Train PA           -> {tr_pa_:.3f}")
                logger.info(f"Train IoU          -> {tr_iou_:.3f}")
                tr_loss.append(tr_loss_)
                tr_iou.append(tr_iou_)
                tr_pa.append(tr_pa_)
            val_loss_ /= val_len
            val_iou_ /= val_len
            val_pa_ /= val_len
            # logger.info(f"val_len = {val_len}")
            logger.info(f"Validation Loss    -> {val_loss_:.3f}")
            logger.info(f"Validation PA      -> {val_pa_:.3f}")
            logger.info(f"Validation IoU     -> {val_iou_:.3f}\n")



            val_loss.append(val_loss_)
            val_iou.append(val_iou_)
            val_pa.append(val_pa_)

            if tr_dl is not None:
                if best_loss > (val_loss_ + threshold):
                    logger.info(f"Loss decreased from {best_loss:.3f} to {val_loss_:.3f}!")
                    best_loss = val_loss_
                    decrease += 1
                    if decrease % 2 == 0:
                        logger.info(f"Saving the model with the best loss value to {self.output_weights}")
                        os.makedirs('/'.join(self.output_weights.split('/')[0:-1]), exist_ok=True)
                        torch.save(model, f"{self.output_weights}")

                else:
                    not_improve += 1
                    best_loss = val_loss_
                    logger.info(f"Loss did not decrease for {not_improve} epoch(s)!")
                    if not_improve == early_stop_threshold:
                        logger.info(
                            f"Stopping training process becuase loss value did not decrease for {early_stop_threshold} epochs!")
                        break
                logger.info("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

        logger.info(f"Train process is completed in {(self.tic_toc(train_start)) / 60:.3f} minutes.")

        return {"tr_loss": tr_loss, "tr_iou": tr_iou, "tr_pa": tr_pa,
                "val_loss": val_loss, "val_iou": val_iou, "val_pa": val_pa}

    def load_model(self, n_cls):
        if self.model_path:
            if os.path.exists(self.model_path):
                return torch.load(f"{self.model_path}", weights_only=False)
        aux_params = dict(
            classes=n_cls,
            activation='sigmoid'
        )
        return CustomDeepLabV3Plus(num_classes=n_cls, encoder_name='resnet50', feature_dim=30)

    def run(self, image_path, mask_path, model_path, output_weights, h=None, w=None, epoches=50, eval_mode=False):
        self.image_path, self.mask_path, self.model_path, self.output_weights = image_path, mask_path, model_path, output_weights
        mean, std, split = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], [0.85, 0.15]
        if eval_mode:
            split = [0, 1]
            epoches = 1
        im_h = h if h else 224
        im_w = w if w else 224
        trans = A.Compose([A.Resize(im_h, im_w), A.augmentations.transforms.Normalize(mean=mean, std=std),
                           ToTensorV2(transpose_mask=True)])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # logger.info(f"device = {device}")

        tr_dl, val_dl, self.n_cls = self.get_dls(transformations=trans, split=split, bs=32)

        model = self.load_model(self.n_cls)

        loss_fn = torch.nn.CrossEntropyLoss()
        classification_loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=3e-4)

        history = self.train(model=model, tr_dl=tr_dl, val_dl=val_dl,
                             segm_loss_fn=loss_fn, class_loss_fn=classification_loss_fn, opt=optimizer, device=device,
                             epochs=epoches, save_prefix="clothing")

    #python run_train.py --image_path /output/images --mask_path /output/masks --output_weights /output/deeplab_weights.pt --host_web ""
