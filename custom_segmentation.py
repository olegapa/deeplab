import logging

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


logging.basicConfig(level=logging.INFO, filename='/output/training.log', filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
class CustomDeepLabV3Plus(smp.DeepLabV3Plus):
    def __init__(self, encoder_name="resnet50", num_classes=9, feature_dim=30):
        super().__init__(encoder_name=encoder_name, classes=num_classes)
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        encoder_channels = self.encoder.out_channels[-1]  # Последний уровень энкодера

        self.pool = nn.AdaptiveAvgPool2d(1)  # (B, encoder_channels, 1, 1)

        self.fc_vectors = nn.Linear(encoder_channels, num_classes * feature_dim)  # (B, N_CLASSES * 30)

        self.fc_probs = nn.Linear(feature_dim, 1)  # (B, N_CLASSES, 1) → Вероятности для каждого класса
        self.sigmoid = nn.Sigmoid()  # Применяем сигмоиду

    def forward(self, x):
        encoder_output = self.encoder(x)

        decoder_output = self.decoder(encoder_output)

        segmentation_output = self.segmentation_head(decoder_output)

        pooled_features = self.pool(encoder_output[-1]).view(encoder_output[-1].shape[0], -1)  # (B, encoder_channels)

        class_vectors = self.fc_vectors(pooled_features).view(-1, self.num_classes, self.feature_dim)  # (B, N_CLASSES, 30)

        class_probs = self.sigmoid(self.fc_probs(class_vectors)).squeeze(-1)  # (B, N_CLASSES)

        return segmentation_output, class_probs, class_vectors
