import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

class CustomDeepLabV3Plus(smp.DeepLabV3Plus):
    def __init__(self, encoder_name="resnet50", num_classes=9, feature_dim=30):
        super().__init__(encoder_name=encoder_name, classes=num_classes)

        self.num_classes = num_classes
        self.feature_dim = feature_dim
        encoder_channels = self.encoder.out_channels[-1]  # Последний уровень энкодера

        # 🔹 1. Усредняем карты признаков
        self.pool = nn.AdaptiveAvgPool2d(1)  # (B, encoder_channels, 1, 1)

        # 🔹 2. Генерируем class_vectors из feature map
        self.fc_vectors = nn.Linear(encoder_channels, num_classes * feature_dim)  # (B, N_CLASSES * 30)

        # 🔹 3. Генерируем class_probs из class_vectors
        self.fc_probs = nn.Linear(feature_dim, 1)  # (B, N_CLASSES, 1) → Вероятности для каждого класса
        self.sigmoid = nn.Sigmoid()  # Применяем сигмоиду

    def forward(self, x):
        # 1️⃣ Получаем ВСЕ уровни признаков из энкодера
        encoder_output = self.encoder(x)

        # 2️⃣ Передаём ВСЕ уровни в decoder для сегментации
        segmentation_output = self.segmentation_head(self.decoder(*encoder_output))

        # 3️⃣ Усредняем feature map
        pooled_features = self.pool(encoder_output[-1]).view(encoder_output[-1].shape[0], -1)  # (B, encoder_channels)

        # 4️⃣ Создаём class_vectors
        class_vectors = self.fc_vectors(pooled_features).view(-1, self.num_classes, self.feature_dim)  # (B, N_CLASSES, 30)

        # 5️⃣ Используем class_vectors для предсказания вероятностей
        class_probs = self.sigmoid(self.fc_probs(class_vectors)).squeeze(-1)  # (B, N_CLASSES)

        return segmentation_output, class_probs, class_vectors
