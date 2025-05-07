# import segmentation_models_pytorch as smp, time
# import torch.nn as nn
#
#
# class MultiClassFeatureExtractor(nn.Module):
#     def __init__(self, in_channels, num_classes, feature_dim=30):
#         super().__init__()
#         self.pool = nn.AdaptiveAvgPool2d(num_classes)
#         self.conv = nn.Conv2d(in_channels, feature_dim, kernel_size=1)
#         self.flatten = nn.Flatten(2)
#         self.fc = nn.Linear(feature_dim, num_classes)  # Для предсказания классов
#
#     def forward(self, x):
#         x = self.pool(x)
#         x = self.conv(x)
#         x = self.flatten(x)  # Получаем [B, feature_dim, num_classes]
#         class_logits = self.fc(x.mean(dim=2))  # Сжимаем в [B, num_classes]
#         return x, class_logits
#
#
# class CustomDeeplabV3Plus(smp.DeepLabV3Plus):
#     def __init__(self, encoder_name="resnet50", num_classes=21, feature_dim=30, **kwargs):
#         super().__init__(encoder_name=encoder_name, classes=num_classes, **kwargs)
#         self.classification_head = MultiClassFeatureExtractor(
#             in_channels=self.encoder.out_channels,
#             num_classes=num_classes,
#             feature_dim=feature_dim
#         )
#
#     def forward(self, x):
#         features = self.encoder(x)
#         masks = self.segmentation_head(features)
#         class_vectors, class_logits = self.classification_head(features)
#         return masks, class_vectors, class_logits