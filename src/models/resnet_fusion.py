import torch
import torch.nn as nn
import torchvision.models as models


class SpectrogramResNet18(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()

        self.backbone = models.resnet18(weights=None)

        self.backbone.conv1 = nn.Conv2d(
            1,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, embedding_dim)

    def forward(self, x):
        return self.backbone(x)


class FusionResNetClassifier(nn.Module):
    def __init__(
        self,
        handcrafted_dim: int,
        num_classes: int = 2,
        deep_dim: int = 256,
        hand_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.deep_branch = SpectrogramResNet18(embedding_dim=deep_dim)

        self.hand_branch = nn.Sequential(
            nn.Linear(handcrafted_dim, hand_dim),
            nn.BatchNorm1d(hand_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hand_dim, hand_dim),
            nn.BatchNorm1d(hand_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        fusion_dim = deep_dim + hand_dim

        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, logmel, handcrafted):
        deep_feat = self.deep_branch(logmel)
        hand_feat = self.hand_branch(handcrafted)

        fused = torch.cat([deep_feat, hand_feat], dim=1)
        return self.classifier(fused)
