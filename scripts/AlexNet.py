import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),       # 224 → 54
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),            # 54 → 26

            nn.Conv2d(96, 256, kernel_size=5, padding=2),     # 26 → 26
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),            # 26 → 12

            nn.Conv2d(256, 384, kernel_size=3, padding=1),    # 12 → 12
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, padding=1),    # 12 → 12
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),    # 12 → 12
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)             # 12 → 5
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 5 * 5, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(4096, num_classes)
        )


    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
