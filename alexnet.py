import torch
import torch.nn as nn


class AlexNet(nn.Module):
    """
    Implementation of AlexNet

    Reference:
    ImageNet Classification with Deep Convolutional Neural Networks; Krizhevsky, Sutskever and Hinton.
    https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
    """    
    def __init__(self, num_classes: int=1000) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.convnet = nn.Sequential(       # In:       [N, 3, 227, 227]
            nn.Conv2d(3, 96, 11, 4),        # Out:      [N, 96, 55, 55]
            nn.LocalResponseNorm(5, 1e-4, 0.75, 2),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2),             # Out:      [N, 96, 27, 27]
            nn.Conv2d(96, 256, 5, 1, 2),    # Out:      [N, 256, 27, 27]
            nn.LocalResponseNorm(5, 1e-4, 0.75, 2),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2),             # Out:      [N, 256, 13, 13]
            nn.Conv2d(256, 384, 3, 1, 1),   # Out:      [N, 384, 13, 13]
            nn.ReLU(True),
            nn.Conv2d(384, 384, 3, 1, 1),   # Out:      [N, 384, 13, 13]
            nn.ReLU(True),
            nn.Conv2d(384, 256, 3, 1, 1),   # Out:      [N, 256, 13, 13]
            nn.ReLU(True),
            nn.MaxPool2d(3, 2),             # Out:      [N, 256, 6, 6]
        )

        self.fc = nn.Sequential(            # In:       [N, 256 * 6 * 6] = [N, 9216]
            nn.Linear(256 * 6 * 6, 4096),   # Out:      [N, 4096]
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),          # Out:      [N, 4096]
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),   # Out:      [N, num_classes]
        )
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convnet(x)                 # Out:      [N, 256, 6, 6]
        x = x.flatten(start_dim=1)          # Out:      [N, 256 * 6 * 6]
        return self.fc(x)                   # Out:      [N, num_classes]
    
    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    if (m.out_channels == 256) or (m.in_channels == 384 and m.out_channels == 384): 
                        nn.init.ones_(m.bias)
                    else:
                        nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    if m.weight.shape[0] != self.num_classes: 
                        nn.init.ones_(m.bias)
                    else:
                        nn.init.zeros_(m.bias)