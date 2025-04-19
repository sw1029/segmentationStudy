import torch.nn as nn

class AutoEncoder(nn.Module):
    '''
    input
        img     (B,3,1024,1024)
        label   (B,C,1024,1024)
    output
        pred     (B,C,1024,1024)
    '''
    def __init__(self):
        super(self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.ConvTranspose2d(64, 91, 2, stride=2),
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x