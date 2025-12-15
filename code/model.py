import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, hidden_channels1=32, hidden_channels2=128, hidden_channels3=256):
        super(CNN, self).__init__()
        self.relu = nn.ReLU()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, hidden_channels1, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(hidden_channels1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        def res_block1(i_c, h_c):
            return nn.Sequential(
                nn.Conv2d(i_c, h_c, kernel_size=1, stride=1),
                nn.BatchNorm2d(h_c),
                nn.ReLU(),
                nn.Conv2d(h_c, h_c, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(h_c, i_c, kernel_size=1, stride=1),
                nn.BatchNorm2d(i_c),
            )

        self.cnn2 = nn.ModuleList([res_block1(hidden_channels1, 128) for _ in range(2)])

        self.cnn3 = nn.Sequential(
            nn.Conv2d(hidden_channels1, hidden_channels2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(hidden_channels2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        def res_block2(i_c, h_c):
            return nn.Sequential(
                nn.Conv2d(i_c, h_c, kernel_size=1, stride=1),
                nn.BatchNorm2d(h_c),
                nn.ReLU(),
                nn.Conv2d(h_c, h_c, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(h_c, i_c, kernel_size=1, stride=1),
                nn.BatchNorm2d(i_c),
            )

        self.cnn4 = nn.ModuleList([res_block2(hidden_channels2, 256) for _ in range(2)])
        self.cnn5 = nn.Sequential(
            nn.Conv2d(hidden_channels2, hidden_channels3, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(hidden_channels3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

    def forward(self, inp):
        out = self.cnn1(inp)
        for block in self.cnn2:
            out = block(out) + out
            out = self.relu(out)
        out = self.cnn3(out)
        for block in self.cnn4:
            out = block(out) + out
            out = self.relu(out)
        out = self.cnn5(out)
        out = out.squeeze(dim=2).permute(0, 2, 1)
        return out


class RNN(nn.Module):
    def __init__(self, input_dims=256, hidden_dims=256, num_classes=11):
        super(RNN, self).__init__()
        self.rnn = nn.GRU(input_dims, hidden_dims, num_layers=3, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dims * 2, num_classes)

    def forward(self, inp):
        out, _ = self.rnn(inp)
        logit = self.fc(out)
        return logit.permute(1, 0, 2)


class HNNet(nn.Module):
    def __init__(self):
        super(HNNet, self).__init__()
        self.cnn = CNN()
        self.rnn = RNN()

    def forward(self, inp):
        return self.rnn(self.cnn(inp))


if __name__ == '__main__':
    x = torch.randn(128, 1, 64, 512)
    net = HNNet()
    y = net(x)
    print(y.size())
