import torch
import time
from data import SVHNDataset
from model import HNNet

if __name__ == '__main__':
    lr = 0.002
    epochs = 10
    batch_size = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = SVHNDataset('tcdata', split='train')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    model = HNNet()
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = torch.nn.CTCLoss(blank=10, reduction='sum', zero_infinity=True)
    start_time = time.time()
    latest_time = start_time
    for epoch in range(epochs):
        running_loss = 0.0
        running_corrects = 0.0
        for labels, imgs in dataloader:
            labels = labels.to(device)
            imgs = imgs.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            log_prob = out.log_softmax(dim=-1)
            input_length = torch.tensor([8] * batch_size).to(device)
            label_length = torch.tensor([6] * batch_size, dtype=torch.long)
            l = loss(log_prob, labels, input_length, label_length)
            l.backward()
            optimizer.step()
            running_loss += l.item()
        running_loss = running_loss / len(dataloader)
        this_time = time.time()
        print(f'[{epoch + 1:3d}/{epochs:3d}]loss: {running_loss:.4f}, time: {this_time - latest_time:.2f}s')
        latest_time = this_time



