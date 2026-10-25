import torch
from torch import nn
import torch, gzip, shutil, os
import time
import tqdm

def preds(model:nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) :
    outputs = []
    labels = []
    model.to(device)

    with torch.no_grad():
        for data in loader:
            images, targets = data
            images, targets = images.to(device), targets.to(device)
            outputs.append(model(images))
            labels.append(targets)
    return torch.cat(outputs), torch.cat(labels)


def compress(model : nn.Module, name :str):
    path = f"{name}.pt"
    compressed_path = path + ".gz"
    torch.save(model.state_dict(), path)
    with open(path, "rb") as f_in, gzip.open(compressed_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(path)
    return os.path.getsize(compressed_path)


def train_model(model, criterion, dataloaders, dataset_sizes, optimizer, epochs, device):
    since = time.time()

    model.to(device)
    for epoch in tqdm.tqdm(range(epochs)):
        print(f'Epoch {epoch}/{epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        print()
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    return model
