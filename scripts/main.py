import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from architecture import ResNet
import warnings
from torch.utils.data import random_split
from create_dataset import RandomImagePixelationDataset
from create_dataset import stack


def get_dataset():
    ds = RandomImagePixelationDataset(
        r"data/resized_training",
        width_range=(4, 32),
        height_range=(4, 32),
        size_range=(4, 16)
    )

    train_ratio = 0.8

    train_size = int(train_ratio * len(ds))
    val_size = len(ds) - train_size

    train_dataset, val_dataset = random_split(ds, [train_size, val_size])
    return train_dataset, val_dataset


def training_step(network, optimizer, data, targets, known_arrays):
    optimizer.zero_grad()
    output = network(data)
    loss = torch.sqrt(F.mse_loss(output.float() * ~known_arrays, targets.float() * ~known_arrays))
    loss.backward()
    optimizer.step()
    return loss.item()


def eval_step(network, data, targets, known_arrays):
    with torch.no_grad():
        output = network(data)
        loss = torch.sqrt(F.mse_loss(output.float() * ~known_arrays, targets.float() * ~known_arrays))
        return loss.item()


def training_loop(
        network: torch.nn.Module,
        train_data: torch.utils.data.Dataset,
        eval_data: torch.utils.data.Dataset,
        num_epochs: int,
        device: str,
        show_progress: bool = False,
) -> tuple[list, list]:
    device = torch.device(device)
    if "cuda" in device.type and not torch.cuda.is_available():
        warnings.warn("CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    torch.manual_seed(0)
    network.to(device)
    batch_size = 32
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.003, weight_decay=0.002)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=stack)
    eval_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=False, collate_fn=stack)
    train_losses = []
    eval_losses = []
    best_eval_loss = float('inf')
    epoch = 0
    for _ in tqdm(range(num_epochs), desc="Epoch", position=0, disable=not show_progress):
        network.train()
        epoch_train_losses = []
        for data, known_arrays, targets in tqdm(train_loader, desc="Minibatch", position=1, leave=False,
                                                disable=not show_progress):
            data = data.to(device)
            targets = targets.to(device)
            known_arrays = known_arrays.to(device)
            loss = training_step(network, optimizer, data, targets, known_arrays)
            epoch_train_losses.append(loss)
        train_losses.append(torch.mean(torch.tensor(epoch_train_losses)))

        scheduler.step()

        network.eval()
        epoch_eval_losses = []
        for data, known_arrays, targets in eval_loader:
            data = data.to(device)
            targets = targets.to(device)
            known_arrays = known_arrays.to(device)
            loss = eval_step(network, data, targets, known_arrays)
            epoch_eval_losses.append(loss)
        eval_losses.append(torch.mean(torch.tensor(epoch_eval_losses)))
        print(f'Epoch {epoch}: --- Train Loss: {train_losses[-1]} --- Eval Loss: {eval_losses[-1]}')
        epoch += 1
        if eval_losses[-1] < best_eval_loss:
            best_eval_loss = eval_losses[-1]
            torch.save(network.state_dict(), "best_model.pt")
            print('best eval loss')
    return train_losses, eval_losses


def plot_losses(train_losses: list, eval_losses: list):
    fig, ax = plt.subplots()
    ax.plot(train_losses, label="Train loss")
    ax.plot(eval_losses, label="Eval loss")
    ax.legend()
    ax.set_xlim(0, len(train_losses) - 1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    network = ResNet(n_in_channels=2, n_hidden_layers=9, n_kernels=32, kernel_size=3)
    train_dataset, val_dataset = get_dataset()
    train_losses, val_losses = training_loop(network, train_dataset, val_dataset, num_epochs=50, device=device,
                                             show_progress=False)
    plot_losses(train_losses, val_losses)

