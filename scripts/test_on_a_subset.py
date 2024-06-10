import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from architecture import ResNet
from create_dataset import stack
import warnings
from create_dataset import RandomImagePixelationDataset


def test_step(network, data, targets, known_arrays):
    with torch.no_grad():
        output = network(data)
        loss = F.mse_loss(output.float() * ~known_arrays, targets.float() * ~known_arrays)
        return loss.item(), output


if __name__ == "__main__":
    device = 'cuda'
    device = torch.device(device)
    if "cuda" in device.type and not torch.cuda.is_available():
        warnings.warn("CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    torch.manual_seed(0)

    network = ResNet(2, 9, 32, 3)

    test_dataset = RandomImagePixelationDataset(
        r"data/resized_test",
        width_range=(4, 32),
        height_range=(4, 32),
        size_range=(4, 16)
    )

    network.load_state_dict(torch.load("models/best_model.pt", map_location=torch.device('cpu')))
    network.eval()
    test_loader = DataLoader(test_dataset, shuffle=False, collate_fn=stack)
    test_losses = []
    outputs_list = []
    targets_list = []
    for data, known_arrays, targets in test_loader:
        data = data.to(device)
        targets = targets.to(device)
        known_arrays = known_arrays.to(device)
        loss, output = test_step(network, data, targets, known_arrays)
        test_losses.append(loss)
        outputs_list.append((output.float() * ~known_arrays + data.float() * known_arrays).cpu().numpy())
        targets_list.append(targets.cpu().numpy())
        print(output.shape)

    avg_test_loss = torch.mean(torch.tensor(test_losses))
    print("Average Test Loss:", avg_test_loss.item())

    # Visualize the predictions
    for data, outputs, targets in zip(test_loader, outputs_list, targets_list):
        fig, axes = plt.subplots(ncols=3)
        data = data[0].cpu().numpy()
        axes[0].imshow(data[0][0], cmap="gray", vmin=0, vmax=255)
        axes[0].set_title("data")
        axes[1].imshow(outputs[0][0], cmap="gray", vmin=0, vmax=255)
        axes[1].set_title("outputs")
        axes[2].imshow(targets[0][0], cmap="gray", vmin=0, vmax=255)
        axes[2].set_title("target")
        fig.tight_layout()
        plt.show()