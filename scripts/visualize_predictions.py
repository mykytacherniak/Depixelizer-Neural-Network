import torch
import pickle
from architecture import ResNet
import matplotlib.pyplot as plt

with open('test_set.pkl', 'rb') as file:
    test_set = pickle.load(file)

pixelated_images = test_set['pixelated_images']
known_arrays = test_set['known_arrays']

network = ResNet(2, 9, 32, 3)
network.load_state_dict(torch.load("models/best_model.pt", map_location=torch.device('cpu')))
network.eval()

predictions = []
for image, known_array in zip(pixelated_images, known_arrays):

    image_tensor = torch.from_numpy(image).float()
    known_array_tensor = torch.from_numpy(known_array).bool()
    combined_array = torch.cat((image_tensor, known_array_tensor), 0)
    with torch.no_grad():
        output = network(combined_array)
    predictions.append((output.float() * ~known_array_tensor + image_tensor * known_array_tensor).cpu().numpy())

    for data, outputs in zip(pixelated_images, predictions):
        fig, axes = plt.subplots(ncols=2)
        axes[0].imshow(data[0], cmap="gray", vmin=0, vmax=255)
        axes[0].set_title("Pixelated Image")
        axes[1].imshow(outputs[0], cmap="gray", vmin=0, vmax=255)
        axes[1].set_title("Output")
        fig.tight_layout()
        plt.show()
print(len(predictions))
