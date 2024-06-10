import torch
import numpy as np
import pickle
from architecture import ResNet

with open('test_set.pkl', 'rb') as file:
    test_set = pickle.load(file)

pixelated_images = test_set['pixelated_images']
known_arrays = test_set['known_arrays']

network = ResNet(2, 9, 32, 3)
network.load_state_dict(torch.load("best_model.pt", map_location=torch.device('cpu')))

# Count the total number of parameters
total_params = sum(p.numel() for p in network.parameters())
print("Total Parameters:", total_params)

network.eval()
predictions = []
for image, known_array in zip(pixelated_images, known_arrays):
    image_tensor = torch.from_numpy(image).float()
    known_array_tensor = torch.from_numpy(known_array).bool()
    combined_array = torch.cat((image_tensor, known_array_tensor), 0)
    with torch.no_grad():
        output = network(combined_array)
    flattened_prediction = torch.masked_select(output, ~known_array_tensor)
    prediction = flattened_prediction.cpu().numpy().astype(np.uint8)
    predictions.append(prediction)

print(predictions)
