import torch
import os
import numpy as np
from datasets.crowd import Crowd
from models.vgg import vgg19
import argparse
from torchvision import transforms

# os.environ['TORCH_HOME'] = '/scratch/vaibhav'
args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Test with TTA')
    parser.add_argument('--data-dir', default='/scratch/vaibhav/Bayesian-Crowd-Counting/datasetsprocessed',
                        help='test data directory')
    parser.add_argument('--save-dir', default='/scratch/vaibhav/Bayesian-Crowd-Counting/logs/1013-145837',
                        help='model directory')
    parser.add_argument('--device', default='0', help='assign device')
    args = parser.parse_args()
    return args

# Define TTA transformations
def apply_tta_transformations(image):
    transformations = [
        image,  # Original
        torch.flip(image, dims=[2]),  # Horizontal flip
        torch.flip(image, dims=[3]),  # Vertical flip
        torch.rot90(image, k=1, dims=[2, 3]),  # Rotate 90 degrees
        torch.rot90(image, k=2, dims=[2, 3]),  # Rotate 180 degrees
        torch.rot90(image, k=3, dims=[2, 3])   # Rotate 270 degrees
    ]
    return transformations

# Apply inverse transformations to align density maps with the original orientation
def inverse_tta_transformations(density_map, index):
    if index == 1:  # Horizontal flip
        return torch.flip(density_map, dims=[2])
    elif index == 2:  # Vertical flip
        return torch.flip(density_map, dims=[3])
    elif index == 3:  # Rotate 90 degrees
        return torch.rot90(density_map, k=3, dims=[2, 3])
    elif index == 4:  # Rotate 180 degrees
        return torch.rot90(density_map, k=2, dims=[2, 3])
    elif index == 5:  # Rotate 270 degrees
        return torch.rot90(density_map, k=1, dims=[2, 3])
    return density_map  # No transformation for the original

if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set visible GPU

    # Load test dataset
    datasets = Crowd(os.path.join(args.data_dir, 'test'), 512, 8, is_gray=False, method='val')
    dataloader = torch.utils.data.DataLoader(datasets, batch_size=1, shuffle=False,
                                             num_workers=4, pin_memory=False)

    # Load model
    model = vgg19()
    device = torch.device('cuda')
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pth'), map_location=device))
    model.eval()  # Set model to evaluation mode

    epoch_minus = []

    # TTA on each image in the dataloader
    for inputs, count, name in dataloader:
        inputs = inputs.to(device)
        assert inputs.size(0) == 1, 'Batch size should be 1 for TTA'

        # Generate density maps for each transformation and average them
        density_maps = []
        transformations = apply_tta_transformations(inputs)
        with torch.no_grad():
            for i, transformed_input in enumerate(transformations):
                # Run the model on the transformed input
                transformed_output = model(transformed_input)
                # Apply inverse transformation to align the output with the original
                aligned_output = inverse_tta_transformations(transformed_output, i)
                density_maps.append(aligned_output)

        # Average the density maps to get the final density map
        final_density_map = torch.mean(torch.stack(density_maps), dim=0)

        # Calculate the difference from ground truth
        temp_minu = count[0].item() - torch.sum(final_density_map).item()
        print(name, temp_minu, count[0].item(), torch.sum(final_density_map).item())
        epoch_minus.append(temp_minu)

    # Calculate final metrics
    epoch_minus = np.array(epoch_minus)
    mse = np.sqrt(np.mean(np.square(epoch_minus)))
    mae = np.mean(np.abs(epoch_minus))
    log_str = 'Final Test with TTA: MAE {}, MSE {}'.format(mae, mse)
    print(log_str)
