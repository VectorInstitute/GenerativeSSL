import torch
from torchvision import transforms
from PIL import Image
import os
import glob
from config import get_config
from rcdm_inference import RCDMInference

def load_image_batch(directory, batch_size, image_size):
    """
    Load a batch of images from a directory.

    Args:
        directory (str): Path to the directory containing images.
        batch_size (int): Number of images to include in the batch.
        image_size (int): Size to which images should be resized.

    Returns:
        torch.Tensor: A batch of images.
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    images = []
    for filename in glob.glob(os.path.join(directory, '*.jpeg'))[:batch_size]:
        img = Image.open(filename).convert('RGB')
        img = transform(img)
        images.append(img)
    return torch.stack(images)


# Function to save a batch of images to disk
def save_images(image_tensors, out_dir, prefix='generated_image'):
    """
    Save a batch of image tensors to disk.

    Args:
        image_tensors (List[torch.Tensor]): List of image tensors to save.
        out_dir (str): Directory to save the images.
        prefix (str): Prefix for the saved image filenames.
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Transform to convert tensors to PIL images
    to_pil = transforms.ToPILImage()

    for i, image_tensor in enumerate(image_tensors):
        img = to_pil(image_tensor.cpu().numpy())
        img.save(os.path.join(out_dir, f"{prefix}_{i}.jpeg"), "JPEG")


def main():
    # Initialize the RCDMInference class
    config = get_config()
    rcdm_inference = RCDMInference(config)

    # Load a batch of images
    image_batch = load_image_batch('./test_sample/1', batch_size=1, image_size=config.image_size)

    # Run inference
    generated_images = rcdm_inference.run_inference(image_batch)

    # Save the generated images to disk
    save_images(generated_images, out_dir='./generated_images')

if __name__ == "__main__":
    main()