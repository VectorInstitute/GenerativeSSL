""""Extract features from images using CLIP model."""
import argparse

import clip
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm


class CLIPFeatureExtractor:
    """Extract features from images using CLIP model."""

    def __init__(self, data_path, output_path, model, batch_size, num_workers, device):
        self.data_path = data_path
        self.output_path = output_path
        self.model = model
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device

        # Load the CLIP model
        self.model, self.preprocess = clip.load(self.model, device=self.device)

    def load_data(self):
        """Load the ImageNet dataset using the preprocessing pipeline."""
        # Load dataset with the preprocessing pipeline
        dataset = ImageFolder(root=self.data_path, transform=self.preprocess)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def extract_features(self):
        """Extract features from the images in the dataset."""
        data_loader = self.load_data()
        features = []

        for images, _ in tqdm(data_loader, desc="Extracting features"):
            with torch.no_grad():
                image_features = self.model.encode_image(images.to(self.device))
                features.append(image_features.cpu().numpy())

        return np.concatenate(features, axis=0)

    def save_features(self, features):
        """Save the extracted features to a numpy file."""
        np.save(self.output_path, features)
        print(f"Features saved to {self.output_path}")
        print(f"Shape of features: {features.shape}")

def main():
    """Extract features from ImageNet using CLIP."""
    parser = argparse.ArgumentParser(description="Extract Features from ImageNet using CLIP")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the ImageNet validation data")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the numpy file of extracted features")
    parser.add_argument("--model", type=str, default="ViT-B/32", help="CLIP model to use")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for processing images")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker processes for loading data")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (cuda or cpu)")

    args = parser.parse_args()

    # Initialize the FeatureExtractor
    extractor = CLIPFeatureExtractor(
        data_path=args.data_path,
        output_path=args.output_path,
        model=args.model,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device if torch.cuda.is_available() else "cpu",
    )

    # Process the extraction
    features = extractor.extract_features()
    extractor.save_features(features)

if __name__ == "__main__":
    main()
