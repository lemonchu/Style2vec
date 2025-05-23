import argparse
import os
import sys
import torch
from torchvision import transforms
from PIL import Image

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define image preprocessing transformations
data_transforms = transforms.Compose([
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Duplicate single channel to 3 channels if necessary
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Apply ImageNet normalization
])

def load_image(image_path):
    """
    Load an image from the given path and preprocess it.
    Converts the image to grayscale ('L' mode) if needed.
    """
    try:
        # Open the image and convert to grayscale if required
        image = Image.open(image_path).convert('L')
    except Exception as e:
        print(f"Failed to load image {image_path}: {e}", file=sys.stderr)
        return None
    return data_transforms(image)

def main(args):
    model_path = args.model_path
    if not os.path.exists(model_path):
        print(f"Model file {model_path} does not exist", file=sys.stderr)
        return

    # Load the entire model with proper map_location
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()

    images = []
    valid_image_paths = []
    for img_path in args.images:
        if not os.path.exists(img_path):
            print(f"Image {img_path} does not exist", file=sys.stderr)
            continue
        img_tensor = load_image(img_path)
        if img_tensor is None:
            continue
        images.append(img_tensor)
        valid_image_paths.append(img_path)
        
    if len(images) == 0:
        print("No valid image inputs provided", file=sys.stderr)
        return

    # Stack the list of image tensors into a single batch tensor
    batch = torch.stack(images).to(device)
    
    # Obtain embeddings from the model without computing gradients
    with torch.no_grad():
        embeddings = model(batch)
    
    # Print the embeddings corresponding to each valid image path
    for path, vec in zip(valid_image_paths, embeddings):
        print(f"Image: {path}\nEmbedding: {vec.cpu().numpy()}\n")
        
    # Print the whole batch of embeddings
    print(f"Batch of embeddings: {embeddings.cpu().numpy()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script: outputs model embeddings for one or more images")
    parser.add_argument("images", nargs="+", help="Paths to one or more input images")
    parser.add_argument("--model_path", type=str, default="./font_identifier_model_epoch_7.pth", help="Path to the model file")
    args = parser.parse_args()
    main(args)