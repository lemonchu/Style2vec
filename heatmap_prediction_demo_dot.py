import argparse
import os
import sys
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import math
from matplotlib.gridspec import GridSpec

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

def save_heatmap(scores, output_path):
    """
    Generate and save a heatmap visualization from the cosine similarity scores.
    """
    # Convert tensor to numpy array and detach from graph if necessary
    scores_np = scores.cpu().detach().numpy()
    plt.figure(figsize=(8, 6))
    plt.imshow(scores_np, cmap='Reds', interpolation='nearest')
    plt.title("Cosine Similarity Heatmap")
    plt.colorbar()
    plt.savefig(output_path)
    plt.close()

def main(args):
    model_path = args.model_path
    if not os.path.exists(model_path):
        print(f"Model file {model_path} does not exist", file=sys.stderr)
        return

    # Load the entire model with proper map_location
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()

    # Obtain candidate image paths either from the candidate file or command line list
    candidate_paths = []
    if args.candidate_file:
        if not os.path.exists(args.candidate_file):
            print(f"Candidate file {args.candidate_file} does not exist", file=sys.stderr)
            return
        with open(args.candidate_file, 'r') as f:
            for line in f:
                path = line.strip()
                if path:
                    candidate_paths.append(path)
    else:
        candidate_paths = args.candidates

    if len(candidate_paths) == 0:
        print("No candidate images provided", file=sys.stderr)
        return

    # Load and preprocess candidate images
    candidate_tensors = []
    valid_candidate_paths = []
    for candidate_path in candidate_paths:
        if not os.path.exists(candidate_path):
            print(f"Candidate image {candidate_path} does not exist", file=sys.stderr)
            continue
        tensor = load_image(candidate_path)
        if tensor is None:
            continue
        candidate_tensors.append(tensor)
        valid_candidate_paths.append(candidate_path)

    if len(candidate_tensors) == 0:
        print("No valid candidate images provided", file=sys.stderr)
        return

    candidate_batch = torch.stack(candidate_tensors).to(device)

    # Obtain embeddings from the model without computing gradients
    with torch.no_grad():
        candidate_embs = model(candidate_batch)             # shape: [num_candidates, embedding_dim]
        # 对向量进行 L2 归一化以计算 cosine 相似度
        candidate_embs = F.normalize(candidate_embs, p=2, dim=1)

    # 计算 cosine 相似度
    scores = torch.matmul(candidate_embs, candidate_embs.T)

    # 对 0 取 max
    scores = torch.clamp(scores, min=0)

    output_path = args.output if args.output is not None else "visualization.png"
    save_heatmap(scores, output_path)
    print(f"Heatmap saved to {output_path}")

    print("Cosine similarity scores:", scores)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Match Prediction Script: Compare a query image against candidate images"
    )
    parser.add_argument("candidates", nargs="*", help="Paths to candidate images")
    parser.add_argument("--candidate_file", type=str, default=None,
                        help="Path to a file containing candidate image paths (one per line)")
    parser.add_argument("--model_path", type=str, default="./font_style2vec_dot(aug).pth",
                        help="Path to the model file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for the visualization image (default: visualization.png)")
    args = parser.parse_args()
    main(args)