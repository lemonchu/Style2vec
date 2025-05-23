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

def visualize_results(query_path, candidate_paths, probs, output_path):
    """
    Create an enhanced visualization using matplotlib:
    - Query image is displayed in one row at the top spanning all columns.
    - Candidate images are arranged in a grid with 3 images per row.
    - Each candidate image shows its matching probability (overlaid on the image).
    - The candidate with the highest probability is highlighted with a red border.
    """
    thumb_size = (192, 192)
    # Load query image
    try:
        query_img = Image.open(query_path).convert('RGB')
    except Exception as e:
        print(f"Failed to open query image for visualization: {e}", file=sys.stderr)
        return

    # Load candidate images
    candidate_imgs = []
    for path in candidate_paths:
        try:
            img = Image.open(path).convert('RGB')
            candidate_imgs.append(img)
        except Exception as e:
            print(f"Failed to open candidate image {path} for visualization: {e}", file=sys.stderr)
            candidate_imgs.append(Image.new('RGB', thumb_size, (128, 128, 128)))

    num_candidates = len(candidate_imgs)
    n_cols = 3
    n_rows = math.ceil(num_candidates / n_cols)

    # Determine the candidate with the highest probability
    max_idx = int(torch.argmax(probs).item())

    # Create figure using GridSpec: top row for query, remaining rows for candidates
    fig = plt.figure(figsize=(5, 5 + 1.75 * n_rows))
    gs = GridSpec(nrows=1 + n_rows, ncols=n_cols, height_ratios=[1] + [1] * n_rows)

    # Query image in the top row, spanning all columns
    ax_query = fig.add_subplot(gs[0, :])
    ax_query.imshow(query_img)
    ax_query.set_title("Query", fontsize=16)
    ax_query.axis('off')

    # Plot candidate images in grid format with probability text below the image
    for i, (cand_img, path) in enumerate(zip(candidate_imgs, candidate_paths)):
        row = 1 + i // n_cols
        col = i % n_cols
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(cand_img.resize(thumb_size))
        ax.set_title(f"Candidate {i+1}", fontsize=12)
        # Set probability text below the image using xlabel
        prob = probs[i].item()
        ax.set_xlabel(f"Prob: {prob:.4f}", fontsize=12)
        # Remove ticks but keep the spines visible
        ax.set_xticks([])
        ax.set_yticks([])
        # Add red border for the best matching candidate
        if i == max_idx:
            for spine in ax.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(3)

    plt.tight_layout()
    try:
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"Visualization saved to {output_path}")
    except Exception as e:
        print(f"Failed to save visualization: {e}", file=sys.stderr)

def main(args):
    model_path = args.model_path
    if not os.path.exists(model_path):
        print(f"Model file {model_path} does not exist", file=sys.stderr)
        return

    # Load the entire model with proper map_location
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()

    # Load and preprocess the query image
    if not os.path.exists(args.query):
        print(f"Query image {args.query} does not exist", file=sys.stderr)
        return
    query_tensor = load_image(args.query)
    if query_tensor is None:
        print("Failed to load query image.", file=sys.stderr)
        return
    # Create a batch of size 1 for the query image
    query_batch = query_tensor.unsqueeze(0).to(device)

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
        query_emb = model(query_batch).squeeze(0)         # shape: [embedding_dim]
        candidate_embs = model(candidate_batch)             # shape: [num_candidates, embedding_dim]

    # Compute dot products between query embedding and candidate embeddings
    scores = torch.matmul(candidate_embs, query_emb)
    # Apply softmax to get a probability distribution over candidate images
    probs = F.softmax(scores, dim=0)

    # 根据参数设置，终端输出不同格式的概率信息
    if args.prob_format == "human":
        print("Matching Distribution:")
        for path, prob in zip(valid_candidate_paths, probs):
            print(f"Image: {path} --> Probability: {prob.item():.4f}")
    elif args.prob_format == "list":
        print(probs.tolist())
    # 如果为 "none"，则不输出概率信息

    # 如果 visualization 被启用, 生成并保存可视化图片
    if args.visualize:
        output_path = args.output if args.output else "visualization.png"
        visualize_results(args.query, valid_candidate_paths, probs, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Match Prediction Script: Compare a query image against candidate images"
    )
    parser.add_argument("query", help="Path to the query image")
    # If candidate_file is provided, candidates from command line will be ignored.
    parser.add_argument("candidates", nargs="*", help="Paths to candidate images")
    parser.add_argument("--candidate_file", type=str, default=None,
                        help="Path to a file containing candidate image paths (one per line)")
    parser.add_argument("--model_path", type=str, default="./font_style2vec.pth",
                        help="Path to the model file")
    parser.add_argument("--visualize", action="store_true",
                        help="If set, generate a visualization image")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for the visualization image (default: visualization.png)")
    parser.add_argument("--prob_format", type=str, choices=["human", "list", "none"], default="human",
                        help="Probability output format: human, list, or none")
    args = parser.parse_args()
    main(args)