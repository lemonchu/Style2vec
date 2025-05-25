# Style2Vec

Before exploring this project, ensure that all required dependencies are installed:

```bash
pip install -r requirements.txt
```

## Training

After preparing your own dataset, you can train the model using one of the following commands:

- **Basic Training:**

  ```bash
  python train_style2vec.py
  ```

  This command initiates the training process using the basic Style2Vec model on your dataset.

- **Cosine Similarity Training:**

  ```bash
  python train_style2vec_dot.py
  ```

  Use this command to train the model with cosine similarity, which is useful for certain types of similarity searches.

## Inference

You can run the following command to obtain the embedding of a handwritten sample:

```bash
python inference.py path/to/sample1.png path/to/sample2.png --model_path path/to/model
```

## Demo

From the main folder, you can run one of the following commands to see the demo in action:

- **Basic Prediction Demo (with human-readable probabilities):**

  ```bash
  python match_prediction_demo.py sample/test/font_0_sample_0.png --candidate_file demo/demo_candidates.txt --visualize --output demo/demo.png --prob_format human
  ```

  This command runs a demo that computes predictions using the basic model and displays probabilities in a human-friendly format.

- **Cosine Similarity Prediction Demo:**

  ```bash
  python match_prediction_demo_dot.py sample/test/font_0_sample_0.png --candidate_file demo/demo_candidates.txt --visualize --output demo/demo.png --prob_format none
  ```

  This variant uses cosine similarity for prediction, outputting results without human-readable probability formats.

- **Heatmap Demo:**

  ```bash
  python heatmap_prediction_demo_dot.py --candidate_file demo/demo_candidates_heatmap.txt --output demo/demo_heatmap.png
  ```

  With this command, the demo generates a heatmap visualization based on the candidate file, illustrating the prediction probability distribution.
