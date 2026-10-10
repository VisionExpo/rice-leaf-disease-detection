# Rice Leaf Disease Detection

A deep learning and computer vision project that classifies rice leaf images into three disease categories using transfer learning and exposes predictions through a FastAPI REST API.

## Overview

Rice crops can be affected by multiple leaf diseases that impact crop health and productivity. This project explores image classification as a way to identify three rice leaf disease categories from photographs.

The project follows a machine learning workflow covering dataset exploration, preprocessing, model experimentation, evaluation, inference, and API development.

## Disease Classes

The model classifies images into the following categories:

1. **Bacterial Leaf Blight**
2. **Brown Spot**
3. **Leaf Smut**

## Dataset

The current dataset contains 119 JPG images distributed across three classes.

| Disease class | Images |
|---|---:|
| Bacterial Leaf Blight | 40 |
| Brown Spot | 40 |
| Leaf Smut | 39 |
| **Total** | **119** |

The initial dataset description expected 120 images. The available files were inspected, and 119 valid images were identified. No images were artificially added to balance the classes.

The dataset is divided using a stratified split:

| Split | Images |
|---|---:|
| Training | 83 |
| Validation | 18 |
| Testing | 18 |
| **Total** | **119** |

The split manifest is stored in `data/split_manifest.csv`. The raw and processed datasets are excluded from Git.

## Project Workflow

1. **Data exploration:** Inspect image dimensions, aspect ratios, colour channels, class distributions, image quality, and duplicate files.
2. **Preprocessing:** Convert images to RGB, preserve their aspect ratios through padding, resize them to the target dimensions, and normalize pixel values.
3. **Baseline model:** Train a Logistic Regression classifier using flattened image pixels.
4. **CNN experimentation:** Develop a convolutional neural network and evaluate an augmentation-based experiment.
5. **Transfer learning:** Evaluate MobileNetV2 and compare frozen-feature extraction with fine-tuning.
6. **Model evaluation:** Use validation data for model selection and a held-out test set for final evaluation.
7. **Inference:** Load the saved model and return the predicted class, confidence, and class probabilities.
8. **API development:** Serve predictions through FastAPI and validate uploaded files.
9. **Testing:** Use automated tests to check preprocessing, inference, and API behaviour.

## Model and Evaluation

The selected final candidate uses **MobileNetV2 transfer learning**.

The final clean-run candidate achieved the following held-out test performance:

| Metric | Result |
|---|---:|
| Test images | 18 |
| Correct predictions | 12 |
| Incorrect predictions | 6 |
| Test accuracy | 66.67% |
| Macro F1-score | 0.6698 |

The dataset is small, so these results should be treated as preliminary. Further evaluation on a larger, more diverse dataset is needed before considering real-world deployment.

## Technology Stack

- **Language:** Python
- **Deep learning:** TensorFlow and Keras
- **Computer vision:** Pillow
- **Machine learning:** scikit-learn
- **Data processing:** NumPy and Pandas
- **API:** FastAPI and Uvicorn
- **Testing:** pytest and HTTPX-compatible test client
- **Environment and dependency management:** uv
- **Version control:** Git and GitHub

## Project Structure

```text
rice-leaf-disease-detection/
├── api/
│   └── main.py
├── app/
├── data/
│   ├── raw/
│   ├── processed/
│   └── split_manifest.csv
├── models/
├── notebooks/
│   └── 01_data_understanding.ipynb
├── src/
│   ├── data/
│   ├── preprocessing/
│   ├── models/
│   └── inference/
├── tests/
│   ├── test_preprocessing.py
│   ├── test_inference.py
│   └── test_api.py
├── .github/
├── pyproject.toml
├── uv.lock
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.12
- uv package manager
- Git

### Installation

Clone the repository and enter the project directory:

```bash
git clone https://github.com/VisionExpo/rice-leaf-disease-detection.git
cd rice-leaf-disease-detection
```

Install the project dependencies:

```bash
uv sync
```

### Model Files

The trained model artifacts are excluded from Git. Before using the prediction endpoint, ensure that these files exist locally:

```text
models/
├── mobilenetv2_final.keras
└── class_names.json
```

The model and class-name file must correspond to the model expected by the inference module.

## Running the API

Start the FastAPI development server:

```bash
uv run uvicorn api.main:app --reload
```

The API will be available at:

`http://127.0.0.1:8000`

### Interactive API Documentation

Open the Swagger UI in your browser:

`http://127.0.0.1:8000/docs`

You can use this interface to inspect endpoints and upload an image for prediction.

## API Endpoints

### 1. Health Check

**Endpoint:** `GET /health`

Checks whether the API is responding.

Example response:

```json
{
  "status": "healthy",
  "service": "rice-leaf-disease-detection"
}
```

### 2. Predict Rice Leaf Disease

**Endpoint:** `POST /predict`

Accepts a leaf image as a multipart file upload and returns a predicted class, confidence score, and probability for each class.

**Supported formats:** JPG, JPEG, PNG

**Maximum upload size:** 10 MB

The required multipart form field is `file`.

Example request using curl:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@rice_leaf.jpg"
```

Example response:

```json
{
  "predicted_class": "Leaf smut",
  "confidence": 0.6595,
  "probabilities": {
    "Bacterial leaf blight": 0.2543,
    "Brown spot": 0.0862,
    "Leaf smut": 0.6595
  }
}
```

The values above illustrate the response format; actual predictions and confidence scores vary by image.

### Common HTTP Errors

| Status code | Meaning |
|---|---|
| `400` | Unsupported file extension, empty upload, or invalid image |
| `413` | Uploaded file exceeds the size limit |
| `422` | Required upload field is missing or request validation fails |
| `500` | An unexpected server-side prediction error occurs |

## Running Tests

Run the automated test suite:

```bash
uv run pytest
```

The suite covers image preprocessing, inference behaviour, API health checks, valid predictions, invalid uploads, oversized files, missing files, and inference failures.

## Limitations and Future Improvements

- The dataset contains only 119 images, limiting the reliability and generalization of the model.
- The held-out test set contains only 18 images.
- Background, lighting, leaf position, and image dimensions vary across the dataset.
- More labelled images and evaluation on independent datasets are needed.
- Future work can explore additional augmentation strategies, hyperparameter tuning, more robust evaluation, and improved deployment practices.

## Disclaimer

This is an educational machine learning project. Predictions should not be treated as a definitive agricultural diagnosis without additional expert validation.

## Author

**Vishal Gorule**

GitHub: [VisionExpo](https://github.com/VisionExpo)
