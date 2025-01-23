# Model Settings
MODEL = "yolo11m.pt"
BATCH_SIZE = 16
EPOCHS = 1000
DATASET_PATH = "datasets"
SAVED_PATH = f"{MODEL.split('/')[-1][: -3]}_batch_{BATCH_SIZE}_epochs_{EPOCHS}"

# Pipeline configuration
BASE_IMAGE = "us-central1-docker.pkg.dev/adalfi-ai-poc-merchandizing/component-container/libs:latest"
PREDICTION_IMAGE = "us-central1-docker.pkg.dev/adalfi-ai-poc-merchandizing/prediction-container/fastapi:latest"
PIPELINE_ROOT = "gs://adalfi_saved_models/pipelines"
PIPELINE_NAME = ""
LOCATION = "us-central1"

# GCS paths
GCS_DATA_FOLDER = "gs://tridorian-adalfi-poc-merchandizing/ultralytics-dataset"
MODEL_BUCKET = "gs://adalfi_saved_models"

PROJECT_ID = "adalfi-ai-poc-merchandizing"

# Container registry configuration
REGION = "us-central1"
