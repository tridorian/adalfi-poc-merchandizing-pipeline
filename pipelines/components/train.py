from kfp.dsl import component, OutputPath, Output, Metrics, Model
import sys

from config import *
from google_cloud_pipeline_components.v1.custom_job import (
    create_custom_training_job_from_component,
)


@component(base_image=BASE_IMAGE, packages_to_install=["google-cloud-storage"])
def load_and_train(
    gcs_data_folder: str,
    dataset_path: str,
    model_version: str,
    model_bucket: str,
    saved_path: str,
    epochs: int,
    saved_folder: Output[Model],
):
    from ultralytics import YOLO
    import os
    from google.cloud import storage

    # os.system(f"gcloud storage cp -r {GCS_DATA_FOLDER} ./{DATASET_PATH}")
    storage_client = storage.Client()
    path = gcs_data_folder
    bucket = storage_client.get_bucket(path.split("/")[2])
    blobs = list(bucket.list_blobs(prefix="/".join(path.split("/")[3:])))

    for blob in blobs:
        dirname = "/".join(os.path.dirname(blob.name).split("/")[1:])
        os.makedirs(f"{dataset_path}/{dirname}", exist_ok=True)
        filename = blob.name.split("/")[-1]
        try:
            blob.download_to_filename(f"{dataset_path}/{dirname}/{filename}")
        except Exception as e:
            continue
    model = YOLO(model_version)  # load a pretrained model (recommended for training)

    train_out = os.path.dirname(os.path.dirname(saved_folder.path)) + f"/{saved_path}"
    os.makedirs(train_out, exist_ok=True)

    saved_folder.path = train_out
    # Train the model
    model.train(
        data=f"{dataset_path}/detection.data.yaml",
        epochs=epochs,  # default
        batch=16,
        imgsz=640,  # default
        device=0,
        project=train_out,
        exist_ok=True,
        patience=50,
    )


load_and_train_wrapper = create_custom_training_job_from_component(
    load_and_train,
    display_name="load_and_train",
    accelerator_type="NVIDIA_TESLA_T4",
)
