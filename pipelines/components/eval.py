from kfp.dsl import (
    Dataset,
    Input,
    Output,
    component,
    Model,
    Metrics,
    InputPath,
    OutputPath,
)
import sys
from config import *

from google_cloud_pipeline_components.v1.custom_job import (
    create_custom_training_job_from_component,
)


@component(
    base_image=BASE_IMAGE,
)
def evaluate(
    dataset_path: str,
    gcs_data_folder: str,
    saved_models_gcs_folder: Input[Model],
    map50: Output[Metrics],
    map50_95: Output[Metrics],
    precision: Output[Metrics],
    recall: Output[Metrics],
    fitness: Output[Metrics],
    deploy: OutputPath(str),
):
    from ultralytics import YOLO
    from google.cloud import storage
    import os
    import torch

    print(f"CUDA available: {torch.cuda.is_available()}")

    # os.system(f"gcloud storage cp -r {GCS_DATA_FOLDER} ./{DATASET_PATH}")
    # load the model weights
    storage_client = storage.Client()
    path = saved_models_gcs_folder.path + "/train/weights/best.pt"
    # path = saved_models_gcs_folder.path + "/train/weights/best.engine"
    bucket = storage_client.get_bucket(path.split("/")[2])
    blobs = list(bucket.list_blobs(prefix="/".join(path.split("/")[3:])))

    for blob in blobs:
        filename = blob.name.split("/")[-1]
        try:
            blob.download_to_filename(f"{filename}")
        except Exception as e:
            print(f"error at: {e}")
            continue

    # load val datasets
    path = gcs_data_folder
    bucket = storage_client.get_bucket(path.split("/")[2])
    blobs = list(bucket.list_blobs(prefix="/".join(path.split("/")[3:])))

    for blob in blobs:
        dirname = "/".join(os.path.dirname(blob.name).split("/")[1:])
        if dirname.split("/")[0] == "train":
            continue
        os.makedirs(f"{dataset_path}/{dirname}", exist_ok=True)
        filename = blob.name.split("/")[-1]
        try:
            blob.download_to_filename(f"{dataset_path}/{dirname}/{filename}")
        except Exception as e:
            continue

    model = YOLO("best.pt")
    # model = YOLO("best.engine", task="detect")

    validation_results = model.val(
        project=f"{saved_models_gcs_folder}/val",
        device="0",
        data=f"{dataset_path}/detection.data.yaml",
    )
    results_dict = validation_results.results_dict

    map50_result = results_dict["metrics/mAP50(B)"]
    map50_95_result = results_dict["metrics/mAP50-95(B)"]

    map50.log_metric(
        "mAP50",
        map50_result,
    )
    map50_95.log_metric(
        "mAP50-95",
        map50_95_result,
    )
    precision.log_metric(
        "precision",
        results_dict["metrics/precision(B)"],
    )
    recall.log_metric(
        "recall",
        results_dict["metrics/recall(B)"],
    )
    fitness.log_metric(
        "fitness",
        results_dict["fitness"],
    )
    if map50_result > 0.85 and map50_95_result > 0.5:
        with open(deploy, "w") as output_file:
            output_file.write("true")
    else:
        with open(deploy, "w") as output_file:
            output_file.write("false")


evaluate_wrapper = create_custom_training_job_from_component(
    evaluate,
    display_name="evaluate",
    accelerator_type="NVIDIA_TESLA_T4",
)
