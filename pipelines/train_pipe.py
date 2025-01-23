from kfp import compiler
from google.cloud import aiplatform
from components import *
from config import *


@pipeline(
    name=f"adalfi_yolov11_pipe",
)
def train_pipeline():
    """YOLOV11 Object Detection Model Training"""
    # train_dummy_task = train_dummy()
    load_and_train_wrapper_task = load_and_train_wrapper(
        gcs_data_folder=GCS_DATA_FOLDER,
        dataset_path=DATASET_PATH,
        model_version=MODEL,
        model_bucket=MODEL_BUCKET,
        saved_path=SAVED_PATH,
        epochs=EPOCHS,
    )

    eval_task = evaluate_wrapper(
        dataset_path=DATASET_PATH,
        gcs_data_folder=GCS_DATA_FOLDER,
        saved_models_gcs_folder=load_and_train_wrapper_task.outputs["saved_folder"],
    )


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=train_pipeline,
        package_path=f"adalfi_yolov11_pipe.json",
    )

    aiplatform.init(
        project=PROJECT_ID,
        staging_bucket="/".join(PIPELINE_ROOT.split("/")[:-1]),
        location=LOCATION,
    )

    job = aiplatform.PipelineJob(
        display_name=f"adalfi_yolov11_pipe",
        template_path=f"adalfi_yolov11_pipe.json",
        pipeline_root=PIPELINE_ROOT,
        enable_caching=True,
    )
    job.run(
        sync=True,
        service_account="500278265408-compute@developer.gserviceaccount.com",
    )
