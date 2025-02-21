from kfp.dsl import Input, component, Model
import sys

from config import *


@component(
    base_image=BASE_IMAGE,
    packages_to_install=["google-cloud-aiplatform"],
)
def deploy(
    image_path: str,
    saved_folder: Input[Model],
    project: str = "adalfi-ai-poc-merchandizing",
    region: str = "us-central1",
    machine_type: str = "n1-standard-2",
    min_replica_count: int = 1,
    max_replica_count: int = 2,
    accelerator_type: str = "NVIDIA_TESLA_T4",
    accelerator_count: int = 1,
):
    import logging
    from google.cloud import aiplatform

    aiplatform.init(project=project, location=region)
    logging.basicConfig(level=logging.DEBUG)

    CONTAINER_IMAGE_URI = image_path

    existing_model = aiplatform.Model.list(
        project="adalfi-ai-poc-merchandizing", location="us-central1"
    )

    if len(existing_model) != 0:
        existing_model = existing_model[0].resource_name
    else:
        existing_model = None

    uploaded_model = aiplatform.Model.upload(
        parent_model=existing_model,
        display_name="adalfi-ai-poc-merchandizing",
        artifact_uri=saved_folder.path,
        serving_container_image_uri=CONTAINER_IMAGE_URI,
        serving_container_predict_route="/predict",
        serving_container_health_route="/health",
    )

    # this will create a new endpoint, so the endpoint used for prediction will be the new one.
    # please check within the GCP console for the newest endpoint
    endpoint = aiplatform.Endpoint.create(
        display_name="adalfi-ai-poc-merchandizing-endpoint",
        project=project,
        location=region,
    )

    deploy_params = {
        "machine_type": machine_type,
        "min_replica_count": min_replica_count,
        "max_replica_count": max_replica_count,
        "traffic_split": {"0": 100},
        "deployed_model_display_name": "adalfi-ai-poc-merchandizing-deployment",
    }

    # Add accelerator config if specified
    if accelerator_type and accelerator_count > 0:
        deploy_params["accelerator_type"] = accelerator_type
        deploy_params["accelerator_count"] = accelerator_count

    endpoint = aiplatform.Endpoint.list(project=project, location=region)[0]
    uploaded_model.deploy(endpoint=endpoint, **deploy_params)

    # undeploy traffic = 0 models
    for model in endpoint.list_models():
        if model.id not in endpoint.traffic_split:
            endpoint.undeploy(deployed_model_id=model.id)
    logging.info(f"Model deployed successfully. Endpoint: {endpoint.resource_name}")
