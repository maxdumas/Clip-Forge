import boto3
import sagemaker
import wandb
from sagemaker.pytorch import PyTorch
from uuid import uuid4


wandb.login()
settings = wandb.setup().settings
current_api_key = wandb.wandb_lib.apikey.api_key(settings=settings)

boto3_session = boto3.Session(region_name="us-east-1", profile_name="cornell")
session = sagemaker.Session(boto_session=boto3_session)


# This job name is used as prefix to the sagemaker training job. Makes it easy
# for your look for your training job in SageMaker Training job console.
job_name = "sm-full-1" # TODO: Generate using same code and W&B
print("Job name: ", job_name)

# This is the location that SageMaker will automatically store (and load)
# checkpoints from. We can use this to automatically resume training from an
# earlier checkpoint when using Spot instances.
checkpoint_bucket = session.default_bucket()
autoresume_checkpoint_prefix = f"{job_name}/{uuid4()}"
autoresume_checkpoint_s3_uri = (
    f"s3://{checkpoint_bucket}/{autoresume_checkpoint_prefix}"
)
print(f"Checkpoints resumable from {autoresume_checkpoint_s3_uri}")

estimator = PyTorch(
    base_job_name=job_name,
    source_dir=".",
    entry_point="sm_entrypoint.py",  #  the entry point that launches the training script with options
    role="arn:aws:iam::870747888580:role/SageMakerTrainingRole",
    sagemaker_session=session,
    framework_version="1.13.1",  # PyTorch version to use
    py_version="py39",  # Python version to use
    instance_count=1,  # Number of instances to launch
    instance_type="ml.g5.8xlarge",  # Instance type to launch
    debugger_hook_config=False,
    environment={
        "WANDB_API_KEY": current_api_key,
        # Allow W&B to resume the current run if Spot interrupts and restarts us on a different machine.
        "WANDB_RESUME": "allow",
        "WANDB_RUN_ID": wandb.util.generate_id(),
    },
    input_mode="FastFile",
    checkpoint_s3_uri=autoresume_checkpoint_s3_uri,  # S3 location to automatically load and store checkpoints from
    keep_alive_period_in_seconds=1000,  # Keep alive instance after starting, for faster restarts during iteration
    # use_spot_instances=True, # Use Managed Spot Training
    # max_wait=24*60*60+1, # Must be larger than 24 hours, the default max runtime,
    hyperparameters={
        "config": "autoencoder_params.yaml",  # Default params
        # "batch_size": "128",
        # "test_batch_size": "128",
        # "lr": "1.0964781961431852e-05" # Learned from the Pytorch Lightining LR Finder
        # "lr": "0.001",
        # "ckpt_path": "maxdumas/clip_forge_autoencoder/model-ntloo4e5:v11"
    },
)

estimator.fit(
    {"train": "s3://cornell-mfd64/dvc/text2building_data_2/citydata/buildingnet/"}
)
