import os
from enum import Enum
from uuid import uuid4

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from plumbum.cli import Application, Flag, ExistingFile, SwitchAttr
import wandb
import yaml

class Phase(str, Enum):
    AUTOENCODER = "autoencoder"
    POST_CLIP = "post_clip"


# "s3://cornell-mfd64/dvc/text2building_data_2/citydata/buildingnet/"


class DistributedTraining(Application):
    develop = Flag(
        ["-d", "--develop"],
        help="Run in development mode. Enable to keep instance alive after training when iterating. Disable to use spot instances.",
    )

    preload_config = Flag(
        ["-p", "--preload-config"],
        help="Preload configuration file and send its contents as a hyperparameter dict, instead of sending the file itself.",
    )

    def main(self, phase: Phase, config_path: ExistingFile, remote_data_location: str):
        config_path = os.path.relpath(config_path)

        wandb.login()
        settings = wandb.setup().settings
        current_api_key = wandb.wandb_lib.apikey.api_key(settings=settings)

        boto3_session = boto3.Session(region_name="us-east-1")
        session = sagemaker.Session(boto_session=boto3_session)

        # This job name is used as prefix to the sagemaker training job. Makes it easy
        # for your look for your training job in SageMaker Training job console.
        job_name = "sm-full-1"  # TODO: Generate using same code as W&B
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

        if self.develop:
            print("Running in development mode. Keeping instance alive after training.")
            runtime_args = {
                # Keep alive instance after starting, for faster restarts during iteration
                "keep_alive_period_in_seconds": 1000,
            }
        else:
            print("Using spot instances.")
            runtime_args = {
                "use_spot_instances": True,  # Use Managed Spot Training
                # max_wait must be larger than 24 hours, the default max runtime,
                "max_wait": 24 * 60 * 60 + 1,
            }

        if self.preload_config:
            hyperparameters = {
                **yaml.safe_load(open(config_path, encoding="utf-8")),
                "dataset_path": "/opt/ml/input/data/train",  # Hardcode dataset_path to where the S3 data will be mounted.
            }
        else:
            hyperparameters = {
                "config": config_path,  # Default params
                "data.dataset_path": "/opt/ml/input/data/train",  # Hardcode dataset_path to where the S3 data will be mounted.
            }

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
                "PHASE": phase.value,
                "WANDB_API_KEY": current_api_key,
                # Allow W&B to resume the current run if Spot interrupts and restarts us on a different machine.
                "WANDB_RESUME": "allow",
                "WANDB_RUN_ID": wandb.util.generate_id(),
            },
            input_mode="FastFile",
            checkpoint_s3_uri=autoresume_checkpoint_s3_uri,  # S3 location to automatically load and store checkpoints from
            hyperparameters=hyperparameters,
            **runtime_args,
        )

        estimator.fit({"train": remote_data_location})


if __name__ == "__main__":
    DistributedTraining.run()
