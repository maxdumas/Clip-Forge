"""
Sagemaker does not support entrypoints with subcommands, so we need to wrap the
CLI in a shim that injects the "fit" subcommand. The script serves as the
entrypoint for the container when running SageMaker Training Jobs.
"""

import sys

from train_autoencoder import main

if __name__ == "__main__":
    args = ["fit"] + sys.argv[1:].copy()
    sys.argv[1:] = []
    main(args)
