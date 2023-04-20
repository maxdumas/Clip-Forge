"""
Sagemaker does not support entrypoints with subcommands, so we need to wrap the
CLI in a shim that injects the "fit" subcommand. The script serves as the
entrypoint for the container when running SageMaker Training Jobs.
"""

import sys
import os

from clip_forge.train_autoencoder import main as autoencoder_main
from clip_forge.train_post_clip import main as post_clip_main

if __name__ == "__main__":
    if os.environ.get("PHASE") == "autoencoder":
        args = ["fit"] + sys.argv[1:].copy()
        sys.argv[1:] = []
        autoencoder_main(args)
    elif os.environ.get("PHASE") == "post_clip":
        post_clip_main()
