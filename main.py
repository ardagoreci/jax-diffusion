"""
Main file for running the training and evaluation loops of the diffusion model.

The majority of the logic is in libraries that can be easily imported
and tested in Colab.
"""

# Define flags for workdir and config file

# Within main:
# 1. Prevent TF from grabbing GPU memory
# 2. Logging
# 3. platform
# 4. train.train_and_evaluate
