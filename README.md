# PWIL_acrobot
A case study in imitation learning with PWIL, applied to an Acrobot swingup task.

This repo supports this [blog post](https://kjabon.github.io/blog/2023/ImitationLearningPWILAcrobot/).

# Installation
Aside from this code, from github, you need to install deepmind/acme, and [the acrobot env](https://github.com/dfki-ric-underactuated-lab/double_pendulum).
Recommended to install google/jax with GPU support, as well. 

# Usage
The flow to replicate results from the blog looks like this:
- runD4PG.py to train a D4PG agent on acrobot-swingup (view training results in Tensorboard)
- run_and_plot_policy.py to visualize the results
- runController.py to generate and log ILQR transitions
- runPWIL.py to imitation-learn from these transitions
- run_and_plot_policy.py to visualize results
- optionally, attempt to fine-tune the PWIL policy with runD4PG.py and setting the finetuneFromPWIL parameter to True. 

# Troubleshooting
If you have issues installing (e.g. dependency conflicts), you may want to try installing the env code, but instead only running acme from a local folder without installing, and installing only those dependencies which you need.
