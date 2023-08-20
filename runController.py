"""Example running ilqr on continuous control tasks."""

from absl import flags
from acme.agents.jax import ppo
import helpers
from absl import app
from acme.jax import experiments
from acme.utils import lp_utils
import launchpad as lp
import gpu
from acme.jax.experiments import config as configClass
import logging
import time
import os
import shutil
from launchpad.nodes.python.local_multi_processing import PythonProcess
import launchpad.context
import sys
from run_controller_experiment import run_experiment

import ilqrBuilder

terminals = ['gnome-terminal', 'gnome-terminal-tabs', 'xterm',
      'tmux_session', 'current_terminal', 'output_to_files']

seed = 0
num_steps = int(1e6)
evalEvery = int(num_steps/100)
trainDistributed = True
trainFresh = True
flags.DEFINE_bool('stabi', False,'')

FLAGS = flags.FLAGS


def build_experiment_config(robot):
  """Builds ilqr experiment config which can be executed in different ways."""
  # Create an environment, grab the spec, and use it to create networks.
  # suite, task = env_name.split(':', 1)
  tensorboardDir = robot + "/_ilqr_expt"

  builder = ilqrBuilder.ILQRBuilder(stabi=FLAGS.stabi)

  # I think the rest is fine as is
  def make_logger(label, steps_key='learner_steps', i=0):
      from acme.utils.loggers.terminal import TerminalLogger
      from acme.utils.loggers.tf_summary import TFSummaryLogger
      from acme.utils.loggers import base, aggregators
      summaryDir = "tensorboardOutput/" +  tensorboardDir
      terminal_logger = TerminalLogger(label=label, print_fn=logging.info)
      tb_logger = TFSummaryLogger(summaryDir, label=label, steps_key=steps_key)

      serialize_fn = base.to_numpy
      logger = aggregators.Dispatcher([terminal_logger, tb_logger], serialize_fn)
      return logger

  layer_sizes = (256, 256, 256)
  stabi = FLAGS.stabi
  def env_factory(envNum):
    return lambda seed: helpers.make_olympic_challenge_env(robot,envLogging=True,eval=not stabi, envNum=envNum,stabi=stabi)


  return experiments.ExperimentConfig(
      builder=builder,
      environment_factory=env_factory,#lambda seed: helpers.make_olympic_challenge_env(robot,envLogging=True),
      network_factory=lambda spec: ppo.make_networks(spec, layer_sizes),
      seed=seed,
      max_num_actor_steps=num_steps,
      logger_factory = make_logger,
      checkpointing=None
  )

def launchDistributed(experiment_config,numActors=1,numLearners=1):

    print("______________________________________________")
    print("numactors: {}, numlearners: {}".format(numActors, numLearners))
    print("______________________________________________")
    time.sleep(2)
    import my_make_controller_dist_expt as mmcde
    program = mmcde.make_distributed_experiment(
        experiment=experiment_config,
        num_actors=numActors, num_learner_nodes=numLearners, stabi=FLAGS.stabi)
    resources = {
        # The 'actor' and 'evaluator' keys refer to
        # the Launchpad resource groups created by Program.group()
        'actor':
            PythonProcess(  # Dataclass used to specify env vars and args (flags) passed to the Python process
                env=dict(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='',
                         XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_ALLOCATOR='platform')),
        'evaluator':
            PythonProcess(env=dict(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='',
                                   XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_ALLOCATOR='platform')),
        'counter':
            PythonProcess(env=dict(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='',
                                   XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_ALLOCATOR='platform')),
        'replay':
            PythonProcess(env=dict(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='',
                                   XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_ALLOCATOR='platform')),
        'learner':
            PythonProcess(env=dict(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='',
                                   XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_ALLOCATOR='platform',
                                   XLA_PYTHON_CLIENT_MEM_FRACTION='.40', TF_FORCE_GPU_ALLOW_GROWTH='true')),
    }

    worker = lp.launch(program, xm_resources=lp_utils.make_xm_docker_resources(program),
              launch_type=launchpad.context.LaunchType.LOCAL_MULTI_PROCESSING,
              terminal=terminals[1], local_resources=resources)
    worker.wait()


def main(_):
    contGymEnvs = ["walker:run", "hopper:stand", "acrobot:swingup", "Ant-v4"]
    try:
        gymDex = int(sys.argv[1])
    except:
        gymDex = 3
    gymEnv = contGymEnvs[gymDex]
    env_name = "gym:" + gymEnv
    task = "acrobot:swingup"
    # env_name = "control:"+task
    env_name='acrobot'
    print("v____________________________________________v")
    print(env_name)
    print("^____________________________________________^")
    time.sleep(2)
    if trainFresh:
        dirToRm = '/home/kenny/acme/' + env_name + '/ilqr'
        if os.path.isdir(dirToRm):
            shutil.rmtree(dirToRm)
    experiment_config = build_experiment_config(env_name)
    if trainDistributed:
        launchDistributed(experiment_config=experiment_config, numActors=32, numLearners=1)
    else:
        run_experiment(experiment_config, num_eval_episodes=0)


if __name__ == '__main__':
    gpu.SetGPU(-1, True)
    app.run(main)
