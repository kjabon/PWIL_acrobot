"""Example running D4PG on continuous control tasks. Adapted from dm-acme example.
This file can (with some quick edits) either run D4PG on acrobot swingup, or instead

"""

from absl import flags
from acme.agents.jax import d4pg
import helpers
from absl import app
from acme.jax import experiments
from acme.utils import lp_utils
import launchpad as lp
from launchpad.nodes.python.local_multi_processing import PythonProcess
import launchpad.context
# import my_make_dist_expt as dist_expt
from acme.jax.experiments import config as configClass
import logging
import gpu
from typing import Sequence
import numpy as np
import jax.numpy as jnp
from acme import specs
import haiku as hk
from acme.jax import utils
from acme.jax import networks as networks_lib

terminals = ['gnome-terminal', 'gnome-terminal-tabs', 'xterm',
      'tmux_session', 'current_terminal', 'output_to_files']

seed = 0
num_steps = int(2e7)
evalEvery = int(num_steps/100)
trainDistributed = True
trainFresh = True


def build_experiment_config(task, plant = None, finetuneFromPWIL=False):
  """Builds D4PG experiment config which can be executed in different ways."""

  tensorboardDir = task + '/D4PG_finetune_both_from_critic_only/'

  #Note, this will overwrite checkpoints from PWIL, so make sure a copy is saved!
  if finetune:
        checkpointDir = '/home/{}/acme/'.format(uname) + '{}/PWIL_rewardfix'.format(task)
  else: 
        checkpointDir = '/home/{}/acme/.format(uname) + {}/D4PG_reward_fix'.format(task)

  vmax = 10.

  def make_networks(
      spec: specs.EnvironmentSpec,
      policy_layer_sizes: Sequence[int] = (256, 256, 256),
      critic_layer_sizes: Sequence[int] = (512, 512, 256),
      vmin: float = -vmax,
      vmax: float = vmax,
      num_atoms: int = 201,
  ) -> d4pg.D4PGNetworks:
    """Creates networks used by the agent."""

    action_spec = spec.actions

    num_dimensions = np.prod(action_spec.shape, dtype=int)
    critic_atoms = jnp.linspace(vmin, vmax, num_atoms)

    def _actor_fn(obs):
      network = hk.Sequential([
        utils.batch_concat,
        networks_lib.LayerNormMLP(list(policy_layer_sizes) + [num_dimensions]),
        networks_lib.TanhToSpec(action_spec),
      ])
      return network(obs)

    def _critic_fn(obs, action):
      network = hk.Sequential([
        utils.batch_concat,
        networks_lib.LayerNormMLP(layer_sizes=[*critic_layer_sizes, num_atoms]),
      ])
      value = network([obs, action])
      return value, critic_atoms

    policy = hk.without_apply_rng(hk.transform(_actor_fn))
    critic = hk.without_apply_rng(hk.transform(_critic_fn))

    # Create dummy observations and actions to create network parameters.
    dummy_action = utils.zeros_like(spec.actions)
    dummy_obs = utils.zeros_like(spec.observations)
    dummy_action = utils.add_batch_dim(dummy_action)
    dummy_obs = utils.add_batch_dim(dummy_obs)

    return d4pg.D4PGNetworks(
      policy_network=networks_lib.FeedForwardNetwork(
        lambda rng: policy.init(rng, dummy_obs), policy.apply),
      critic_network=networks_lib.FeedForwardNetwork(
        lambda rng: critic.init(rng, dummy_obs, dummy_action), critic.apply))

  # Configure the agent.
  d4pg_config = d4pg.D4PGConfig(learning_rate=3e-4, sigma=0.2,discount=0.9995)#, #min_replay_size=int(1e6-1))

  def make_logger(label, steps_key='learner_steps', i=0):
      # gpu.SetGPU(-1)
      from acme.utils.loggers.terminal import TerminalLogger
      from acme.utils.loggers.tf_summary import TFSummaryLogger
      from acme.utils.loggers import base, aggregators
      summaryDir = "tensorboardOutput/" + tensorboardDir
      terminal_logger = TerminalLogger(label=label, print_fn=logging.info)
      tb_logger = TFSummaryLogger(summaryDir, label=label, steps_key=steps_key)
      serialize_fn = base.to_numpy
      logger = aggregators.Dispatcher([tb_logger, terminal_logger], serialize_fn)
      return logger

  checkpointingConfig = configClass.CheckpointingConfig()
  checkpointingConfig.max_to_keep = 5
  checkpointingConfig.directory = checkpointDir
  checkpointingConfig.time_delta_minutes = 10
  checkpointingConfig.add_uid = False
  checkpointingConfig.replay_checkpointing_time_delta_minutes = 1

  return experiments.ExperimentConfig(
      builder=d4pg.D4PGBuilder(d4pg_config),
      environment_factory=lambda _: helpers.make_olympic_challenge_env(task,plant=plant),
      network_factory=make_networks,
      seed=seed,
      max_num_actor_steps=num_steps,
      logger_factory=make_logger,
      checkpointing=checkpointingConfig
  )

def launchDistributed(experiment_config, numActors=1,
                        numLearners=1, ckpt_path=None):
    program = experiments.make_distributed_experiment(
        experiment=experiment_config,
        num_actors=numActors, num_learner_nodes=numLearners)#, reverb_ckpt_path=ckpt_path)
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
            PythonProcess(env=dict(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES="0",
                                   XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_ALLOCATOR='platform')),
        'learner':
            PythonProcess(env=dict(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES="0",
                                   XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_ALLOCATOR='platform',
                                   XLA_PYTHON_CLIENT_MEM_FRACTION='.40', TF_FORCE_GPU_ALLOW_GROWTH='true')),
    }

    worker = lp.launch(program, xm_resources=lp_utils.make_xm_docker_resources(program),
                       launch_type=launchpad.context.LaunchType.LOCAL_MULTI_PROCESSING,
                       terminal=terminals[1], local_resources=resources)
    worker.wait()

def main(_):
  robot = 'acrobot'
  experiment_config = build_experiment_config(robot, finetuneFromPWIL=False)
  numLearners=8
  if trainDistributed:
      launchDistributed(experiment_config=experiment_config, numActors=16,
                        numLearners=numLearners, ckpt_path=None)
  else:
      import my_run_experiment as mre
      mre.run_experiment(experiment_config,
                                 eval_every=evalEvery)


if __name__ == '__main__':
    gpuNum = 0
    assert isinstance(gpuNum, int)
    gpu.SetGPU(gpuNum, True)
    app.run(main)
