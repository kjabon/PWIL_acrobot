"""Example running PWIL on continuous control tasks. Adapted from dm-acme example.

The network structure and hyperparameters are the same as the one used in the
PWIL paper: https://arxiv.org/pdf/2006.04678.pdf.
"""

from typing import Sequence

import rlds.transformations
from absl import flags
from acme import specs,types
from acme.agents.jax import d4pg
from acme.agents.jax import pwil
from acme.datasets import tfds as tfds_acme
import tensorflow_datasets as tfds
import helpers
from absl import app
from acme.jax import experiments
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.utils import lp_utils
import tensorflow as tf
import dm_env
import haiku as hk
import jax.numpy as jnp
import launchpad as lp
import numpy as np
import gpu
from envlogger.backends import rlds_utils
from acme.jax.experiments import config as configClass
import logging
import my_make_dist_expt as dist_expt
from launchpad.nodes.python.local_multi_processing import PythonProcess
import launchpad.context

flags.DEFINE_integer('num_steps', 1_000_000,
                     'Number of environment steps to run for.')

FLAGS = flags.FLAGS
terminals = ['gnome-terminal', 'gnome-terminal-tabs', 'xterm',
      'tmux_session', 'current_terminal', 'output_to_files']

uname = 'YOUR_USERNAME_HERE' #TODO

seed = 0
run_distributed = True
trainFresh = False
evaluation_episodes = 10

def make_networks(
    spec: specs.EnvironmentSpec,
    policy_layer_sizes: Sequence[int] = (256, 256, 256),
    critic_layer_sizes: Sequence[int] = (512, 512, 256),
    vmin: float = -100,
    vmax: float = 100,
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


def build_experiment_config(env_name,eval=False,plant=None) -> experiments.ExperimentConfig:
  """Returns a configuration for PWIL experiments."""

  # Create an environment, grab the spec, and use it to create networks.
  # env_name = FLAGS.env_name
 
  tensorboardDir = env_name + '/PWIL_rewardfix'
  checkpointDir = '/home/{}/acme/'.format(uname) + '{}/PWIL_rewardfix'.format(env_name)


  def environment_factory(seed: int) -> dm_env.Environment:
    del seed
    return helpers.make_olympic_challenge_env(robot=env_name,eval=eval,plant=plant)

  # Create d4pg agent
  d4pg_config = d4pg.D4PGConfig(
      learning_rate=5e-5, sigma=0.2, samples_per_insert=256)
  d4pg_builder = d4pg.D4PGBuilder(config=d4pg_config)


  num_datasets_to_use = 1  # keep it low, increase slightly. 15000 would already be 10 demos, so you have ~66 in 1 already
  # 100k for 10s would be 20. 
  
  # the source datasets are made with runController.py, which records ILQR transitions with envlogger
  # convert to PWIL-friendly transitions
  def make_demonstrations():
    ds = None
    j = 0
    for i in range(32):
      if num_datasets_to_use == 0:
        break
      try:
        recover_dataset_path = './ilqr_env_logs/{}'.format(i)
        builder = tfds.builder_from_directory(recover_dataset_path)
        builder = rlds_utils.maybe_recover_last_shard(builder)
        currDs = builder.as_dataset(split='train')

        try:
          ds = ds.concatenate(currDs)
        except:
          print("First working dataset")
          ds = currDs
        j+=1
        if j >= num_datasets_to_use:
          break

      except Exception as e:
        print('{} failed, error: '.format(i), end='')
        print(e)
    j=0
    for i in range(32):
      if num_stabi_ds_to_use == 0:
        break
      try:
        recover_dataset_path = './ilqr_env_logs/{}'.format(i+32)
        builder = tfds.builder_from_directory(recover_dataset_path)
        builder = rlds_utils.maybe_recover_last_shard(builder)
        currDs = builder.as_dataset(split='train')

        try:
          ds = ds.concatenate(currDs)
        except:
          print("First working dataset")
          ds = currDs
        j+=1
        if j >= num_stabi_ds_to_use:
          break

      except Exception as e:
        print('{} failed, error: '.format(i), end='')
        print(e)
        
    def batch_steps(episode):
      return rlds.transformations.batch(
        episode[rlds.STEPS], size=2, shift=1, drop_remainder=True)

    def _batched_step_to_transition(step: rlds.BatchedStep) -> types.Transition:
      return types.Transition(
        observation=tf.nest.map_structure(lambda x: x[0], step[rlds.OBSERVATION]),
        action=tf.nest.map_structure(lambda x: x[0], step[rlds.ACTION]),
        reward=tf.nest.map_structure(lambda x: x[0], step[rlds.REWARD]),
        discount=1.0 - tf.cast(step[rlds.IS_TERMINAL][1], dtype=tf.float32),
        # If next step is terminal, then the observation may be arbitrary.
        next_observation=tf.nest.map_structure(
          lambda x: x[1], step[rlds.OBSERVATION])
      )
    batched_steps = ds.flat_map(batch_steps)
    transitions = rlds.transformations.map_steps(batched_steps,
                                          _batched_step_to_transition)
    demos = pwil.PWILDemonstrations(demonstrations=transitions, episode_length=5000)
    print('Constructed demonstrations')
    return demos

  # Construct PWIL agent
  pwil_config = pwil.PWILConfig(num_transitions_rb=0)
  pwil_builder = pwil.PWILBuilder(
      rl_agent=d4pg_builder,
      config=pwil_config,
      demonstrations_fn=make_demonstrations)

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
  num_steps = FLAGS.num_steps  
  eval_every = int(num_steps / 20)

  return experiments.ExperimentConfig(
      builder=pwil_builder,
      environment_factory=environment_factory,
      network_factory=make_networks,
      seed=seed,
      max_num_actor_steps=num_steps,
      logger_factory=make_logger,
      checkpointing=checkpointingConfig)

def launchDistributed(experiment_config, numActors=1,
                      numLearners=1, ckpt_path=None):
    program = dist_expt.make_distributed_experiment(
      experiment=experiment_config,
      num_actors=numActors, num_learner_nodes=numLearners, reverb_ckpt_path=ckpt_path)
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
        PythonProcess(env=dict(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES="0,1,2",
                               XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_ALLOCATOR='platform')),
      'learner':
        PythonProcess(env=dict(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES="0,1,2",
                               XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_ALLOCATOR='platform',
                               XLA_PYTHON_CLIENT_MEM_FRACTION='.40', TF_FORCE_GPU_ALLOW_GROWTH='true')),
    }

    worker = lp.launch(program, xm_resources=lp_utils.make_xm_docker_resources(program),
                       launch_type=launchpad.context.LaunchType.LOCAL_MULTI_PROCESSING,
                       terminal=terminals[1], local_resources=resources)
    worker.wait()

def main(_):
  robot = 'acrobot'

  config = build_experiment_config(robot, eval=False)
  numLearners = 8
  rvb_ckpt_path = None#'/tmp/reverbCheckpoints/{}/PWIL/'.format(env_name)
  if trainFresh:
    import os
    import shutil
    dirToRm = '/home/{}/acme/'.format(uname) + '{}/PWIL_envFix'.format(robot)
    if os.path.isdir(dirToRm):
      shutil.rmtree(dirToRm)

  num_steps = FLAGS.num_steps  # int(5000 * 2**5)
  eval_every = int(num_steps / 20)
  if run_distributed:
    launchDistributed(experiment_config=config, numActors=32,
                      numLearners=numLearners, ckpt_path=rvb_ckpt_path)
  else:
    experiments.run_experiment(
        experiment=config,
        eval_every=eval_every,
        num_eval_episodes=evaluation_episodes)


if __name__ == '__main__':
  gpu.SetGPU(1, True)
  app.run(main)
