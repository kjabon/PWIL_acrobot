# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared helpers for rl_continuous experiments."""


from gym import wrappers as gymWrappers
import my_gym_wrapper
from time import time

import functools
import os

from absl import flags
from acme import specs
from acme import wrappers
# import my_atari_wrapper
from acme.agents.jax import dqn
from acme.jax import networks as networks_lib
from acme.jax import utils
import atari_py  # pylint:disable=unused-import
import dm_env
import gym
import haiku as hk
import jax.numpy as jnp

from timeit import default_timer



class Timer(object):
  def __init__(self, verbose=False):
    self.verbose = verbose
    self.timer = default_timer

  def __enter__(self):
    self.start = self.timer()
    return self

  def __exit__(self, *args):
    end = self.timer()
    self.elapsed_secs = end - self.start
    self.elapsed = self.elapsed_secs  # millisecs
    if self.verbose:
      print('elapsed time: %f s' % self.elapsed)


_VALID_TASK_SUITES = ('gym', 'control')


FLAGS = flags.FLAGS


def make_dqn_atari_network_from_pixels(
    environment_spec: specs.EnvironmentSpec) -> dqn.DQNNetworks:
  """Creates networks for training DQN on Atari."""
  def pixelNetwork(inputs):
    model = hk.Sequential([
        networks_lib.AtariTorso(),
        hk.nets.MLP([512, environment_spec.actions.num_values]),
    ])
    return model(inputs)
  return make_dqn_atari_network_helper(environment_spec, pixelNetwork)

def make_dqn_atari_network_from_ram(
      environment_spec: specs.EnvironmentSpec) -> dqn.DQNNetworks:
  """Creates networks for training DQN on Atari."""
  def ramNetwork(inputs):
    inputs = jnp.array(inputs,dtype=jnp.float32)
    if inputs.ndim == 1:
        if inputs.shape[0] == 1:
            inputs = jnp.reshape(inputs, (1, -1))
        else:
            inputs = jnp.reshape(inputs, (inputs.shape[0], -1))
    model = hk.Sequential([
        hk.nets.MLP([512, 512],activate_final=True),
        hk.nets.MLP([512, environment_spec.actions.num_values]),
    ])
    return model(inputs)
  return make_dqn_atari_network_helper(environment_spec, ramNetwork)


def make_dqn_atari_network_helper(
    environment_spec: specs.EnvironmentSpec,
    network_factory) -> dqn.DQNNetworks:
  """Creates networks for training DQN on Atari."""
  network_hk = hk.without_apply_rng(hk.transform(network_factory)) #I.e., without apply rng means a deterministic policy

  obsSpace = environment_spec.observations
  if isinstance(obsSpace, specs.DiscreteArray):
      dummy_obs = jnp.zeros((1,))
  else: dummy_obs = utils.zeros_like(obsSpace)
  dummy_obs = utils.add_batch_dim(dummy_obs)

  network = networks_lib.FeedForwardNetwork(
      init=lambda rng: network_hk.init(rng, dummy_obs), apply=network_hk.apply)
  typed_network = networks_lib.non_stochastic_network_to_typed(network)
  return dqn.DQNNetworks(policy_network=typed_network)

def make_olympic_challenge_env(robot, eval = False, plant = None, envLogging = False, envNum=0,stabi=False):
    import numpy as np

    import gym

    from double_pendulum_local.model.symbolic_plant import SymbolicDoublePendulum
    from double_pendulum_local.model.model_parameters import model_parameters
    from double_pendulum_local.simulation.simulation import Simulator
    from double_pendulum_local.simulation.gym_env import (
        CustomEnv,
        double_pendulum_dynamics_func,
    )
    if stabi:
      tf = 10.
    else:
      tf = 10.
    # model parameters
    # design = "design_A.0"
    # model = "model_2.0"
    design = "design_C.0"
    model = "model_3.0"
    radiusThresh = 0.2
    # robot = "acrobot"

    if robot == "pendubot":
        torque_limit = [6.0, 0.0]
    elif robot == "acrobot":
        torque_limit = [0.0, 6.0]
    else: torque_limit=None

    model_par_path = (
            "./double_pendulum_local/data/system_identification/identified_parameters/"
            + design
            + "/"
            + model
            + "/model_parameters.yml"
    )
    mpar = model_parameters(filepath=model_par_path)

    mpar.set_motor_inertia(0.0)
    mpar.set_damping([0.0, 0.0])
    mpar.set_cfric([0.0, 0.0])
    mpar.set_torque_limit(torque_limit)
    dt = 0.002
    integrator = "runge_kutta"
    state_representation = 3
    if plant is None:
        print("Making plant... ")
        with Timer(True) as t:
          plant = SymbolicDoublePendulum(model_pars=mpar)
    simulator = Simulator(plant=plant, stabi=stabi)
    process_noise_sigmas = [0.0, 0.0, 0.0, 0.0]
    meas_noise_sigmas = [0.0, 0.0, 0.05, 0.05]
    u_noise_sigmas = [0.01, 0.01]
    u_responsiveness = 1.0
    simulator.set_process_noise(process_noise_sigmas=process_noise_sigmas)
    simulator.set_measurement_parameters(meas_noise_sigmas=meas_noise_sigmas)
    simulator.set_motor_parameters(u_noise_sigmas=u_noise_sigmas,
                             u_responsiveness=u_responsiveness)

    # learning environment parameters

    obs_space = gym.spaces.Box(
        np.array([-1.0, -1.0, -1.0, -1.0,-1.0, -1.0, -1.0, -1.0]),
        np.array([1.0, 1.0, 1.0, 1.0,1.0, 1.0, 1.0, 1.0])
    )
    act_space = gym.spaces.Box(np.array([-1]), np.array([1]))


    dynamics_func = double_pendulum_dynamics_func(
        simulator=simulator,
        dt=dt,
        integrator=integrator,
        robot=robot,
        state_representation=state_representation,
    )



    def reward_func(observation, action):

        # angularObs = dynamics_func.unscale_state(observation[0:6])
        jointPos = plant.forward_kinematics(observation[0:2])
        eePos = np.array([jointPos[1][0], jointPos[1][1]])
        targetPos = np.array([0, plant.l[0]+plant.l[1]])
        distToTarget = np.linalg.norm(eePos-targetPos)


        def inTolerance(x, target, tolerance):
            # return x > target - tolerance and x < target + tolerance
            return abs(x-target) < tolerance


        if stabi:
          reward = 10
          reward -= (distToTarget # move to balance position
                  # + abs(observation[0])
                  # + abs(observation[1])
                  + abs(observation[2])  #deter high speeds
                  + abs(observation[3])  #deter high speeds
                  # + 0.1*observation[0] ** 2 #deter continuous spinning
                  # + 0.1*observation[1] ** 2 #deter continuous spinning
                  # + 0.1 * action[0] ** 2.0 #use a smaller torque when possible]
          )
          if eePos[1] < 0.45:
            reward -= 100
          # if not inTolerance(distToTarget, 0, radiusThresh) or eePos[1] < 0.45:
          #   reward -= 100
          # else:
          #   if inTolerance(observation[2], 0, 0.5) and inTolerance(observation[3], 0, 0.5):
          #     reward += 10
          # if eePos[1] > 0.45:
          #   reward += 10

        else:
          reward = 0
          if distToTarget < radiusThresh and eePos[1] > 0.45:
            reward += 10
            # if inTolerance(observation[2], 0, 0.5) and inTolerance(observation[3], 0, 0.5):
            #   reward += 10

          reward -=    (
                  5*distToTarget # move to balance position
                  + abs(observation[0])
                  + abs(observation[1])
                  + abs(observation[2])  #deter high speeds
                  + abs(observation[3])  #deter high speeds
                  # + 0.1*observation[0] ** 2 #deter continuous spinning
                  # + 0.1*observation[1] ** 2 #deter continuous spinning
                  + np.linalg.norm(action)  #use a smaller torque when possible
              )

        return reward /(tf/dt) #normalize by num timesteps

    def terminated_func(observation):
      # return False
      jointPos = plant.forward_kinematics(observation[0:2])


      if stabi:
        # return distToTarget > radiusThresh or eePos[1] < 0.45
        eePosY = jointPos[1][1]
        return eePosY < 0.45
      else:
        eePos = np.array([jointPos[1][0], jointPos[1][1]])
        targetPos = np.array([0, plant.l[0] + plant.l[1]])
        distToTarget = np.linalg.norm(eePos - targetPos)
        return distToTarget < radiusThresh and eePos[1] > 0.45

    def noisy_reset_func():
        x = simulator.reset(eval)
        obs = np.array([x[0],x[1],x[2],x[3],
                              np.cos(x[0]), np.sin(x[0]), np.cos(x[1]), np.sin(x[1])])
        return obs

    env = CustomEnv(
        simulator=simulator,
        dynamics_func=dynamics_func,
        reward_func=reward_func,
        terminated_func=terminated_func,
        reset_func=noisy_reset_func,
        obs_space=obs_space,
        act_space=act_space,
        max_episode_steps=tf//dt,
        env_logging=envLogging
    )
    env= my_gym_wrapper.OlympicWrapper(env)
    env = wrappers.CanonicalSpecWrapper(env, clip=True)
    env = wrappers.SinglePrecisionWrapper(env)

    if envLogging:
      #For logging to a dataset for imitation learning:
      import envlogger
      import tensorflow_datasets as tfds
      from envlogger.backends import tfds_backend_writer as tfbw

      dataset_config = tfds.rlds.rlds_base.DatasetConfig(
        name='ilqr_acrobot',
        observation_info=tfds.features.Tensor(
          shape=obs_space.shape,dtype=obs_space.dtype,
          encoding=tfds.features.Encoding.ZLIB),
        action_info=tfds.features.Tensor(
          shape=act_space.shape,dtype=act_space.dtype,
          encoding=tfds.features.Encoding.ZLIB),
        reward_info=np.float32, #tf can deal with it
        discount_info=np.float32
      )

      from pathlib import Path
      dataDir = './ilqr_env_logs/{}'.format(envNum)
      Path(dataDir).mkdir(parents=True, exist_ok=True)
      env = envlogger.EnvLogger(env,
                                backend=tfbw.TFDSBackendWriter(
                                  data_directory=dataDir,
                                  split_name='train',
                                  #max_episodes_per_file=1000,
                                  ds_config=dataset_config
                                )
                               )


    return env


def make_environment(suite: str, task: str) -> dm_env.Environment:
  """Makes the requested continuous control environment.

  Args:
    suite: One of 'gym' or 'control'.
    task: Task to load. If `suite` is 'control', the task must be formatted as
      f'{domain_name}:{task_name}'

  Returns:
    An environment satisfying the dm_env interface expected by Acme agents.
  """

  if suite not in _VALID_TASK_SUITES:
    raise ValueError(
        f'Unsupported suite: {suite}. Expected one of {_VALID_TASK_SUITES}')

  if suite == 'gym':
    env = gym.make(task)
    env = gymWrappers.RecordVideo(env, './videos/' + task + '/' + str(time()) + '/')

    # Make sure the environment obeys the dm_env.Environment interface.
    env = my_gym_wrapper.GymWrapper(env)

  elif suite == 'control':
    # Load dm_suite lazily not require Mujoco license when not using it.
    from dm_control import suite as dm_suite  # pylint: disable=g-import-not-at-top
    domain_name, task_name = task.split(':')
    env = dm_suite.load(domain_name, task_name)
    env = wrappers.ConcatObservationWrapper(env)

  # Wrap the environment so the expected continuous action spec is [-1, 1].
  # Note: this is a no-op on 'control' tasks.
  env = wrappers.CanonicalSpecWrapper(env, clip=True)
  env = wrappers.SinglePrecisionWrapper(env)
  return env
