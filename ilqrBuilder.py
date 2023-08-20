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

"""PPO Builder."""
from typing import Iterator, List, Optional

from acme import adders
from acme import core
from acme import specs
from acme.adders import reverb as adders_reverb
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import actors
from acme.agents.jax import builders

from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.utils import counting
from acme.utils import loggers
from acme.agents.jax.ppo import networks as ppo_networks
import jax
import numpy as np
import reverb

import os
from datetime import datetime

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.controller.ilqr.ilqr_mpc_cpp import ILQRMPCCPPController


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


class ILQRBuilder(
  builders.ActorLearnerBuilder[ppo_networks.PPONetworks,
  actor_core_lib.FeedForwardPolicyWithExtra,
  reverb.ReplaySample]):
  """ILQR Builder."""

  def __init__(
      self,
      unroll_length = 8,
      replay_table_name = 'default',
      batch_size = 180,
      obs_normalization_fns_factory = None,
      variable_update_period = 1,
      stabi=False
  ):
    """Creates PPO builder."""


    # An extra step is used for bootstrapping when computing advantages.
    self._sequence_length = unroll_length+1
    self.replay_table_name = replay_table_name
    self.batch_size = batch_size
    self.obs_normalization_fns_factory = obs_normalization_fns_factory
    self.variable_update_period = variable_update_period
    self.stabi = stabi

  def make_replay_tables(
      self,
      environment_spec: specs.EnvironmentSpec,
      policy_dummy: actor_core_lib.FeedForwardPolicyWithExtra,
  ) -> List[reverb.Table]:
    """Creates reverb tables for the algorithm."""
    del policy_dummy
    # params_num_sgd_steps is used to track how old the actor parameters are
    extra_spec = {}
    signature = adders_reverb.SequenceAdder.signature(
        environment_spec, extra_spec, sequence_length=self._sequence_length)
    return [
        reverb.Table.queue(
            name=self.replay_table_name,
            max_size=self.batch_size,
            signature=signature)
    ]


  def make_dataset_iterator(
      self, replay_client: reverb.Client) -> Iterator[reverb.ReplaySample]:
    """Creates a dataset.

    The iterator batch size is computed as follows:

    Let:
      B := learner batch size (config.batch_size)
      H := number of hosts (jax.process_count())
      D := number of local devices per host

    The Reverb iterator will load batches of size B // (H * D). After wrapping
    the iterator with utils.multi_device_put, this will result in an iterable
    that provides B // H samples per item, with B // (H * D) samples placed on
    each local device. In a multi-host setup, each host has its own learner
    node and builds its own instance of the iterator. This will result
    in a total batch size of H * (B // H) == B being consumed per learner
    step (since the learner is pmapped across all devices). Note that
    jax.device_count() returns the total number of devices across hosts,
    i.e. H * D.

    Args:
      replay_client: the reverb replay client

    Returns:
      A replay buffer iterator to be used by the local devices.
    """
    iterator_batch_size, ragged = divmod(self.batch_size,
                                         jax.device_count())
    if ragged:
      raise ValueError(
          'Learner batch size must be divisible by total number of devices!')

    # We don't use datasets.make_reverb_dataset() here to avoid interleaving
    # and prefetching, that doesn't work well with can_sample() check on update.
    # NOTE: Value for max_in_flight_samples_per_worker comes from a
    # recommendation here: https://git.io/JYzXB
    dataset = reverb.TrajectoryDataset.from_table_signature(
        server_address=replay_client.server_address,
        table=self.replay_table_name,
        max_in_flight_samples_per_worker=(
            2 * self.batch_size // jax.process_count()
        ),
    )
    dataset = dataset.batch(iterator_batch_size, drop_remainder=True)
    dataset = dataset.as_numpy_iterator()
    return utils.multi_device_put(iterable=dataset, devices=jax.local_devices())

  def make_adder(
      self,
      replay_client: reverb.Client,
      environment_spec: Optional[specs.EnvironmentSpec],
      policy_dummy: Optional[actor_core_lib.FeedForwardPolicyWithExtra],
  ) -> Optional[adders.Adder]:
    """Creates an adder which handles observations."""
    del environment_spec, policy_dummy
    # Note that the last transition in the sequence is used for bootstrapping
    # only and is ignored otherwise. So we need to make sure that sequences
    # overlap on one transition, thus "-1" in the period length computation.
    return adders_reverb.SequenceAdder(
        client=replay_client,
        priority_fns={self.replay_table_name: None},
        period=self._sequence_length - 1,
        sequence_length=self._sequence_length,
    )

  def make_learner(
      self,
      random_key: networks_lib.PRNGKey,
      networks,
      dataset: Iterator[reverb.ReplaySample],
      logger_fn: loggers.LoggerFactory,
      environment_spec: specs.EnvironmentSpec,
      replay_client: Optional[reverb.Client] = None,
      counter: Optional[counting.Counter] = None,
  ):
    del replay_client
    return None

  def make_actor(
      self,
      random_key: networks_lib.PRNGKey,
      policy_dummy: actor_core_lib.FeedForwardPolicyWithExtra,
      environment_spec: specs.EnvironmentSpec,
      variable_source: Optional[core.VariableSource] = None,
      adder: Optional[adders.Adder] = None,
      robot = 'acrobot',
      plant=None
  ) -> core.Actor:
    # assert variable_source is not None
    actor_core = self.make_actor_core_from_controller(robot,plant)
    # variable_client = variable_utils.VariableClient(
    #     variable_source,
    #     'params',
    #     device='cpu',
    #     update_period=self.variable_update_period)
    actor = actors.GenericActor(
        actor_core, random_key, None, adder, backend='cpu', jit=False)
    return actor

  def make_policy(
      self,
      networks,
      environment_spec: specs.EnvironmentSpec,
      evaluation: bool = False):
    del environment_spec
    return None

  def make_actor_core_from_controller(self, robot='acrobot',plant=None):
    #TODO: when the robot is near the target, switch to a different controller; possibly PID; look for the best one.
    design = "design_C.0"
    model = "model_3.0"
    traj_model = "model_3.1"
    # robot = "acrobot"

    friction_compensation = True

    # # model parameters
    if robot == "acrobot":
      torque_limit = [0.0, 6.0]
    elif robot == "pendubot":
      torque_limit = [5.0, 0.0]
    else:
      raise TypeError("robot must be acrobot or pendubot")

    model_par_path = "./double_pendulum_local/data/system_identification/identified_parameters/" + design + "/" + model + "/model_parameters.yml"
    mpar = model_parameters(filepath=model_par_path)

    mpar_con = model_parameters(filepath=model_par_path)
    # mpar_con.set_motor_inertia(0.)
    if friction_compensation:
      mpar_con.set_damping([0., 0.])
      mpar_con.set_cfric([0., 0.])
    mpar_con.set_torque_limit(torque_limit)

    # simulation parameter
    dt = 0.002
    t_final = 5.0  # 4.985
    integrator = "runge_kutta"
    start = [0., 0., 0., 0.]
    goal = [np.pi, 0., 0., 0.]

    # noise
    process_noise_sigmas = [0.0, 0.0, 0.0, 0.0]
    meas_noise_sigmas = [0.0, 0.0, 0.05, 0.05]
    delay_mode = "vel"
    delay = 0.01 #different for stabi
    if self.stabi:
      delay = 0
    u_noise_sigmas = [0.01, 0.01] # different for stabi
    if self.stabi:
      u_noise_sigmas = [0,0]
    u_responsiveness = 1.0
    perturbation_times = []
    perturbation_taus = []

    # filter args
    meas_noise_vfilter = "none"
    meas_noise_cut = 0.1
    filter_kwargs = {"lowpass_alpha": [1., 1., 0.3, 0.3],
                     "kalman_xlin": goal,
                     "kalman_ulin": [0., 0.],
                     "kalman_process_noise_sigmas": process_noise_sigmas,
                     "kalman_meas_noise_sigmas": meas_noise_sigmas,
                     "ukalman_integrator": integrator,
                     "ukalman_process_noise_sigmas": process_noise_sigmas,
                     "ukalman_meas_noise_sigmas": meas_noise_sigmas}

    # controller parameters
    # N = 20
    N = int(100*(0.005/dt))
    con_dt = dt
    N_init = 1000
    max_iter = 10
    max_iter_init = 1500
    regu_init = 1.
    max_regu = 10000.
    min_regu = 0.01
    break_cost_redu = 1e-6
    trajectory_stabilization = True
    shifting = 1

    # trajectory parameters
    # init_csv_path = os.path.join("./double_pendulum/data/trajectories", design, traj_model, robot, "ilqr_2/trajectory.csv")
    init_csv_path =None
    if robot == "acrobot":

      sCu = [.1, .1]
      sCp = [.1, .1]
      sCv = [0.01, 0.1]
      sCen = 0.0
      fCp = [100., 10.]
      fCv = [10., 1.]
      fCen = 0.0

      f_sCu = [0.1, 0.1]
      f_sCp = [.1, .1]
      f_sCv = [.01, .01]
      f_sCen = 0.0
      f_fCp = [10., 10.]
      f_fCv = [1., 1.]
      f_fCen = 0.0



    elif robot == "pendubot":

      sCu = [0.001, 0.001]
      sCp = [0.01, 0.01]
      sCv = [0.01, 0.01]
      sCen = 0.
      fCp = [100., 100.]
      fCv = [1., 1.]
      fCen = 0.
 #no f_ for stabi
      f_sCu = sCu
      f_sCp = sCp
      f_sCv = sCv
      f_sCen = sCen
      f_fCp = fCp
      f_fCv = fCv
      f_fCen = fCen
    else:
      raise TypeError('robot specified incorrectly')

    init_sCu = sCu
    init_sCp = sCp
    init_sCv = sCv
    init_sCen = sCen
    init_fCp = fCp
    init_fCv = fCv
    init_fCen = fCen



    # construct simulation objects
    if plant is None:
      print("Making plant... ")
      with Timer(True) as t:
        plant = SymbolicDoublePendulum(model_pars=mpar)

    sim = Simulator(plant=plant)
    sim.set_process_noise(process_noise_sigmas=process_noise_sigmas)
    sim.set_measurement_parameters(meas_noise_sigmas=meas_noise_sigmas,
                                   delay=delay,
                                   delay_mode=delay_mode)
    sim.set_motor_parameters(u_noise_sigmas=u_noise_sigmas,
                             u_responsiveness=u_responsiveness)

    controller = ILQRMPCCPPController(model_pars=mpar_con)
    controller.set_start(start)
    controller.set_goal(goal)
    controller.set_parameters(N=N,
                              dt=con_dt,
                              max_iter=max_iter,
                              regu_init=regu_init,
                              max_regu=max_regu,
                              min_regu=min_regu,
                              break_cost_redu=break_cost_redu,
                              integrator=integrator,
                              trajectory_stabilization=trajectory_stabilization,
                              shifting=shifting)
    controller.set_cost_parameters(sCu=sCu,
                                   sCp=sCp,
                                   sCv=sCv,
                                   sCen=sCen,
                                   fCp=fCp,
                                   fCv=fCv,
                                   fCen=fCen)
    #no final cost for stabi
    if not self.stabi:
      controller.set_final_cost_parameters(sCu=f_sCu,
                                           sCp=f_sCp,
                                           sCv=f_sCv,
                                           sCen=f_sCen,
                                           fCp=f_fCp,
                                           fCv=f_fCv,
                                           fCen=f_fCen)
    #no initial traj for stabi
    if not self.stabi:
      if init_csv_path is None:
        controller.compute_init_traj(N=N_init,
                                     dt=dt,
                                     max_iter=max_iter_init,
                                     regu_init=regu_init,
                                     max_regu=max_regu,
                                     min_regu=min_regu,
                                     break_cost_redu=break_cost_redu,
                                     sCu=init_sCu,
                                     sCp=init_sCp,
                                     sCv=init_sCv,
                                     sCen=init_sCen,
                                     fCp=init_fCp,
                                     fCv=init_fCv,
                                     fCen=init_fCen,
                                     integrator=integrator)
      else:
        controller.load_init_traj(csv_path=init_csv_path,
                                  num_break=40,
                                  poly_degree=3)

    controller.set_filter_args(filt=meas_noise_vfilter, x0=goal, dt=dt, plant=plant,
                               simulator=sim, velocity_cut=meas_noise_cut,
                               filter_kwargs=filter_kwargs)
    if friction_compensation:
      controller.set_friction_compensation(damping=mpar.b, coulomb_fric=mpar.cf)

    controller.init()


    def select_action(params: networks_lib.Params,
                      observation: networks_lib.Observation,
                      state):
      #Ok, we receive observation as: x0,y0,x1,y1,qd0_norm,qd1_norm,q0_raw,q1_raw
      #All we need is to call dynamicsfunc.unscale_state(observation),
      #and we should have the correct input to the ilqr controller.
      #Below I've stolen that piece of the code to avoid instantiations and imports
      #Note, the original code was wrong; the first arg of arctan2 is y, not x!!!

      #Assume state representation is 3, as this is canonical for this env type
      # x = np.array(
      #   [
      #     np.arctan2(observation[1], observation[0]),
      #     np.arctan2(observation[3], observation[2]),
      #     observation[4] * 8.0,
      #     observation[5] * 8.0,
      #   ]
      # )


      action = controller.get_control_output_(observation)
      if robot == 'acrobot':
        action = action[1]
      elif robot == 'pendubot':
        action = action[0]
      else:
        raise TypeError("invalid robot str")
      action = np.expand_dims(np.array(action, dtype = np.float32), 0)
      return action, state

    def init(rng):
      controller.init()
      return rng

    def get_extras(unused_rng):
      return ()

    return actor_core_lib.ActorCore(init=init, select_action=select_action,
                     get_extras=get_extras)


