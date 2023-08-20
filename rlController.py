from double_pendulum.controller.abstract_controller import AbstractController

import runD4PG
import runMPO
import ppoExample
import runPWIL
import jax
from acme import specs
from acme.tf import savers
from acme.utils import counting
from acme.jax import utils
import reverb
from acme.jax.experiments import config
from acme import core
from typing import Optional, Sequence, Tuple
from acme import types
import dm_env
import sys
import time
import numpy as np

class RLPolicy(AbstractController):
    """Controller Template"""

    def __init__(self, agent, simulator, env_name = 'acrobot',plant=None,stabi=False):
        super().__init__()
        self.simulator = simulator
        self.env_name = env_name
        self.plant=plant
        self.stabiMode = False
        spi = 16
        retrace = True
        policyEval = False
        bs = 192
        lrCoef = 1
        numLearners = 8
        if agent =='mpo':
            experiment,checkpointDir = runMPO.build_experiment_config(env_name, spi, retrace,bs,lrCoef,numLearners,plant=plant)
            experiment_stabi = experiment
        elif agent == 'ppo':
            experiment = ppoExample.build_experiment_config(env_name)
        elif agent == 'pwil':
            experiment = runPWIL.build_experiment_config(env_name,eval=False, plant=plant, stabi=False)
            experiment_stabi=experiment
            # experiment_stabi = runPWIL.build_experiment_config(env_name, eval=False, plant=plant, stabi=True)
        elif agent == 'd4pg':
            experiment = runD4PG.build_experiment_config(env_name,plant=plant)
            experiment_stabi = experiment
        #Load parameters
        #Make the MPO network for the policy
        key = jax.random.PRNGKey(experiment.seed)

        # Create the environment and get its spec.
        environment = experiment.environment_factory(experiment.seed)
        environment_spec = experiment.environment_spec or specs.make_environment_spec(
            environment)

        # Create the networks and policy.
        networks = experiment.network_factory(environment_spec)
        policy = config.make_policy(
            experiment=experiment,
            networks=networks,
            environment_spec=environment_spec,
            evaluation=policyEval)

        # Create the replay server and grab its address.
        replay_tables = experiment.builder.make_replay_tables(environment_spec,
                                                              policy)

        # Disable blocking of inserts by tables' rate limiters, as this function
        # executes learning (sampling from the table) and data generation
        # (inserting into the table) sequentially from the same thread
        # which could result in blocked insert making the algorithm hang.
        replay_tables, rate_limiters_max_diff = _disable_insert_blocking(
            replay_tables)

        replay_server = reverb.Server(replay_tables, port=None)
        replay_client = reverb.Client(f'localhost:{replay_server.port}')

        # Parent counter allows to share step counts between train and eval loops and
        # the learner, so that it is possible to plot for example evaluator's return
        # value as a function of the number of training episodes.
        parent_counter = counting.Counter(time_delta=0.)

        dataset = experiment.builder.make_dataset_iterator(replay_client)
        # We always use prefetch as it provides an iterator with an additional
        # 'ready' method.
        dataset = utils.prefetch(dataset, buffer_size=1)

        # Create actor, adder, and learner for generating, storing, and consuming
        # data respectively.
        # NOTE: These are created in reverse order as the actor needs to be given the
        # adder and the learner (as a source of variables).
        learner_key, key = jax.random.split(key)
        learner = experiment.builder.make_learner(
            random_key=learner_key,
            networks=networks,
            dataset=dataset,
            logger_fn=experiment.logger_factory,
            environment_spec=environment_spec,
            replay_client=replay_client,
            counter=counting.Counter(parent_counter, prefix='learner', time_delta=0.))

        adder = experiment.builder.make_adder(replay_client, environment_spec, policy)


        #__________________________start
        environment_stabi = experiment_stabi.environment_factory(experiment.seed)
        environment_spec_stabi = experiment_stabi.environment_spec or specs.make_environment_spec(
          environment_stabi)

        # Create the networks and policy.
        networks_stabi = experiment_stabi.network_factory(environment_spec)
        policy_stabi = config.make_policy(
          experiment=experiment_stabi,
          networks=networks_stabi,
          environment_spec=environment_spec_stabi,
          evaluation=policyEval)

        # Create the replay server and grab its address.
        replay_tables_stabi = experiment_stabi.builder.make_replay_tables(environment_spec_stabi,
                                                              policy_stabi)

        # Disable blocking of inserts by tables' rate limiters, as this function
        # executes learning (sampling from the table) and data generation
        # (inserting into the table) sequentially from the same thread
        # which could result in blocked insert making the algorithm hang.
        replay_tables_stabi, rate_limiters_max_diff = _disable_insert_blocking(
          replay_tables_stabi)

        replay_server_stabi = reverb.Server(replay_tables_stabi, port=None)
        replay_client_stabi = reverb.Client(f'localhost:{replay_server_stabi.port}')

        # Parent counter allows to share step counts between train and eval loops and
        # the learner, so that it is possible to plot for example evaluator's return
        # value as a function of the number of training episodes.
        parent_counter_stabi = counting.Counter(time_delta=0.)

        dataset_stabi = experiment.builder.make_dataset_iterator(replay_client_stabi)
        # We always use prefetch as it provides an iterator with an additional
        # 'ready' method.
        dataset_stabi = utils.prefetch(dataset_stabi, buffer_size=1)

        # Create actor, adder, and learner for generating, storing, and consuming
        # data respectively.
        # NOTE: These are created in reverse order as the actor needs to be given the
        # adder and the learner (as a source of variables).
        learner_key, key = jax.random.split(key)
        learner_stabi = experiment_stabi.builder.make_learner(
          random_key=learner_key,
          networks=networks_stabi,
          dataset=dataset_stabi,
          logger_fn=experiment_stabi.logger_factory,
          environment_spec=environment_spec_stabi,
          replay_client=replay_client_stabi,
          counter=counting.Counter(parent_counter_stabi, prefix='learner', time_delta=0.))

        # adder_stabi = experiment_stabi.builder.make_adder(replay_client_stabi, environment_spec_stabi, policy_stabi)

        #__________________________



        actor_key, key = jax.random.split(key)
        actor = experiment.builder.make_actor(
            actor_key, policy, environment_spec, variable_source=learner, adder=adder)
        actor_stabi = experiment_stabi.builder.make_actor(
          actor_key, policy_stabi, environment_spec_stabi, variable_source=learner_stabi, adder=adder)

        # Create the environment loop used for training.
        # train_counter = counting.Counter(
        #     parent_counter, prefix='actor', time_delta=0.)
        # train_logger = experiment.logger_factory('actor',
        #                                          train_counter.get_steps_key(), 0)

        checkpointer = None
        if experiment.checkpointing is not None:
            checkpointing = experiment.checkpointing
            checkpointer = savers.Checkpointer(
                objects_to_save={'learner': learner},
                time_delta_minutes=checkpointing.time_delta_minutes,
                directory=checkpointing.directory,
                subdirectory='learner',
                add_uid=checkpointing.add_uid,
                max_to_keep=checkpointing.max_to_keep,
                keep_checkpoint_every_n_hours=checkpointing.keep_checkpoint_every_n_hours,
                checkpoint_ttl_seconds=checkpointing.checkpoint_ttl_seconds,
            )

        checkpointer_stabi = None
        if experiment_stabi.checkpointing is not None:
          checkpointing_stabi = experiment_stabi.checkpointing
          checkpointer_stabi = savers.Checkpointer(
            objects_to_save={'learner': learner_stabi},
            time_delta_minutes=checkpointing_stabi.time_delta_minutes,
            directory=checkpointing_stabi.directory,
            subdirectory='learner',
            add_uid=checkpointing_stabi.add_uid,
            max_to_keep=checkpointing_stabi.max_to_keep,
            keep_checkpoint_every_n_hours=checkpointing_stabi.keep_checkpoint_every_n_hours,
            checkpoint_ttl_seconds=checkpointing_stabi.checkpoint_ttl_seconds,
          )

        actor._state = actor._init(actor_key)
        actor = _LearningActor(actor, learner, dataset, replay_tables,
                               rate_limiters_max_diff, checkpointer)

        actor._actor.update()
        self.actor = actor

        actor_stabi._state = actor_stabi._init(actor_key)
        actor_stabi = _LearningActor(actor_stabi, learner_stabi, dataset_stabi, replay_tables_stabi,
                               rate_limiters_max_diff, checkpointer_stabi)

        actor_stabi._actor.update()
        self.actor_stabi = actor_stabi

        self.init()

    def set_parameters(self):
        """
        Set controller parameters. Optional.
        Can be overwritten by actual controller.
        """
        pass

    def set_goal(self, x):
        """
        Set the desired state for the controller. Optional.
        Can be overwritten by actual controller.

        Parameters
        ----------
        x : array_like, shape=(4,), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]
        """
        self.goal = x

    def init_(self):
        """
        Initialize the controller. Optional.
        Can be overwritten by actual controller.
        Initialize function which will always be called before using the
        controller.
        """
        self.stabiMode=False

    def reset_(self):
        """
        Reset the Controller. Optional
        Can be overwritten by actual controller.
        Function to reset parameters inside the controller.
        """
        pass

    def get_control_output_(self, x, t=None):
        """
        The function to compute the control input for the double pendulum's
        actuator(s).
        Supposed to be overwritten by actual controllers. The API of this
        method should not be changed. Unused inputs/outputs can be set to None.

        Parameters
        ----------
        x : array_like, shape=(4,), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]
        t : float, optional
            time, unit=[s]
            (Default value=None)

        Returns
        -------
        array_like
            shape=(2,), dtype=float
            actuation input/motor torque,
            order=[u1, u2],
            units=[Nm]
        """

        # take observation: x(shape = 4), x0,y0,x1,y1.
        # xLocal = the first four
        # x = x[:4]
        x = np.array([x[0],x[1],x[2],x[3], np.cos(x[0]),np.sin(x[0]), np.cos(x[1]),np.sin(x[1])])
        if self.stabiMode:
          a = self.actor_stabi.select_action(x).item() * 6
        else:
          a = self.actor.select_action(x).item() * 6
          jointPos = self.plant.forward_kinematics(x[0:2])
          eePos = np.array([jointPos[1][0], jointPos[1][1]])
          targetPos = np.array([0, self.plant.l[0] + self.plant.l[1]])
          distToTarget = np.linalg.norm(eePos - targetPos)
          if distToTarget < 0.1:
            self.stabiMode = True
            print("stabilization mode engaged")
        # a = self.actor.select_action(x).item()*6
        # a = self.actor_stabi.select_action(x).item() * 6
        if self.env_name == "pendubot":
            u = [a, 0.0]
        elif self.env_name == "acrobot":
            u = [0.0, a]
        else:
            u = None

        return u

    def save_(self, save_dir):
        """
        Save controller parameters. Optional
        Can be overwritten by actual controller.

        Parameters
        ----------
        save_dir : string or path object
            directory where the parameters will be saved
        """
        pass

    def get_forecast(self):
        """
        Get a forecast trajectory as planned by the controller. Optional.
        Can be overwritten by actual controller.

        Returns
        -------
        list
            Time array
        list
            X array
        list
            U array
        """
        return [], [], []

    def get_init_trajectory(self):
        """
        Get an initial (reference) trajectory used by the controller. Optional.
        Can be overwritten by actual controller.

        Returns
        -------
        list
            Time array
        list
            X array
        list
            U array
        """
        return [], [], []

class _LearningActor(core.Actor):
  """Actor which learns (updates its parameters) when `update` is called.

  This combines a base actor and a learner. Whenever `update` is called
  on the wrapping actor the learner will take a step (e.g. one step of gradient
  descent) as long as there is data available for training
  (provided iterator and replay_tables are used to check for that).
  Selecting actions and making observations are handled by the base actor.
  Intended to be used by the `run_experiment` only.
  """

  def __init__(self, actor: core.Actor, learner: core.Learner,
               iterator: core.PrefetchingIterator,
               replay_tables: Sequence[reverb.Table],
               sample_sizes: Sequence[int],
               checkpointer: Optional[savers.Checkpointer]):
    """Initializes _LearningActor.

    Args:
      actor: Actor to be wrapped.
      learner: Learner on which step() is to be called when there is data.
      iterator: Iterator used by the Learner to fetch training data.
      replay_tables: Collection of tables from which Learner fetches data
        through the iterator.
      sample_sizes: For each table from `replay_tables`, how many elements the
        table should have available for sampling to wait for the `iterator` to
        prefetch a batch of data. Otherwise more experience needs to be
        collected by the actor.
      checkpointer: Checkpointer to save the state on update.
    """
    self._actor = actor
    self._learner = learner
    self._iterator = iterator
    self._replay_tables = replay_tables
    self._sample_sizes = sample_sizes
    self._learner_steps = 0
    self._checkpointer = checkpointer

  def select_action(self, observation: types.NestedArray) -> types.NestedArray:
    return self._actor.select_action(observation)

  def observe_first(self, timestep: dm_env.TimeStep):
    self._actor.observe_first(timestep)

  def observe(self, action: types.NestedArray, next_timestep: dm_env.TimeStep):
    self._actor.observe(action, next_timestep)

  def _maybe_train(self):
    trained = False
    while True:
      if self._iterator.ready():
        self._learner.step()
        batches = self._iterator.retrieved_elements() - self._learner_steps
        self._learner_steps += 1
        assert batches == 1, (
            'Learner step must retrieve exactly one element from the iterator'
            f' (retrieved {batches}). Otherwise agent can deadlock. Example '
            'cause is that your chosen agent'
            's Builder has a `make_learner` '
            'factory that prefetches the data but it shouldn'
            't.')
        trained = True
      else:
        # Wait for the iterator to fetch more data from the table(s) only
        # if there plenty of data to sample from each table.
        for table, sample_size in zip(self._replay_tables, self._sample_sizes):
          if not table.can_sample(sample_size):
            return trained
        # Let iterator's prefetching thread get data from the table(s).
        time.sleep(0.001)

  def update(self):
    if self._maybe_train():
      # Update the actor weights only when learner was updated.
      self._actor.update()
    if self._checkpointer:
      self._checkpointer.save()
def _disable_insert_blocking(
        tables: Sequence[reverb.Table]
) -> Tuple[Sequence[reverb.Table], Sequence[int]]:
    """Disables blocking of insert operations for a given collection of tables."""
    modified_tables = []
    sample_sizes = []
    for table in tables:
        rate_limiter_info = table.info.rate_limiter_info
        rate_limiter = reverb.rate_limiters.RateLimiter(
            samples_per_insert=rate_limiter_info.samples_per_insert,
            min_size_to_sample=rate_limiter_info.min_size_to_sample,
            min_diff=rate_limiter_info.min_diff,
            max_diff=sys.float_info.max)
        modified_tables.append(table.replace(rate_limiter=rate_limiter))
        # Target the middle of the rate limiter's insert-sample balance window.
        sample_sizes.append(
            max(1, int(
                (rate_limiter_info.max_diff - rate_limiter_info.min_diff) / 2)))
    return modified_tables, sample_sizes