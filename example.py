
from double_pendulum.model.model_parameters import model_parameters

# from double_pendulum.simulation.gym_env import (
#     CustomEnv,
#     double_pendulum_dynamics_func,
# )
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.controller.pid.point_pid_controller import PointPIDController
from double_pendulum.utils.plotting import plot_timeseries
import gpu
import rlController
from timeit import default_timer
from absl import flags
flags.DEFINE_string("label", "test","")

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

import matplotlib
matplotlib.use('TkAgg')


def minimalExample():


    plant = SymbolicDoublePendulum(mass=[0.6, 0.5],
                                   length=[0.3, 0.2])

    sim = Simulator(plant=plant)

    controller = PointPIDController(torque_limit=[0, 5.0])
    controller.set_parameters(Kp=10, Ki=1, Kd=1)
    controller.init()
    import matplotlib
    matplotlib.use('TkAgg')
    T, X, U = sim.simulate_and_animate(t0=0.0, x0=[3.14, 0.0, 0., 0.],
                                       tf=10., dt=0.02, controller=controller)
    plot_timeseries(T, X, U)


def mpoExample(_):
    gpu.SetGPU(1, True)

    # model parameters
    design = "design_C.0"
    model = "model_3.0"
    robot = "acrobot"

    if robot == "pendubot":
        torque_limit = [6.0, 0.0]
    elif robot == "acrobot":
        torque_limit = [0.0, 6.0]
    else:
        torque_limit = None

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
    print("Making plant... ")

    with Timer(True) as t:
        plant = SymbolicDoublePendulum(model_pars=mpar)
    # plant = SymbolicDoublePendulum(mass=[0.6, 0.5],
    #                                length=[0.3, 0.2])
    simulator = Simulator(plant=plant)

    # controller = PointPIDController(torque_limit=torque_limit)
    # controller.set_parameters(Kp=10, Ki=1, Kd=1)
    # controller.init()

    # agent = 'ppo'
    # agent = 'mpo'
    agent = 'd4pg'
    # agent = 'pwil'
    controller = rlController.RLPolicy(agent, simulator, env_name=robot, plant=plant)
    import numpy as np
    import matplotlib
    matplotlib.use('TkAgg')
    simulator.set_state(0.0, [0.05, -0.05, 0.0, 0.0])
    simulator.reset_data_recorder()
    simulator.reset()
    T, X, U = simulator.simulate_and_animate(t0=0.0, x0=[0.05, -0.05, 0.0, 0.0],
                                             tf=10., dt=0.002, controller=controller,
                                             video_name='videos/{}_swingup_finetuned.mp4'.format(agent), save_video=True)

    plot_timeseries(T, X, U)

    # learning environment parameters
    # state_representation = 2
    # obs_space = gym.spaces.Box(
    #     np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0])
    # )
    # act_space = gym.spaces.Box(np.array([-1]), np.array([1]))
    #
    # dynamics_func = double_pendulum_dynamics_func(
    #     simulator=simulator,
    #     dt=dt,
    #     integrator=integrator,
    #     robot=robot,
    #     state_representation=state_representation,
    # )


def pwilExample(_):
    gpu.SetGPU(1, True)

    # model parameters
    design = "design_C.0"
    model = "model_3.0"
    robot = "acrobot"

    if robot == "pendubot":
        torque_limit = [6.0, 0.0]
    elif robot == "acrobot":
        torque_limit = [0.0, 6.0]
    else:
        torque_limit = None

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
    # plant = SymbolicDoublePendulum(model_pars=mpar)
    # if plant is None:
    print("Making plant... ")

    with Timer(True) as t:
      plant = SymbolicDoublePendulum(model_pars=mpar)
    # plant = SymbolicDoublePendulum(mass=[0.6, 0.5],
    #                                length=[0.3, 0.2])
    simulator = Simulator(plant=plant)

    # controller = PointPIDController(torque_limit=torque_limit)
    # controller.set_parameters(Kp=10, Ki=1, Kd=1)
    # controller.init()

    # agent = 'ppo'
    agent = 'pwil'

    controller = rlController.RLPolicy(agent, simulator, env_name=robot, plant=plant)
    import numpy as np

    # a2 = controller.get_control_output_(np.array([np.pi, 0, 0, 0]))
    # a = controller.get_control_output_(np.array([0, 0, 0, 0]))
    # for _ in range(100):
    #     controller.init_()
    label = flags.FLAGS.label

    # T, X, U = simulator.simulate_and_animate(t0= 0.0, x0 = [0.0, 0.0, 0.0, 0.0],
    #                                    tf=6., dt=0.002, controller=controller, video_name='videos/pwil_{}.mp4'.format(label), save_video=True)
    # controller.init()
    controller.stabiMode=True

    simulator.set_state(0.0, [np.pi+0.05, -0.05, 0.0, 0.0])
    simulator.reset_data_recorder()
    simulator.reset()
    T, X, U = simulator.simulate_and_animate(t0=0.0, x0=[np.pi+0.05, -0.05, 0.0, 0.0],
                                             tf=3., dt=0.002, controller=controller,
                                             video_name='videos/pwil_stabiOnly_{}.mp4'.format(label), save_video=True)
    # plot_timeseries(T, X, U)


    # learning environment parameters
    # state_representation = 2
    # obs_space = gym.spaces.Box(
    #     np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0])
    # )
    # act_space = gym.spaces.Box(np.array([-1]), np.array([1]))
    #
    # dynamics_func = double_pendulum_dynamics_func(
    #     simulator=simulator,
    #     dt=dt,
    #     integrator=integrator,
    #     robot=robot,
    #     state_representation=state_representation,
    # )
def loadAndPlot():
    path = './trajectory.pickle'
    import pickle
    with open(path, 'rb') as handle:
        myDict = pickle.load(handle)
    t, x, u = myDict['t'],myDict['x'],myDict['u']
    plot_timeseries(t,x,u)
    # print(myDict)

if __name__ == '__main__':

    # minimalExample()
    from absl import app
    app.run(mpoExample)
    # loadAndPlot()
