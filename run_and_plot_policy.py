
# Once you've trained a policy for the acrobot swingup task, use this script to run it 
# in the environment, plot the trajectory (incl. torque), and optionally save a video.

from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.controller.pid.point_pid_controller import PointPIDController
from double_pendulum.utils.plotting import plot_timeseries
import gpu
import rlController
from timeit import default_timer
import matplotlib
matplotlib.use('TkAgg')
from absl import flags
from absl import app
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

def run_and_plot(_):
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


if __name__ == '__main__':
    app.run(runAndPlot)
