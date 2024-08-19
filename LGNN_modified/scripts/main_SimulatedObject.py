import csv
import json
import os
import socket
import sys
import time
from datetime import datetime
from functools import wraps

import fire
import matplotlib

from psystems.npendulum import (edge_order)
from psystems.nsprings import (edge_order)

MAINPATH = ".."  # nopep8
sys.path.append(MAINPATH)  # nopep8

import random
import jraph
import src
from src.graph import *
from src.md import *
from src.models import MSE, initialize_mlp
from src.utils import *
from scipy.signal import butter, filtfilt, freqz

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter

jax.config.update("jax_enable_x64", True)

path_data_root = "C:\YutongZhang\LGNN\dataset"
path_data_train = path_data_root + "\data_20240605_135014.csv"
path_data_test = path_data_root + "\data_20240605_135053.csv"
matplotlib.rcParams['figure.max_open_warning'] = 50

use_RealData = False
use_SimData = True
use_SimNoise = True
noise_meas = 0.001

use_drag = True

const_numerical = 2e-7  # note that linalg also has numerical problem (because of float precision)
const_gravity_acc = 9.81

use_object = 1
execute_StaticTest = False
execute_learn = True
message_pass_num = 1  # 1
execute_render = False
execute_test = False
model_trained_path = r"C:\YutongZhang\LGNN\trained\simulation\08-14-2024_20-32-06_Object1\lgnn_trained_model_True_0.dil"

if use_object == 5:
    N = 8
else:
    N = 5  # number of points

if use_object == 2:
    masses = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
elif use_object == 5:
    masses = np.array([40, 64, 56, 67, 45, 62, 54, 70]) * 1.0
else:
    masses = np.array([40.0, 40.0, 40.0, 40.0, 40.0])

if use_object == 3:
    length = np.array([0.1, 0.1, 0.1, 0.1])
elif use_object == 5:
    length = np.array([0.05, 0.054, 0.059, 0.062, 0.07, 0.061, 0.056])
else:
    length = np.array([0.05, 0.05, 0.05, 0.05])

species = jnp.zeros(N, dtype=int)  # node types
object_mass = np.sum(masses)
object_length = np.sum(length)

use_stretching = False
object_stiffness_stretching_scale = 5000 * 1e3
if use_object == 4:
    object_stiffness_stretching_sim = np.array([5.0, 5.0, 5.0, 5.0, 5.0]) * object_stiffness_stretching_scale
elif use_object == 5:
    object_stiffness_stretching_sim = np.array(
        [2.1, 3.0, 1.5, 1.9, 1.2, 2.5, 2.7, 1.0]) * object_stiffness_stretching_scale
else:
    object_stiffness_stretching_sim = np.array([1.0, 1.0, 1.0, 1.0, 1.0]) * object_stiffness_stretching_scale

use_bending = False
object_stiffness_bending_scale = 0.1 * 1e3
if use_object == 4:
    object_stiffness_bending_sim = np.array([5.0, 5.0, 5.0, 5.0, 5.0]) * object_stiffness_bending_scale
elif use_object == 5:
    object_stiffness_bending_sim = np.array([2.1, 3.0, 1.5, 1.9, 1.2, 2.5, 2.7, 1.0]) * object_stiffness_bending_scale
else:
    object_stiffness_bending_sim = np.array([1.0, 1.0, 1.0, 1.0, 1.0]) * object_stiffness_bending_scale

use_twisting = False
object_stiffness_twisting_scale = 0.001 * 1e3
object_stiffness_twisting_sim = object_stiffness_twisting_scale * 1.0

use_tension = False
object_stiffness_tension_scale = 1000 * 1e3
object_stiffness_tension_sim = object_stiffness_tension_scale * 5.0
object_damping_tension_scale = 10 * 1e3 * 0.0
object_damping_tension_sim = object_damping_tension_scale * 5.0

use_damping = True
object_damping_scale = 0.01 * 1e3
object_damping_sim = np.array([5.0] * N) * object_damping_scale

use_gravity_energy = True
use_gravity_force = False  # proven to be the same

use_VirtualCoupling = True
VirtualCoupling_stiffness = 500 * 1e3
VirtualCoupling_damping = 5 * 1e3

VirtualCoupling_sim_magnitude_xy = 0.1  # m
VirtualCoupling_sim_magnitude_z = 0.02  # m
VirtualCoupling_sim_period = 0.8  # s

use_separate_feedback = False
feedback_stiffness = 0.5 * 1e3 * 1e3
use_feedback_VirtualScaling = False
feedback_VirtualScaling = 5 * 1e-3
use_feedback_OverallScaling = False
feedback_OverallScaling = 0.5

# Kalman filter
use_KalmanFilter = True
process_var = 0.001  # Process noise variance 0.005
measurement_var = 0.001  # Measurement noise variance 0.01
estimated_var = 0.01  # Initial estimate of error covariance 1

# Server information
SERVER_IP = "127.0.0.1"  # or "localhost"
PORT = 12312
BUFLEN = 512  # max length of answer

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Bind the socket to the server address
server_address = (SERVER_IP, PORT)
print("Starting server on {} port {}".format(SERVER_IP, PORT))
sock.bind(server_address)


class KalmanFilter3D:
    def __init__(self, process_var, measurement_var, estimated_var):
        # Initialize the state vector [x, y, z, vx, vy, vz, ax, ay, az]
        self.x = np.zeros(9)  # Initial state estimate
        self.P = np.eye(9) * estimated_var  # Initial error covariance

        # State transition matrix (will be updated with dt)
        self.F = np.eye(9)

        # Measurement matrix
        self.H = np.zeros((3, 9))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1

        # Process noise covariance
        self.Q = np.zeros((9, 9))
        self.Q[0, 0] = process_var  # Position noise variance
        self.Q[1, 1] = process_var  # Position noise variance
        self.Q[2, 2] = process_var  # Position noise variance
        self.Q[3, 3] = process_var  # Velocity noise variance
        self.Q[4, 4] = process_var  # Velocity noise variance
        self.Q[5, 5] = process_var  # Velocity noise variance
        self.Q[6, 6] = process_var  # Acceleration noise variance
        self.Q[7, 7] = process_var  # Acceleration noise variance
        self.Q[8, 8] = process_var  # Acceleration noise variance

        # Measurement noise covariance
        self.R = np.eye(3) * measurement_var

    def predict(self, dt):
        # Update state transition matrix with new dt
        self.F[0, 3] = dt
        self.F[0, 6] = 0.5 * dt ** 2
        self.F[1, 4] = dt
        self.F[1, 7] = 0.5 * dt ** 2
        self.F[2, 5] = dt
        self.F[2, 8] = 0.5 * dt ** 2

        # Prediction step
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        # Measurement update step
        y = z - self.H @ self.x  # Measurement residual
        S = self.H @ self.P @ self.H.T + self.R  # Residual covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        self.x = self.x + K @ y
        self.P = self.P - K @ self.H @ self.P  # Update error covariance

    def get_state(self):
        return self.x

    def get_state_position(self):
        return self.x[:3]

    def get_state_velocity(self):
        return self.x[3:6]


def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]


def pprint(*args, namespace=globals()):
    for arg in args:
        print(f"{namestr(arg, namespace)[0]}: {arg}")


def wrap_main(f):
    def fn(*args, **kwargs):
        config = (args, kwargs)
        print("Configs: ")
        print(f"Args: ")
        for i in args:
            print(i)
        print(f"KwArgs: ")
        for k, v in kwargs.items():
            print(k, ":", v)
        return f(*args, **kwargs, config=config)

    return fn


def Main(epochs=1000, seed=42, rname=True, saveat=10, error_fn="L2error",
         dt=1.0e-4, stride=10, trainm=0, grid=False, lr=0.001,
         withdata=None, datapoints=None, batch_size=50, dim=3):
    return wrap_main(main)(epochs=epochs, seed=seed, rname=rname, saveat=saveat, error_fn=error_fn,
                           dt=dt, stride=stride, trainm=trainm, grid=grid, lr=lr,
                           withdata=withdata, datapoints=datapoints, batch_size=batch_size, dim=dim)


def main(epochs=5000, seed=42, rname=True, saveat=10, error_fn="L2error",
         dt=1.0e-3, stride=100, trainm=1, grid=False, lr=0.001, withdata=None, datapoints=None,
         batch_size=1000, config=None, dim=3):
    print("\n******************** Basic settings ********************\n")

    np.random.seed(seed)
    key = random.PRNGKey(seed)

    randfilename = datetime.now().strftime("%m-%d-%Y_%H-%M-%S") + f"_{datapoints}"

    if execute_render:
        PSYS = f"Object{use_object}-render"
    if execute_learn:
        PSYS = f"Object{use_object}-learn"
    if execute_test:
        PSYS = f"Object{use_object}-test"
    if use_SimData:
        PSYS += "-SimData"
    if use_RealData:
        PSYS += "-RealData"
    TAG = f"lgnn"
    out_dir = f"../results"

    def _filename(name, tag=TAG):
        rstring = randfilename if (rname and (tag != "data")) else (
            "0" if (tag == "data") or (withdata == None) else f"{withdata}")
        filename_prefix = f"{out_dir}/{PSYS}-{tag}/{rstring}/"
        file = f"{filename_prefix}/{name}"
        os.makedirs(os.path.dirname(file), exist_ok=True)
        filename = f"{filename_prefix}/{name}".replace("//", "/")
        # print("===", filename, "===")
        return filename

    def displacement(a, b):
        return a - b

    def shift(R, dR, V):
        return R + dR, V

    def OUT(f):
        @wraps(f)
        def func(file, *args, tag=TAG, **kwargs):
            return f(_filename(file, tag=tag), *args, **kwargs)

        return func

    loadmodel = OUT(src.models.loadmodel)
    savemodel = OUT(src.models.savemodel)

    loadfile = OUT(src.io.loadfile)
    savefile = OUT(src.io.savefile)
    save_ovito = OUT(src.io.save_ovito)

    savefile(f"config_{use_drag}_{trainm}.pkl", config)

    @jit
    def get_distance_vec(vec):
        return jnp.linalg.norm(vec + const_numerical)

    @jit
    def get_angle_vec(vec1, vec2):
        vec1_norm = vec1 / (jnp.linalg.norm(vec1) + const_numerical)
        vec2_norm = vec2 / (jnp.linalg.norm(vec2) + const_numerical)
        angle_dot = jnp.dot(vec1_norm, vec2_norm)
        angle_dot = jnp.clip(angle_dot, -1.0, 1.0)
        angle = jnp.arccos(angle_dot)
        return angle

    @jit
    def get_angle_surface(vec1, vec2, vec3):
        vec1_norm = vec1 / (jnp.linalg.norm(vec1) + const_numerical)
        vec2_norm = vec2 / (jnp.linalg.norm(vec2) + const_numerical)
        vec3_norm = vec3 / (jnp.linalg.norm(vec3) + const_numerical)
        surface1 = jnp.cross(vec1_norm, vec2_norm) + const_numerical
        surface1_norm = surface1 / (jnp.linalg.norm(surface1) + const_numerical)
        surface2 = jnp.cross(vec2_norm, vec3_norm) + const_numerical
        surface2_norm = surface2 / (jnp.linalg.norm(surface2) + const_numerical)
        angle_dot = jnp.dot(surface1_norm, surface2_norm)
        angle_dot = jnp.clip(angle_dot, -1.0, 1.0)
        angle = jnp.arccos(angle_dot)
        return angle

    @jit
    def get_velocity_relative(vel_from, pos_from, vel_to, pos_to):
        pos_relative = pos_to - pos_from
        vel_relative = vel_to - vel_from
        distance = jnp.sqrt(jnp.square(pos_relative).sum())
        vel_angular = vel_relative / (distance + const_numerical)
        return vel_angular

    @jit
    def get_force_tension(x, x_lead, stiffness):
        x_diff_horizontal = x[0:2] - x_lead[0:2]
        force = -stiffness * x_diff_horizontal
        return force

    @jit
    def get_distance_delta_square(x, x_lead, original_length):
        x_diff = x - x_lead
        direction = x_diff / (jnp.linalg.norm(x_diff) + const_numerical)
        return jnp.square(x_diff - original_length * direction).sum()

    if use_object == 5:
        @jit
        def get_energy_stretching(x, x_lead, stiffness, length):
            distance = jnp.sqrt(get_distance_delta_square(x, x_lead, length))
            stiffness = stiffness * (1.0 + 0.5 * jnp.abs(jnp.sin(distance / 0.2 * (jnp.pi / 2.0))))
            return 0.5 * stiffness * (distance ** 2)
    else:
        @jit
        def get_energy_stretching(x, x_lead, stiffness, length):
            return 0.5 * stiffness * get_distance_delta_square(x, x_lead, length)

    @jit
    def get_force_VirtualCoupling(x, v, x_lead, v_lead):
        x_diff = x[0, :] - x_lead[0, :]
        v_diff = v[0, :] - v_lead[0, :]
        force = -VirtualCoupling_stiffness * x_diff - VirtualCoupling_damping * v_diff
        force_gravity = jnp.zeros((dim))
        force_gravity = force_gravity.at[2].set(object_mass * const_gravity_acc)
        force += force_gravity
        return force

    @jit
    def get_force_feedback(x, v, x_lead, v_lead):
        x_diff = x[0, :] - x_lead[0, :]
        v_diff = v[0, :] - v_lead[0, :]
        force_gravity = jnp.zeros((dim))
        force_gravity = force_gravity.at[2].set(object_mass * const_gravity_acc)
        if use_separate_feedback:
            force = -feedback_stiffness * x_diff
            force += force_gravity
        else:
            force = -VirtualCoupling_stiffness * x_diff - VirtualCoupling_damping * v_diff
            if use_feedback_VirtualScaling:
                force *= feedback_VirtualScaling
            force += force_gravity
            if use_feedback_OverallScaling:
                force *= feedback_OverallScaling
        return -force * 1e-3

    @jit
    def get_coordinate_Python2TouchX(array):
        input = array.reshape((dim))
        output = jnp.zeros((dim))
        output = output.at[0].set(input[0])  # x = x
        output = output.at[1].set(input[2])  # y = z
        output = output.at[2].set(-input[1])  # z = -y
        return output.reshape(array.shape)

    @jit
    def get_coordinate_TouchX2Python(array):
        input = array.reshape((dim))
        output = jnp.zeros((dim))
        output = output.at[0].set(input[0])  # x = x
        output = output.at[1].set(-input[2])  # y = -z
        output = output.at[2].set(input[1])  # z = y
        return output.reshape(array.shape)

    @jit
    def get_force_stick(pos_lead, acc_lead, pos_follow, acc_follow, mass):
        pos_relative = pos_lead - pos_follow
        distance = jnp.linalg.norm(pos_relative + const_numerical)
        direction = pos_relative / (distance + const_numerical)
        acc_relative = acc_lead - acc_follow
        acc_direction = jnp.dot(acc_relative, direction)
        force_stick = mass * acc_direction * direction
        # force_gravity = jnp.zeros_like(force_stick)
        # force_gravity = force_gravity.at[2].set(-mass * 10.0)
        force = force_stick
        scaling = jnp.ones_like(force)
        # scaling = scaling.at[0:2].set(1.0 / 7.5)
        # scaling = scaling.at[2].set(1.0 / 15.0)
        return force * scaling

    if use_object == 5:
        @jit
        def get_energy_bending(angle, stiffness):
            stiffness = stiffness * (1.0 + 0.5 * jnp.abs(jnp.sin(angle)))
            return 0.5 * stiffness * (angle ** 2)
    else:
        @jit
        def get_energy_bending(angle, stiffness):
            return 0.5 * stiffness * (angle ** 2)

    @jit
    def get_energy_twisting(x, stiffness, initial_length):
        return 0.5 * stiffness * ((x - initial_length) ** 2)

    def apply_KalmanFilter(timestamp, data_num, data_position, point_num):
        output_position = np.zeros((data_num, point_num, dim))
        output_velocity = np.zeros((data_num, point_num, dim))
        output_acceleration = np.zeros((data_num, point_num, dim))

        t = timestamp.reshape(-1)

        for i_N in range(point_num):

            measured_position = np.zeros((dim, data_num))
            for i_time in range(data_num):
                for i_dim in range(3):
                    measured_position[i_dim, i_time] = data_position[i_time, i_N, i_dim]

            kf = KalmanFilter3D(process_var, measurement_var, estimated_var)

            estimated_position = np.zeros_like(measured_position)
            estimated_velocity = np.zeros_like(measured_position)
            estimated_acceleration = np.zeros_like(measured_position)

            for i in range(len(t)):
                if i == 0:
                    kf.x[:3] = measured_position[:, i]
                    # kf.x[8] = -const_gravity_acc
                else:
                    dt = t[i] - t[i - 1]
                    kf.predict(dt)
                    kf.update(measured_position[:, i])

                state = kf.get_state()
                estimated_position[:, i] = state[:3]
                estimated_velocity[:, i] = state[3:6]
                estimated_acceleration[:, i] = state[6:9]

            for i_time in range(data_num):
                for i_dim in range(3):
                    output_position[i_time, i_N, i_dim] = estimated_position[i_dim, i_time]
                    output_velocity[i_time, i_N, i_dim] = estimated_velocity[i_dim, i_time]
                    output_acceleration[i_time, i_N, i_dim] = estimated_acceleration[i_dim, i_time]

        return output_position, output_velocity, output_acceleration

    def plot_trajectory(data_traj, data_num, data_time, data_type, data_compare=None,
                        show_history=False, show_structure=True, show_skip=20, data_force=None,
                        data_virtual=None, show_fps=20):
        # Set up the plot
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        if show_skip is not None:

            data_num = (int)(data_num / show_skip)

            data_traj_temp = np.zeros((data_num, N, dim))
            data_time_temp = np.zeros((data_num))
            for i in range(data_num):
                data_traj_temp[i, :, :] = data_traj[i * show_skip, :, :]
                data_time_temp[i] = data_time[i * show_skip]
            data_traj = data_traj_temp
            data_time = data_time_temp

            if data_compare is not None:
                data_compare_temp = np.zeros((data_num, N, dim))
                for i in range(data_num):
                    data_compare_temp[i, :, :] = data_compare[i * show_skip, :, :]
                data_compare = data_compare_temp

            if data_force is not None:
                data_force_temp = np.zeros((data_num, dim))
                for i in range(data_num):
                    data_force_temp[i, :] = data_force[i * show_skip, :]
                data_force = data_force_temp

            if data_virtual is not None:
                data_virtual_temp = np.zeros((data_num, 1, dim))
                for i in range(data_num):
                    data_virtual_temp[i, :, :] = data_virtual[i * show_skip, :, :]
                data_virtual = data_virtual_temp

        if data_compare is None:
            limit_refer = data_traj
        else:
            limit_refer = data_compare

        x_max = float('-inf')
        x_min = float('inf')
        y_max = float('-inf')
        y_min = float('inf')
        z_max = float('-inf')
        z_min = float('inf')
        for i_N in range(N):
            if x_max < max(limit_refer[:, i_N, 0]):
                x_max = max(limit_refer[:, i_N, 0])
            if x_min > min(limit_refer[:, i_N, 0]):
                x_min = min(limit_refer[:, i_N, 0])
            if y_max < max(limit_refer[:, i_N, 1]):
                y_max = max(limit_refer[:, i_N, 1])
            if y_min > min(limit_refer[:, i_N, 1]):
                y_min = min(limit_refer[:, i_N, 1])
            if z_max < max(limit_refer[:, i_N, 2]):
                z_max = max(limit_refer[:, i_N, 2])
            if z_min > min(limit_refer[:, i_N, 2]):
                z_min = min(limit_refer[:, i_N, 2])

        scale = x_max - x_min
        if y_max - y_min > scale:
            scale = y_max - y_min
        if z_max - z_min > scale:
            scale = z_max - z_min

        x_min_scaled = (x_max + x_min) / 2.0 - scale / 2.0
        x_max_scaled = (x_max + x_min) / 2.0 + scale / 2.0
        y_min_scaled = (y_max + y_min) / 2.0 - scale / 2.0
        y_max_scaled = (y_max + y_min) / 2.0 + scale / 2.0
        z_min_scaled = (z_max + z_min) / 2.0 - scale / 2.0
        z_max_scaled = (z_max + z_min) / 2.0 + scale / 2.0

        ax.set_xlim(x_min_scaled, x_max_scaled)
        ax.set_ylim(y_min_scaled, y_max_scaled)
        ax.set_zlim(z_min_scaled, z_max_scaled)

        # compute the length of each edge
        edge_length = np.empty((data_num, N))
        for i_time in range(data_num):
            R = data_traj[i_time, :, :]
            if data_virtual is None:
                edge_length[i_time, :] = np.sqrt(np.square(R - np.vstack([np.zeros_like(R[0]), R[:-1]])).sum(axis=1))
            else:
                edge_length[i_time, :] = np.sqrt(np.square(R - np.vstack([data_virtual[i_time], R[:-1]])).sum(axis=1))
        if data_compare is not None:
            edge_length_compare = np.empty((data_num, N))
            for i_time in range(data_num):
                R = data_compare[i_time, :, :]
                if data_virtual is None:
                    edge_length_compare[i_time, :] = np.sqrt(
                        np.square(R - np.vstack([np.zeros_like(R[0]), R[:-1]])).sum(axis=1))
                else:
                    edge_length_compare[i_time, :] = np.sqrt(
                        np.square(R - np.vstack([data_virtual[i_time], R[:-1]])).sum(axis=1))

        if show_history:
            lines = [ax.plot([], [], [], 'b-')[0] for _ in range(N)]
            points = [ax.plot([], [], [], 'o')[0] for _ in range(N)]
        if show_structure:
            lines = [ax.plot([], [], [], 'b-')[0] for _ in range(N)]
            points = [ax.plot([], [], [], 'ko')[0] for _ in range(N)]
            if data_compare is not None:
                lines_compare = [ax.plot([], [], [], 'm:')[0] for _ in range(N)]
                points_compare = [ax.plot([], [], [], 'r+')[0] for _ in range(N)]
            if data_force is not None:
                quivers_length_max = np.max(np.abs(data_force))
                quivers_length_target = np.max(edge_length[:, 1])
                quivers_scaling = quivers_length_target * 0.8 / (quivers_length_max + const_numerical)

                def get_quiver_args(i_time):
                    x = data_traj[i_time, 0, 0]
                    y = data_traj[i_time, 0, 1]
                    z = data_traj[i_time, 0, 2]
                    u = data_force[i_time, 0] * quivers_scaling
                    v = data_force[i_time, 1] * quivers_scaling
                    w = data_force[i_time, 2] * quivers_scaling
                    return x, y, z, u, v, w

                quivers = ax.quiver(*get_quiver_args(0))
            if data_virtual is not None:
                points_virtual = [ax.plot([], [], [], 'ro')[0] for _ in range(1)]

        def init():
            if show_history:
                for line, point in zip(lines, points):
                    line.set_data([], [])
                    line.set_3d_properties([])
                    point.set_data([], [])
                    point.set_3d_properties([])
                return lines + points
            if show_structure:
                for line in lines:
                    line.set_data([], [])
                    line.set_3d_properties([])
                for point in points:
                    point.set_data([], [])
                    point.set_3d_properties([])
                if data_compare is not None:
                    for line in lines_compare:
                        line.set_data([], [])
                        line.set_3d_properties([])
                    for point in points_compare:
                        point.set_data([], [])
                        point.set_3d_properties([])
                if data_virtual is not None:
                    for point in points_virtual:
                        point.set_data([], [])
                        point.set_3d_properties([])
                return (lines + points
                        + (lines_compare + points_compare if data_compare is not None else [])
                        + (points_virtual if data_virtual is not None else []))

        def update(num):

            nonlocal quivers
            title = f"Time: {data_time[num]:.2f} s\nData: l0={edge_length[num, 0]:.4f}"
            for i in range(1, N):
                title += f",l{i}={edge_length[num, i]:.2f}"
            if data_compare is not None:
                title += f"\nComp: l0={edge_length_compare[num, 0]:.4f}"
                for i in range(1, N):
                    title += f",l{i}={edge_length_compare[num, i]:.2f}"
            if data_force is not None:
                title += f"\nForce Feedback: x={data_force[num, 0]:.2f},y={data_force[num, 1]:.2f},z={data_force[num, 2]:.2f}"
            ax.set_title(title)

            if show_history:
                for i, (line, point) in enumerate(zip(lines, points)):
                    line.set_data(data_traj[:num, i, 0], data_traj[:num, i, 1])
                    line.set_3d_properties(data_traj[:num, i, 2])
                    point.set_data(data_traj[num - 1:num, i, 0], data_traj[num - 1:num, i, 1])
                    point.set_3d_properties(data_traj[num - 1:num, i, 2])
                return lines + points

            if show_structure:
                # lines
                for i in range(N):
                    if i == 0:
                        if data_virtual is not None:
                            x_data = [data_virtual[num, 0, 0], data_traj[num, i, 0]]
                            y_data = [data_virtual[num, 0, 1], data_traj[num, i, 1]]
                            z_data = [data_virtual[num, 0, 2], data_traj[num, i, 2]]
                        else:
                            x_data = [0, data_traj[num, i, 0]]
                            y_data = [0, data_traj[num, i, 1]]
                            z_data = [0, data_traj[num, i, 2]]
                    else:
                        x_data = [data_traj[num, i - 1, 0], data_traj[num, i, 0]]
                        y_data = [data_traj[num, i - 1, 1], data_traj[num, i, 1]]
                        z_data = [data_traj[num, i - 1, 2], data_traj[num, i, 2]]
                    lines[i].set_data(x_data, y_data)
                    lines[i].set_3d_properties(z_data)
                    if data_compare is not None:
                        if i == 0:
                            if data_virtual is not None:
                                x_data_compare = [data_virtual[num, 0, 0], data_compare[num, i, 0]]
                                y_data_compare = [data_virtual[num, 0, 1], data_compare[num, i, 1]]
                                z_data_compare = [data_virtual[num, 0, 2], data_compare[num, i, 2]]
                            else:
                                x_data_compare = [0, data_compare[num, i, 0]]
                                y_data_compare = [0, data_compare[num, i, 1]]
                                z_data_compare = [0, data_compare[num, i, 2]]
                        else:
                            x_data_compare = [data_compare[num, i - 1, 0], data_compare[num, i, 0]]
                            y_data_compare = [data_compare[num, i - 1, 1], data_compare[num, i, 1]]
                            z_data_compare = [data_compare[num, i - 1, 2], data_compare[num, i, 2]]
                        lines_compare[i].set_data(x_data_compare, y_data_compare)
                        lines_compare[i].set_3d_properties(z_data_compare)
                # points
                for i in range(N):
                    points[i].set_data([data_traj[num, i, 0]], [data_traj[num, i, 1]])
                    points[i].set_3d_properties([data_traj[num, i, 2]])
                    if data_compare is not None:
                        points_compare[i].set_data([data_compare[num, i, 0]], [data_compare[num, i, 1]])
                        points_compare[i].set_3d_properties([data_compare[num, i, 2]])
                if data_virtual is not None:
                    points_virtual[0].set_data([data_virtual[num, 0, 0]], [data_virtual[num, 0, 1]])
                    points_virtual[0].set_3d_properties([data_virtual[num, 0, 2]])

                # others
                if data_force is not None:
                    quivers.remove()
                    quivers = ax.quiver(*get_quiver_args(num))

                return (lines + points
                        + (lines_compare + points_compare if data_compare is not None else [])
                        + (points_virtual if data_virtual is not None else []))

        ani = FuncAnimation(fig, update, frames=data_num, init_func=init, blit=True)
        ani.save(_filename(f"trajectory_{data_type}.gif"), writer=PillowWriter(fps=show_fps))
        plt.close(fig)  # not displayed

    def plot_data_example(data, data_type, data_meaning, timestamp, index=0):
        fig, axis = plt.subplots(3, 1)
        label_title_ax = ["X", "Y", "Z"]
        for i, ax in enumerate(axis):
            ax.plot(timestamp[100:], data[100:, index, i])
            ax.set_title(label_title_ax[i])
        plt.suptitle(f"{data_meaning} (Example {index}, {data_type})")
        plt.tight_layout()
        plt.savefig(_filename(f"dataset_example{index}_{data_meaning}_{data_type}.png"))

    # ******************** Dataset preparation ********************

    print("\n******************** Dataset preparation ********************\n")

    if use_SimData:
        print("Simulation data used\n")
        if execute_learn:
            train_num = 6000
            train_num_i = 2000  # one set, requires to be dividable (train_num / train_num_i)

        if execute_render or execute_test:
            train_num = 10
            train_num_i = 10

        train_init_num = (int)(train_num / train_num_i)
        test_num = train_num_i
        Rs = np.empty((train_num, N, dim))
        Vs = np.empty((train_num, N, dim))
        Rst = np.empty((test_num, N, dim))
        Vst = np.empty((test_num, N, dim))
        train_timestamp = np.arange(train_num) * dt * stride
        test_timestamp = np.arange(test_num) * dt * stride

        if use_VirtualCoupling:
            Rs_lead = np.zeros((train_num, 1, dim))
            Vs_lead = np.zeros((train_num, 1, dim))
            for i in range(train_init_num):
                key, subkey = jax.random.split(key)
                period_train_x = jax.random.uniform(subkey,
                                                    minval=VirtualCoupling_sim_period * 0.5,
                                                    maxval=VirtualCoupling_sim_period * 1.5)
                omega_train_x = 2 * np.pi / period_train_x
                key, subkey = jax.random.split(key)
                magnitude_train_x = jax.random.uniform(subkey,
                                                       minval=VirtualCoupling_sim_magnitude_xy * 0.5,
                                                       maxval=VirtualCoupling_sim_magnitude_xy * 1.5)

                key, subkey = jax.random.split(key)
                period_train_y = jax.random.uniform(subkey,
                                                    minval=VirtualCoupling_sim_period * 0.5,
                                                    maxval=VirtualCoupling_sim_period * 1.5)
                omega_train_y = 2 * np.pi / period_train_y
                key, subkey = jax.random.split(key)
                magnitude_train_y = jax.random.uniform(subkey,
                                                       minval=VirtualCoupling_sim_magnitude_xy * 0.5,
                                                       maxval=VirtualCoupling_sim_magnitude_xy * 1.5)

                key, subkey = jax.random.split(key)
                period_train_z = jax.random.uniform(subkey,
                                                    minval=VirtualCoupling_sim_period * 0.5,
                                                    maxval=VirtualCoupling_sim_period * 1.5)
                omega_train_z = 2 * np.pi / period_train_z
                key, subkey = jax.random.split(key)
                magnitude_train_z = jax.random.uniform(subkey,
                                                       minval=VirtualCoupling_sim_magnitude_z * 0.5,
                                                       maxval=VirtualCoupling_sim_magnitude_z * 1.5)

                for i_time in range(train_num_i):
                    index = train_num_i * i + i_time

                    Rs_lead[index, 0, 0] = magnitude_train_x * np.sin(omega_train_x * train_timestamp[i_time])
                    Vs_lead[index, 0, 0] = omega_train_x * magnitude_train_x * np.cos(
                        omega_train_x * train_timestamp[i_time])
                    if train_timestamp[i_time] >= period_train_x:
                        Rs_lead[index, 0, 0] = 0.0
                        Vs_lead[index, 0, 0] = 0.0

                    Rs_lead[index, 0, 1] = magnitude_train_y * np.sin(omega_train_y * train_timestamp[i_time])
                    Vs_lead[index, 0, 1] = omega_train_y * magnitude_train_y * np.cos(
                        omega_train_y * train_timestamp[i_time])
                    if train_timestamp[i_time] >= period_train_y:
                        Rs_lead[index, 0, 1] = 0.0
                        Vs_lead[index, 0, 1] = 0.0

                    Rs_lead[index, 0, 2] = magnitude_train_z * np.sin(omega_train_z * train_timestamp[i_time])
                    Vs_lead[index, 0, 2] = omega_train_z * magnitude_train_z * np.cos(
                        omega_train_z * train_timestamp[i_time])
                    if train_timestamp[i_time] >= period_train_z:
                        Rs_lead[index, 0, 2] = 0.0
                        Vs_lead[index, 0, 2] = 0.0

            Rst_lead = np.zeros((test_num, 1, dim))
            Vst_lead = np.zeros((test_num, 1, dim))

            key, subkey = jax.random.split(key)
            period_test_x = jax.random.uniform(subkey,
                                               minval=VirtualCoupling_sim_period * 0.5,
                                               maxval=VirtualCoupling_sim_period * 1.5)
            omega_test_x = 2 * np.pi / period_test_x
            key, subkey = jax.random.split(key)
            magnitude_test_x = jax.random.uniform(subkey,
                                                  minval=VirtualCoupling_sim_magnitude_xy * 0.5,
                                                  maxval=VirtualCoupling_sim_magnitude_xy * 1.5)

            key, subkey = jax.random.split(key)
            period_test_y = jax.random.uniform(subkey,
                                               minval=VirtualCoupling_sim_period * 0.5,
                                               maxval=VirtualCoupling_sim_period * 1.5)
            omega_test_y = 2 * np.pi / period_test_y
            key, subkey = jax.random.split(key)
            magnitude_test_y = jax.random.uniform(subkey,
                                                  minval=VirtualCoupling_sim_magnitude_xy * 0.5,
                                                  maxval=VirtualCoupling_sim_magnitude_xy * 1.5)

            key, subkey = jax.random.split(key)
            period_test_z = jax.random.uniform(subkey,
                                               minval=VirtualCoupling_sim_period * 0.5,
                                               maxval=VirtualCoupling_sim_period * 1.5)
            omega_test_z = 2 * np.pi / period_test_z
            key, subkey = jax.random.split(key)
            magnitude_test_z = jax.random.uniform(subkey,
                                                  minval=VirtualCoupling_sim_magnitude_z * 0.5,
                                                  maxval=VirtualCoupling_sim_magnitude_z * 1.5)

            for i_time in range(test_num):
                index = i_time

                Rst_lead[index, 0, 0] = magnitude_test_x * np.sin(omega_test_x * test_timestamp[i_time])
                Vst_lead[index, 0, 0] = omega_test_x * magnitude_test_x * np.cos(omega_test_x * test_timestamp[i_time])
                if test_timestamp[i_time] >= period_test_x:
                    Rst_lead[index, 0, 0] = 0.0
                    Vst_lead[index, 0, 0] = 0.0

                Rst_lead[index, 0, 1] = magnitude_test_y * np.sin(omega_test_y * test_timestamp[i_time])
                Vst_lead[index, 0, 1] = omega_test_y * magnitude_test_y * np.cos(omega_test_y * test_timestamp[i_time])
                if test_timestamp[i_time] >= period_test_y:
                    Rst_lead[index, 0, 1] = 0.0
                    Vst_lead[index, 0, 1] = 0.0

                Rst_lead[index, 0, 2] = magnitude_test_z * np.sin(omega_test_z * test_timestamp[i_time])
                Vst_lead[index, 0, 2] = omega_test_z * magnitude_test_z * np.cos(omega_test_z * test_timestamp[i_time])
                if test_timestamp[i_time] >= period_test_z:
                    Rst_lead[index, 0, 2] = 0.0
                    Vst_lead[index, 0, 2] = 0.0

            if execute_StaticTest:
                Rs_lead *= 0.0
                Vs_lead *= 0.0
                Rst_lead *= 0.0
                Vst_lead *= 0.0
        else:
            Rs_lead = None
            Vs_lead = None
            Rst_lead = None
            Vst_lead = None

        def get_sim_data_initial(N, dim, key):
            use_set1 = False
            use_set2 = False
            use_set3 = True
            if use_set1:
                basic_scale = 1
                rand_pos = basic_scale * 0.05  # for all points' x and y direction
                rand_vel = basic_scale * 0.05  # for the top point x and y direction
                key, subkey = jax.random.split(key)
                # initial position
                R = jnp.zeros((N, dim))
                for i_N in range(N):
                    # z direction
                    R = R.at[i_N, 2].set(-(i_N + 1))
                    # x direction
                    key, subkey = jax.random.split(key)
                    random_value = jax.random.uniform(subkey, minval=0.0, maxval=rand_pos)
                    if i_N == 0:
                        R = R.at[i_N, 0].set(basic_scale + random_value)
                    else:
                        R = R.at[i_N, 0].set(random_value)
                    # y direction
                    key, subkey = jax.random.split(key)
                    random_value = jax.random.uniform(subkey, minval=0.0, maxval=rand_pos)
                    if i_N == 0:
                        R = R.at[i_N, 1].set(basic_scale + random_value)
                    else:
                        R = R.at[i_N, 1].set(random_value)
                # initial velocity
                V = jnp.zeros((N, dim))
                # top point, x direction
                key, subkey = jax.random.split(key)
                random_value = jax.random.uniform(subkey, minval=0.0, maxval=rand_vel)
                V = V.at[0, 0].set(random_value)
                # top point, y direction
                key, subkey = jax.random.split(key)
                random_value = jax.random.uniform(subkey, minval=0.0, maxval=rand_vel)
                V = V.at[0, 1].set(random_value)

            if use_set2:
                basic_scale = 20
                rand_vel = basic_scale * 0.1  # for the top point x and y direction
                key, subkey = jax.random.split(key)
                # initial position
                R = jnp.zeros((N, dim))
                for i_N in range(N):
                    # z direction
                    R = R.at[i_N, 2].set(-(i_N + 1))
                # initial velocity
                V = jnp.zeros((N, dim))
                # top point, x direction
                key, subkey = jax.random.split(key)
                random_value = jax.random.uniform(subkey, minval=0.0, maxval=basic_scale + rand_vel)
                V = V.at[0, 0].set(random_value)
                # top point, y direction
                key, subkey = jax.random.split(key)
                random_value = jax.random.uniform(subkey, minval=0.0, maxval=basic_scale + rand_vel)
                V = V.at[0, 1].set(random_value)

            if use_set3:
                # initial position
                R = jnp.zeros((N, dim))
                for i_N in range(N):
                    # z direction
                    result = 0.0
                    for j in range(i_N):
                        result += length[j]
                    R = R.at[i_N, 2].set(-result)
                # initial velocity
                V = jnp.zeros((N, dim))

            return R, V

        Rs = np.empty((train_num, N, dim))
        Vs = np.empty((train_num, N, dim))
        for i in range(train_init_num):
            R, V = get_sim_data_initial(N, dim, random.PRNGKey(i))
            Rs[train_num_i * i, :, :] = R
            Vs[train_num_i * i, :, :] = V

        Rst = np.empty((test_num, N, dim))
        Vst = np.empty((test_num, N, dim))
        R, V = get_sim_data_initial(N, dim, random.PRNGKey(50))
        Rst[0, :, :] = R
        Vst[0, :, :] = V

    # ******************** System configuration ********************

    print("\n******************** System configuration ********************\n")

    R, V = Rs[0], Vs[0]
    # length = jnp.sqrt(jnp.square(R[1:] - R[:-1]).sum(axis=1))
    print(f"species = {species}\n")
    print(f"masses = {masses}\n")
    print(f"length = {length}\n")

    #################
    ### Pendulum ####
    #################

    print("\n>> Pendulum\n")

    N_pen = N
    mpass_pen = message_pass_num
    senders_pen = jnp.array([i for i in range(N_pen - 1)] + [i for i in range(1, N_pen)], dtype=int)
    receivers_pen = jnp.array([i for i in range(1, N_pen)] + [i for i in range(N_pen - 1)], dtype=int)
    # senders_pen, receivers_pen = pendulum_connections(N_pen)
    eorder_pen = edge_order(len(senders_pen))  # how does edge_order work???

    def nn_bending(angle, params):
        stiffness = object_stiffness_bending_scale * models.forward_pass(params,
                                                                         angle,
                                                                         activation_fn=models.SquarePlus)
        return 0.5 * stiffness * (angle ** 2)

    def pot_energy_model_bending(x, v, params):
        angle = jnp.zeros((N))
        for i_N in range(2, N):
            vec1 = x[i_N - 1, :] - x[i_N - 2, :]
            vec2 = x[i_N, :] - x[i_N - 1, :]
            angle_temp = get_angle_vec(vec1, vec2)
            angle = angle.at[i_N].set(angle_temp)
        nnresult = nn_bending(angle, params).sum()
        return nnresult

    def nn_twisting(angle, params):
        stiffness = object_stiffness_twisting_scale * models.forward_pass(params,
                                                                          angle,
                                                                          activation_fn=models.SquarePlus)
        return 0.5 * stiffness * (angle ** 2)

    def pot_energy_model_twisting(x, v, params):
        angle = jnp.zeros((N))
        for i_N in range(3, N):
            vec1 = x[i_N - 2, :] - x[i_N - 3, :]
            vec2 = x[i_N - 1, :] - x[i_N - 2, :]
            vec3 = x[i_N, :] - x[i_N - 1, :]
            angle_temp = get_angle_surface(vec1, vec2, vec3)
            angle = angle.at[i_N].set(angle_temp)
        nnresult = nn_twisting(angle, params).sum()
        return nnresult

    def nn_stretching(distance_delta_square, params):
        stiffness = object_stiffness_stretching_scale * models.forward_pass(params,
                                                                            jnp.sqrt(distance_delta_square),
                                                                            activation_fn=models.SquarePlus)
        return 0.5 * stiffness * distance_delta_square

    def pot_energy_model_stretching(x, v, params):
        distance_delta_square = jnp.zeros((N))
        for i_N in range(1, N):
            distance_delta_square_temp = get_distance_delta_square(x=x[i_N, :], x_lead=x[i_N - 1, :],
                                                                   original_length=length[i_N - 1])
            distance_delta_square = distance_delta_square.at[i_N].set(distance_delta_square_temp)
        nnresult = nn_stretching(distance_delta_square, params).sum()
        return nnresult

    if trainm:
        print("kinetic energy: learnable")

        def L_energy_fn_pen(params, graph):
            g, V, T = cal_graph(params, graph, mpass=mpass_pen, eorder=eorder_pen, useT=True)
            return T - V

    else:
        print("kinetic energy: 0.5mv^2")

        kin_energy = partial(lnn._T, mass=masses)

        def L_energy_fn_pen(params, graph):
            g, V, T = cal_graph(params, graph, mpass=mpass_pen, eorder=eorder_pen, useT=True)
            return kin_energy(graph.nodes["velocity"]) - V

    def constraints_pen_func(R, l):
        out = jnp.sqrt(jnp.square(R[1:] - R[:-1]).sum(axis=1)) - l ** 2
        return out

    def constraints_pen(x, v, params):
        if use_stretching:
            return jnp.zeros((1, N * dim))
        else:
            return jax.jacobian(lambda x: constraints_pen_func(x.reshape(-1, dim), length), 0)(x)

    if use_drag == 0:
        print("Drag: 0.0")

        def drag_model_pen(x, v, params):
            return 0.0
    elif use_drag == 1:
        print("Drag: nn")

        @jit
        def drag_model_pen(x, v, x_lead, v_lead, params):
            result = jnp.zeros((N, dim))

            if use_tension:
                def nn_tension_model_pen(x, x_lead, params):
                    stiffness = object_stiffness_tension_scale * models.forward_pass(params,
                                                                                     0.0,
                                                                                     activation_fn=models.ReLU)
                    return get_force_tension(x, x_lead, stiffness)

                nnresult = jnp.zeros((N, dim))
                for i_N in range(1, N):
                    drag_tension = nn_tension_model_pen(x[i_N, :], x[i_N - 1, :], params["drag_tension"]).reshape(-1)
                    nnresult = nnresult.at[i_N, 0:2].set(drag_tension)
                result += nnresult

            if use_VirtualCoupling:
                result_VirtualCoupling = jnp.zeros((N, dim))
                if (x_lead is not None) and (v_lead is not None):
                    force_VirtualCoupling = get_force_VirtualCoupling(x=x, v=v, x_lead=x_lead, v_lead=v_lead)
                    result_VirtualCoupling = result_VirtualCoupling.at[0, :].set(force_VirtualCoupling)
                    result += result_VirtualCoupling
                else:
                    raise ValueError("Using virtual coupling but one/both of x_lead and v_lead is/are None!\n")

            return result.reshape(-1, 1)

    def damping_model_pen(x, v, x_lead, v_lead, params):
        if use_damping:
            def nn_damping_model_pen(vel_relative, params):
                damping = object_damping_scale * models.forward_pass(params,
                                                                     vel_relative * 0.001,
                                                                     activation_fn=models.SquarePlus)
                return - damping * vel_relative

            vel_relative = jnp.zeros((N, dim))
            for i_N in range(1, N):
                vel_relative = vel_relative.at[i_N, :].set(
                    get_velocity_relative(vel_from=v[i_N - 1, :], vel_to=v[i_N, :],
                                          pos_from=x[i_N - 1, :], pos_to=x[i_N, :]))
            nnresult = jnp.zeros((N, dim))
            for i_dim in range(dim):
                nnresult_temp = nn_damping_model_pen(vel_relative[:, i_dim].reshape(-1), params["object_damping"])
                nnresult = nnresult.at[:, i_dim].set(nnresult_temp)
            result = nnresult.reshape(-1, 1)
            return result
        else:
            return 0.0

    def energy_fn_pen(species):
        state_graph_pen = jraph.GraphsTuple(nodes={"position": R, "velocity": V, "type": species},
                                            edges={},
                                            senders=senders_pen,
                                            receivers=receivers_pen,
                                            n_node=jnp.array([R.shape[0]]),
                                            n_edge=jnp.array([senders_pen.shape[0]]),
                                            globals={})

        def apply(R, V, params):
            state_graph_pen.nodes.update(position=R)
            state_graph_pen.nodes.update(velocity=V)
            return L_energy_fn_pen(params, state_graph_pen)

        return apply

    apply_fn_pen = energy_fn_pen(species)

    # create simulation dataset

    mpass = mpass_pen
    senders = senders_pen
    receivers = receivers_pen
    constraints_sim = constraints_pen

    def pot_energy_orig_bending(x):
        result = 0.0
        for i_N in range(2, N):
            vec1 = x[i_N - 1, :] - x[i_N - 2, :]
            vec2 = x[i_N, :] - x[i_N - 1, :]
            angle_temp = get_angle_vec(vec1, vec2)
            result += get_energy_bending(angle_temp,
                                         stiffness=object_stiffness_bending_sim[i_N])
        return result

    def pot_energy_orig_gravity_func(R, g, mass):
        out = (mass * g * R[:, 2]).sum()
        return out

    pot_energy_orig_gravity = partial(pot_energy_orig_gravity_func, g=const_gravity_acc, mass=masses)

    def pot_energy_orig_twisting(x):
        angle = jnp.zeros((N))
        for i_N in range(2, N - 1):
            vec1 = x[i_N - 1, :] - x[i_N - 2, :]
            vec2 = x[i_N, :] - x[i_N - 1, :]
            vec3 = x[i_N + 1, :] - x[i_N, :]
            angle_temp = get_angle_surface(vec1, vec2, vec3)
            angle = angle.at[i_N].set(angle_temp)
        result = vmap(partial(get_energy_twisting, stiffness=object_stiffness_twisting_sim, initial_length=0.0))(
            angle).sum()
        return result

    def pot_energy_orig_stretching(x):
        result = 0.0
        for i_N in range(1, N):
            result += get_energy_stretching(x=x[i_N, :], x_lead=x[i_N - 1, :],
                                            stiffness=object_stiffness_stretching_sim[i_N],
                                            length=length[i_N - 1])
        return result

    def pot_energy_orig(x):
        result = 0.0
        if use_gravity_energy:
            result += pot_energy_orig_gravity(x)
        if use_bending:
            result += pot_energy_orig_bending(x)
        if use_twisting:
            result += pot_energy_orig_twisting(x)
        if use_stretching:
            result += pot_energy_orig_stretching(x)
        return result

    kin_energy_orig = partial(lnn._T, mass=masses)

    def Lsim(x, v, params):
        return kin_energy_orig(v) - pot_energy_orig(x)

    if use_SimData:
        def drag_sim(x, v, x_lead, v_lead, params):
            if use_drag:
                result = jnp.zeros((N, dim))

                if use_tension:
                    result_tension = jnp.zeros((N, dim))
                    for i_N in range(N):
                        force_tension = 0.0
                        if i_N != 0:
                            force_tension += get_force_tension(x=x[i_N, :], x_lead=x[i_N - 1, :],
                                                               stiffness=object_stiffness_tension_sim)
                            force_tension += get_force_tension(x=v[i_N, :], x_lead=v[i_N - 1, :],
                                                               stiffness=object_damping_tension_sim)
                        # if i_N != N - 1:
                        #     force_tension += get_force_tension(x=x[i_N, :], x_lead=x[i_N + 1, :],
                        #                                        stiffness=object_stiffness_tension_sim)
                        #     force_tension += get_force_tension(x=v[i_N, :], x_lead=v[i_N + 1, :],
                        #                                        stiffness=object_damping_tension_sim)
                        result_tension = result_tension.at[i_N, 0:2].set(force_tension)
                    result += result_tension

                if use_gravity_force:
                    result_gravity = jnp.zeros((N, dim))
                    for i_N in range(N):
                        result_gravity = result_gravity.at[i_N, 2].set(-const_gravity_acc * masses[i_N])
                    result += result_gravity

                if use_VirtualCoupling:
                    result_VirtualCoupling = jnp.zeros((N, dim))
                    # print(f"drag_sim: x = {x}, v = {v}, x_lead = {x_lead}, v_lead = {v_lead}\n")
                    # input("Press Enter to continue...")
                    if (x_lead is not None) and (v_lead is not None):
                        force_VirtualCoupling = get_force_VirtualCoupling(x=x, v=v, x_lead=x_lead, v_lead=v_lead)
                        result_VirtualCoupling = result_VirtualCoupling.at[0, :].set(force_VirtualCoupling)
                    else:
                        raise ValueError("Using virtual coupling but one/both of x_lead and v_lead is/are None!\n")
                    result += result_VirtualCoupling

                return result.reshape(-1, 1)
            else:
                return 0.0

        def damping_sim(x, v, x_lead, v_lead, params):
            if use_damping:
                vel_relative = jnp.zeros((N, dim))
                for i_N in range(1, N):
                    vel_relative = vel_relative.at[i_N, :].set(
                        get_velocity_relative(vel_from=v[i_N - 1, :], vel_to=v[i_N, :],
                                              pos_from=x[i_N - 1, :], pos_to=x[i_N, :]))
                damping_force_object = - vel_relative * object_damping_sim.reshape((N, 1))
                result = damping_force_object.reshape(-1, 1)
                return result
            else:
                return 0.0

        acceleration_fn_sim = jit(lnn.accelerationFull(N, dim,
                                                       lagrangian=Lsim,
                                                       non_conservative_forces=damping_sim,
                                                       external_force=drag_sim,
                                                       constraints=constraints_sim))

        v_acceleration_fn_sim = vmap(acceleration_fn_sim, in_axes=(0, 0, 0, 0, None))

        def force_fn_sim(R, V, R_lead, V_lead, params, mass):
            if mass is None:
                return acceleration_fn_sim(R, V, R_lead, V_lead, params)
            else:
                return acceleration_fn_sim(R, V, R_lead, V_lead, params) * mass.reshape(-1, 1)

        @partial(jax.jit, static_argnames=['runs'])
        def get_forward_sim_data(R, V, R_lead, V_lead, runs):
            # print(f"get_forward_sim_data: R_lead = {R_lead}, V_lead = {V_lead}")
            # input("Press Enter to continue...")
            return predition(R, V, R_lead, V_lead, None, force_fn_sim, shift, dt, masses, runs, stride)

        if use_VirtualCoupling:
            print("\n>> Creating simulation dataset: training dataset\n")
            start_time = time.time()
            for i in range(train_init_num):
                for i_time in range(train_num_i - 1):
                    index = i_time + train_num_i * i
                    # print(f"Rs = {Rs[index, :, :]}\n"
                    #       f"Vs = {Vs[index, :, :]}\n"
                    #       f"Rs_lead = {Rs_lead[index, :]}\n"
                    #       f"Vs_lead = {Vs_lead[index, :]}\n")
                    # input("Press Enter to continue...")
                    gen_data_train = get_forward_sim_data(Rs[index, :, :], Vs[index, :, :],
                                                          Rs_lead[index, :], Vs_lead[index, :],
                                                          1)
                    # print(f"index = {index}")
                    # print(f"Rs[index, :, :] = {Rs[index, :, :]}")
                    # print(f"Rs_lead[index, :] = {Rs_lead[index, :]}")
                    # print(f"gen_data_train.position = {gen_data_train.position}")
                    # input("Press Enter to continue...")
                    if index % (test_num / 5) == 0:
                        print(f"In progress: index = {index}")
                    Rs[index + 1, :, :] = gen_data_train.position
                    Vs[index + 1, :, :] = gen_data_train.velocity
            end_time = time.time()
            execution_time = end_time - start_time
            execution_time_aver = execution_time / train_num
            print(f"\nAverage execution frequency = {(1.0 / execution_time_aver):.2f} Hz\n")

            print(">> Creating simulation dataset: test dataset\n")
            start_time = time.time()
            for i_time in range(test_num - 1):
                index = i_time
                gen_data_test = get_forward_sim_data(Rst[index, :, :],
                                                     Vst[index, :, :],
                                                     Rst_lead[index, :],
                                                     Vst_lead[index, :],
                                                     1)
                if index % (test_num / 5) == 0:
                    print(f"In progress: index = {index}")
                Rst[index + 1, :, :] = gen_data_test.position
                Vst[index + 1, :, :] = gen_data_test.velocity
            end_time = time.time()
            execution_time = end_time - start_time
            execution_time_aver = execution_time / test_num
            print(f"\nAverage execution frequency = {(1.0 / (execution_time_aver + const_numerical)):.2f} Hz\n")
        else:
            for i in range(train_init_num):
                gen_data_train = get_forward_sim_data(R=Rs[train_num_i * i, :, :],
                                                      V=Vs[train_num_i * i, :, :],
                                                      R_lead=Rs_lead[train_num_i * i, :],
                                                      V_lead=Vs_lead[train_num_i * i, :],
                                                      runs=train_num_i - 1)
                Rs[train_num_i * i + 1:train_num_i * (i + 1), :, :] = gen_data_train.position
                Vs[train_num_i * i + 1:train_num_i * (i + 1), :, :] = gen_data_train.velocity

            gen_data_test = get_forward_sim_data(R=Rst[0, :, :],
                                                 V=Vst[0, :, :],
                                                 R_lead=Rs_lead[0, :],
                                                 V_lead=Vs_lead[0, :],
                                                 runs=test_num - 1)
            Rst[1:, :, :] = gen_data_test.position
            Vst[1:, :, :] = gen_data_test.velocity

        # add some measurement noise
        if use_SimNoise:
            noise_level_pos = noise_meas  # in m (= 5 mm)
            noise_level_vel = noise_meas  # in m/s (= 1 mm/s)
            for i in range(train_num):
                Rs[i] = Rs[i] + noise_level_pos * np.random.normal(0, 0.1, size=Rs[i].shape)
                Vs[i] = Vs[i] + noise_level_vel * np.random.normal(0, 0.1, size=Vs[i].shape)
                if use_VirtualCoupling:
                    Rs_lead[i] = Rs_lead[i] + noise_level_pos / 4.0 * np.random.normal(0, 0.1, size=Rs_lead[i].shape)
                    Vs_lead[i] = Vs_lead[i] + noise_level_vel / 4.0 * np.random.normal(0, 0.1, size=Vs_lead[i].shape)
            for i in range(test_num):
                Rst[i] = Rst[i] + noise_level_pos * np.random.normal(0, 0.1, size=Rst[i].shape)
                Vst[i] = Vst[i] + noise_level_vel * np.random.normal(0, 0.1, size=Vst[i].shape)
                if use_VirtualCoupling:
                    Rst_lead[i] = Rst_lead[i] + noise_level_pos / 4.0 * np.random.normal(0, 0.1, size=Rst_lead[i].shape)
                    Vst_lead[i] = Vst_lead[i] + noise_level_vel / 4.0 * np.random.normal(0, 0.1, size=Vst_lead[i].shape)

        print(">> Simulation data created\n")

        if use_drag and execute_learn:
            drag_truth_train = np.zeros((train_num, N, dim))
            for i_time in range(train_num):
                R = np.empty((N, dim))
                V = np.empty((N, dim))
                R = Rs[i_time, :, :]
                V = Vs[i_time, :, :]
                if use_VirtualCoupling:
                    R_lead = np.empty((N, dim))
                    V_lead = np.empty((N, dim))
                    R_lead = Rs_lead[i_time, :, :]
                    V_lead = Vs_lead[i_time, :, :]
                else:
                    R_lead = None
                    V_lead = None
                drag_truth_train[i_time, :, :] = drag_sim(R, V, R_lead, V_lead, None).reshape(N, dim)
            # for i_N in range(N):
            #     plot_data_example(drag_truth_train, "TrainTruth", "Drag", train_timestamp, i_N)
            print("Drag: training data ground truth saved\n")

            drag_truth_test = np.zeros((test_num, N, dim))
            for i_time in range(test_num):
                R = np.empty((N, dim))
                V = np.empty((N, dim))
                R = Rst[i_time, :, :]
                V = Vst[i_time, :, :]
                if use_VirtualCoupling:
                    R_lead = np.empty((N, dim))
                    V_lead = np.empty((N, dim))
                    R_lead = Rst_lead[i_time, :, :]
                    V_lead = Vst_lead[i_time, :, :]
                else:
                    R_lead = None
                    V_lead = None
                drag_truth_test[i_time, :, :] = drag_sim(R, V, R_lead, V_lead, None).reshape(N, dim)
            # for i_N in range(N):
            #     plot_data_example(drag_truth_test, "TestTruth", "Drag", test_timestamp, i_N)
            print("Drag: test data ground truth saved\n")

    # ********************************** Filter ******************************************

    filter_freq_cutoff = 100.0
    filter_freq_sample = 1000.0
    filter_order = 5

    def lowpass_butter(cutoff, fs, order=5):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def filter_lowpass(data, cutoff, fs, order=5):
        b, a = lowpass_butter(cutoff, fs, order=order)
        y = filtfilt(b, a, data)  # Use filtfilt for zero-phase filtering
        return y

    def plot_frequency_spectrum(data_before, data_after, fs, data_type, data_meaning):
        fig, axis = plt.subplots(3, 1)
        label_title = ['FX', 'FY', 'FZ']
        for i, ax in enumerate(axis):
            n = len(data_before[:, i])
            k = np.arange(n)
            T = n / fs
            frq = k / T
            frq = frq[range(n // 2)]  # one side frequency range
            Y = np.fft.fft(data_before[:, i]) / n  # FFT and normalization
            Y = Y[range(n // 2)]
            ax.plot(frq, np.abs(Y), label='Raw')

            n = len(data_after[:, i])
            k = np.arange(n)
            T = n / fs
            frq = k / T
            frq = frq[range(n // 2)]  # one side frequency range
            Y = np.fft.fft(data_after[:, i]) / n  # FFT and normalization
            Y = Y[range(n // 2)]
            ax.plot(frq, np.abs(Y), label='Filtered')

            if i == 2:
                ax.set_xlabel('Freq (Hz)')
            ax.set_ylabel('Amplitude')
            ax.set_title(label_title[i])
            ax.legend()
            ax.grid(which='major', color='k', linestyle='-')  # Major grid
            ax.grid(which='minor', color='gray', linestyle=':', linewidth=0.5)  # Minor grid
            ax.minorticks_on()
        plt.suptitle(f"FFT: Filtered ({data_meaning}, {data_type})")
        plt.savefig(_filename(f"dataset_FFT_{data_meaning}_{data_type}.png"))

    def plot_filter_frequency_response(cutoff, fs, order, data_type, data_meaning):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        w, h = freqz(b, a, worN=8000)
        plt.figure()
        plt.plot(0.5 * fs * w / np.pi, np.abs(h), 'b')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.grid(which='major', color='k', linestyle='-')  # Major grid
        plt.grid(which='minor', color='gray', linestyle=':', linewidth=0.5)  # Minor grid
        plt.minorticks_on()
        plt.title(f"Filter Frequency Response ({data_meaning}, {data_type})")
        plt.savefig(_filename(f"dataset_LPF_{data_meaning}_{data_type}.png"))

    # acceleration of top point, test dataset

    # Fst_vel = np.zeros_like(Rst)
    # for i_time in range(1, test_num):
    #     for i_N in range(N):
    #         delta_vel = Vst[i_time, i_N, :] - Vst[i_time - 1, i_N, :]
    #         delta_t = test_timestamp[i_time] - test_timestamp[i_time - 1]
    #         Fst_vel[i_time, i_N, :] = delta_vel / delta_t
    # for i_N in range(N):
    #     plot_data_example(Fst_vel, "Test", f"AccelerationVelocity", test_timestamp, index=i_N)

    # Fst_vel_aver = np.zeros_like(Rst)
    # for i_time in range(1, test_num):
    #     Fst_vel_aver[i_time, :, :] = (Fst_vel[i_time, :, :] + Fst_vel[i_time - 1, :, :]) / 2
    # for i_N in range(N):
    #     plot_data_example(Fst_vel_aver, "Test", f"AccelerationVelocityAver", test_timestamp, index=i_N)

    Fs_pos = np.zeros_like(Rs)
    for i_time in range(1, train_num - 1):
        for i_N in range(N):
            pos_p1 = Rs[i_time + 1, i_N, :]
            pos_n1 = Rs[i_time - 1, i_N, :]
            pos = Rs[i_time, i_N, :]
            delta_t = (train_timestamp[i_time + 1] - train_timestamp[i_time - 1]) / 2
            Fs_pos[i_time, i_N, :] = (pos_p1 + pos_n1 - 2 * pos) / (delta_t ** 2)
    for i_N in range(N):
        plot_data_example(Fs_pos, "Train", f"AccelerationPosition", train_timestamp, index=i_N)

    Fst_pos = np.zeros_like(Rst)
    for i_time in range(1, test_num - 1):
        for i_N in range(N):
            pos_p1 = Rst[i_time + 1, i_N, :]
            pos_n1 = Rst[i_time - 1, i_N, :]
            pos = Rst[i_time, i_N, :]
            delta_t = (test_timestamp[i_time + 1] - test_timestamp[i_time - 1]) / 2
            Fst_pos[i_time, i_N, :] = (pos_p1 + pos_n1 - 2 * pos) / (delta_t ** 2)
    for i_N in range(N):
        plot_data_example(Fst_pos, "Test", f"AccelerationPosition", test_timestamp, index=i_N)

    # Fst_pos_aver = np.zeros_like(Rst)
    # for i_time in range(1, test_num):
    #     Fst_pos_aver[i_time, :, :] = (Fst_pos[i_time, :, :] + Fst_pos[i_time - 1, :, :]) / 2
    # for i_N in range(N):
    #     plot_data_example(Fst_pos_aver, "Test", f"AccelerationPositionAver", test_timestamp, index=i_N)

    plot_filter_frequency_response(filter_freq_cutoff,
                                   filter_freq_sample,
                                   filter_order,
                                   "Test",
                                   "AccelerationPositionLPF")

    Fs_pos_LPF = np.zeros_like(Rs)
    for i_N in range(N):
        for i_dim in range(dim):
            Fs_pos_LPF[:, i_N, i_dim] = filter_lowpass(Fs_pos[:, i_N, i_dim].reshape(-1),
                                                       filter_freq_cutoff,
                                                       filter_freq_sample,
                                                       filter_order)

    Fst_pos_LPF = np.zeros_like(Rst)
    for i_N in range(N):
        for i_dim in range(dim):
            Fst_pos_LPF[:, i_N, i_dim] = filter_lowpass(Fst_pos[:, i_N, i_dim].reshape(-1),
                                                        filter_freq_cutoff,
                                                        filter_freq_sample,
                                                        filter_order)
    plot_frequency_spectrum(Fst_pos[:, 1, :],
                            Fst_pos_LPF[:, 1, :],
                            filter_freq_sample,
                            "Test",
                            "AccelerationPosition")
    for i_N in range(N):
        plot_data_example(Fst_pos_LPF, "Test", f"AccelerationPositionLPF", test_timestamp, index=i_N)

    if execute_learn and use_KalmanFilter:
        Rs = np.array(Rs)
        Vs = np.array(Vs)
        Fs_Kalman = np.zeros_like(Rs)
        for i in range(train_init_num):
            (Rs[train_num_i * i:train_num_i * (i + 1), :, :],
             Vs[train_num_i * i:train_num_i * (i + 1), :, :],
             Fs_Kalman[train_num_i * i:train_num_i * (i + 1), :, :]) = apply_KalmanFilter(
                train_timestamp[train_num_i * i:train_num_i * (i + 1)],
                train_num_i,
                Rs[train_num_i * i:train_num_i * (i + 1), :, :],
                N)
            (Rs_lead[train_num_i * i:train_num_i * (i + 1), :, :],
             Vs_lead[train_num_i * i:train_num_i * (i + 1), :, :],
             _) = apply_KalmanFilter(
                train_timestamp[train_num_i * i:train_num_i * (i + 1)],
                train_num_i,
                Rs_lead[train_num_i * i:train_num_i * (i + 1), :, :],
                1)
        Fst_Kalman = np.zeros_like(Rst)
        Rst, Vst, Fst_Kalman = apply_KalmanFilter(test_timestamp, test_num, Rst, N)
        Rst_lead, Vst_lead, _ = apply_KalmanFilter(test_timestamp, test_num, Rst_lead, 1)

        # for i_N in range(N):
        #     plot_data_example(Fst_Kalman, "Test", "AccelerationKalman", test_timestamp, i_N)

    # ****************************************************

    if execute_learn:
        print("Rs shape = {}, Vs shape = {}\n".format(Rs.shape, Vs.shape))
        print("Rst shape = {}, Vst shape = {}\n".format(Rst.shape, Vst.shape))
        plot_data_example(Rs, "Train", "Position", train_timestamp)
        plot_data_example(Vs, "Train", "Velocity", train_timestamp)
        plot_data_example(Rst, "Test", "Position", test_timestamp)
        plot_data_example(Vst, "Test", "Velocity", test_timestamp)

    # compute ground truth acceleration from Lagrangian mechanics

    if use_SimData:
        force_fn_truth = force_fn_sim
    if use_RealData:  # needs to be changed if used
        acceleration_fn_orig = lnn.accelerationFull(N, dim,
                                                    lagrangian=Lsim,
                                                    constraints=constraints_sim)

        def force_fn_truth(R, V, params, mass):
            if mass is None:
                return acceleration_fn_orig(R, V, params)
            else:
                return acceleration_fn_orig(R, V, params) * mass.reshape(-1, 1)

    Fs = v_acceleration_fn_sim(Rs, Vs, Rs_lead, Vs_lead, None)
    Fs_feedback = np.empty((train_num, dim))
    for i in range(train_num):
        Fs_feedback[i, :] = get_force_feedback(x=Rs[i, :, :], v=Vs[i, :, :],
                                               x_lead=Rs_lead[i, :, :], v_lead=Vs_lead[i, :, :])

    Fst = v_acceleration_fn_sim(Rst, Vst, Rst_lead, Vst_lead, None)
    Fst_feedback = np.empty((test_num, dim))
    for i in range(test_num):
        Fst_feedback[i, :] = get_force_feedback(x=Rst[i, :, :], v=Vst[i, :, :],
                                                x_lead=Rst_lead[i, :, :], v_lead=Vst_lead[i, :, :])

    if execute_learn:
        print("Fs shape = {}, Fst shape = {}\n".format(Fs.shape, Fst.shape))

        for i_N in range(N):
            # plot_data_example(Fs, "Train", f"AccelerationLagrangian", train_timestamp, index=i_N)
            plot_data_example(Fst, "Test", f"AccelerationLagrangian", test_timestamp, index=i_N)

        print("Fs_feedback shape = {}, Fst feedback shape = {}\n".format(Fs_feedback.shape, Fs_feedback.shape))
        plot_data_example(Fs_feedback.reshape(-1, 1, dim), "Train", "ForceFeedback", train_timestamp, index=0)
        plot_data_example(Fst_feedback.reshape(-1, 1, dim), "Test", "ForceFeedback", test_timestamp, index=0)

        plot_trajectory(data_traj=Rs[0:train_num_i, :, :], data_num=train_num_i,
                        data_time=train_timestamp[0:train_num_i], data_type="TrainTruth",
                        data_force=Fs_feedback, data_virtual=Rs_lead[0:train_num_i, :, :])
        print("Trajectory: training data ground truth saved\n")
        plot_trajectory(data_traj=Rst, data_num=test_num,
                        data_time=test_timestamp, data_type="TestTruth",
                        data_force=Fst_feedback, data_virtual=Rst_lead)
        print("Trajectory: test data ground truth saved\n")

    plt.show()
    # input("Check and press Enter to continue...\n")

    # Fs = Fs_pos_LPF
    # Fst = Fst_pos_LPF
    # ******************** Build LGNN Model ********************

    print("\n******************** Build LGNN Model ********************\n")

    #################
    ### Pendulum ####
    #################

    print("\n>> Pendulum\n")

    Ef_pen = 1  # eij dim
    Nf_pen = dim
    Oh_pen = 1

    Eei_pen = 5
    Nei_pen = 5

    hidden_pen = 5
    nhidden_pen = 2

    def get_layers_pen(in_, out_):
        return [in_] + [hidden_pen] * nhidden_pen + [out_]

    def mlp_pen(in_, out_, key, **kwargs):
        return initialize_mlp(get_layers_pen(in_, out_), key, **kwargs)

    fneke_params_pen = initialize_mlp([Oh_pen, Nei_pen], key)
    fne_params_pen = initialize_mlp([Oh_pen, Nei_pen], key)
    fde_params_pen = initialize_mlp([3, Nei_pen], key)

    fb_params_pen = mlp_pen(Ef_pen, Eei_pen, key)
    fbangle_params_pen = mlp_pen(Ef_pen, Eei_pen, key)

    fv_params_pen = mlp_pen(Nei_pen + Eei_pen, Nei_pen, key)

    fe_params_pen = mlp_pen(Nei_pen, Eei_pen, key)
    feangle_params_pen = mlp_pen(1, Eei_pen, key)

    ff1_params_pen = mlp_pen(Eei_pen, 1, key)
    ff1angle_params_pen = mlp_pen(Eei_pen, 1, key)

    ff2_params_pen = mlp_pen(Nei_pen, 1, key)
    ff3_params_pen = mlp_pen(dim + Nei_pen, 1, key)
    ke_params_pen = initialize_mlp([1 + Nei_pen, 10, 10, 1], key, affine=[True])

    Lparams_pen = dict(fb=fb_params_pen,
                       fbangle=fbangle_params_pen,
                       fv=fv_params_pen,
                       fe=fe_params_pen,
                       feangle=feangle_params_pen,
                       ff1=ff1_params_pen,
                       ff1angle=ff1angle_params_pen,
                       ff2=ff2_params_pen,
                       ff3=ff3_params_pen,
                       fne=fne_params_pen,
                       fneke=fneke_params_pen,
                       fde=fde_params_pen,
                       ke=ke_params_pen)

    params_pen = {"L": Lparams_pen}

    params_pen["drag_general"] = initialize_mlp([dim * 2, dim * 2, dim * 2, dim], key)

    if use_tension:
        params_pen["drag_tension"] = initialize_mlp([1, 1], key)
    if use_bending:
        params_pen["object_bending"] = initialize_mlp([N, N * 2, N], key)
    if use_twisting:
        params_pen["object_twisting"] = initialize_mlp([N, N * 2, N], key)
    if use_stretching:
        params_pen["object_stretching"] = initialize_mlp([N, N * 2, N], key)
    if use_damping:
        params_pen["object_damping"] = initialize_mlp([N, N], key)

    def Lmodel_pen(x, v, params):
        result = apply_fn_pen(x, v, params["L"])
        if use_bending:
            result -= pot_energy_model_bending(x, v, params["object_bending"])
        if use_twisting:
            result -= pot_energy_model_twisting(x, v, params["object_twisting"])
        if use_stretching:
            result -= pot_energy_model_stretching(x, v, params["object_stretching"])
        return result

    acceleration_fn_model_pen = jit(lnn.accelerationFull(N_pen, dim,
                                                         lagrangian=Lmodel_pen,
                                                         non_conservative_forces=damping_model_pen,
                                                         external_force=drag_model_pen,
                                                         constraints=constraints_pen))
    v_acceleration_fn_model_pen = vmap(acceleration_fn_model_pen, in_axes=(0, 0, 0, 0, None))

    params = params_pen
    acceleration_fn_model = acceleration_fn_model_pen
    v_acceleration_fn_model = v_acceleration_fn_model_pen
    constraints_model = constraints_pen
    drag_model = drag_model_pen
    damping_model = damping_model_pen

    def force_fn_model(R, V, R_lead, V_lead, params, mass):
        if mass is None:
            return acceleration_fn_model(R, V, R_lead, V_lead, params)
        else:
            return acceleration_fn_model(R, V, R_lead, V_lead, params) * mass.reshape(-1, 1)

    def get_forward_model(params=None, force_fn=None, runs=10):
        @jit
        def fn(R, V, R_lead, V_lead):
            return predition(R, V, R_lead, V_lead, params, force_fn, shift, dt=dt, mass=masses,
                             stride=stride, runs=runs)

        return fn

    def pred_recursive(data_pos, data_vel, num_timestamp, pred_model, points_known=1):
        pred_pos_recur = np.empty((num_timestamp, N, dim))
        pred_vel_recur = np.empty((num_timestamp, N, dim))
        R = np.empty((N, dim))
        V = np.empty((N, dim))
        pred_pos_recur[0, :, :] = data_pos[0, :, :]  # initial position
        pred_vel_recur[0, :, :] = data_vel[0, :, :]  # initial velocity
        for i in range(num_timestamp - 1):
            R[0:points_known, :] = data_pos[i, 0:points_known, :]  # information that we have
            V[0:points_known, :] = data_vel[i, 0:points_known, :]
            R[points_known:, :] = pred_pos_recur[i, points_known:, :]  # recursive information
            V[points_known:, :] = pred_vel_recur[i, points_known:, :]
            pred_temp = pred_model(R, V)
            pred_temp_pos = pred_temp.position
            pred_temp_vel = pred_temp.velocity
            pred_pos_recur[i + 1, :, :] = pred_temp_pos
            pred_vel_recur[i + 1, :, :] = pred_temp_vel
        return pred_pos_recur, pred_vel_recur

    # ******************** Learning process ********************

    if execute_learn:
        print("\n******************** Learning process ********************\n")

        LOSS = getattr(src.models, error_fn)
        print(f"LOSS = {error_fn}\n")

        @jit  # comment @jit to enable information print
        def loss_fn(params, Rs, Vs, Fs, Rs_lead, Vs_lead):
            # print("Rs shape = {}, Vs shape = {}, Fs shape = {}\n".format(Rs.shape, Vs.shape, Fs.shape))
            loss_total = 0.0

            pred_acc = v_acceleration_fn_model(Rs, Vs, Rs_lead, Vs_lead, params)
            # print(f"pred_acc shape = {pred_acc.shape}")
            loss_acc = LOSS(pred_acc, Fs)
            return loss_acc

        @jit
        def gloss(*args):
            return value_and_grad(loss_fn)(*args)

        opt_init, opt_update_, get_params = optimizers.adam(lr)

        @jit
        def opt_update(i, grads_, opt_state):
            grads_ = jax.tree_map(jnp.nan_to_num, grads_)
            grads_ = jax.tree_map(
                partial(jnp.clip, a_min=-1000.0, a_max=1000.0), grads_)
            return opt_update_(i, grads_, opt_state)

        @jit
        def update(i, opt_state, params, loss__, *data):
            """ Compute the gradient for a batch and update the parameters """
            value, grads_ = gloss(params, *data)
            opt_state = opt_update(i, grads_, opt_state)
            return opt_state, get_params(opt_state), value

        @jit
        def step(i, ps, *args):
            return update(i, *ps, *args)

        def batching(*args, size=None):
            L = len(args[0])
            if size != None:
                nbatches1 = int((L - 0.5) // size) + 1
                nbatches2 = max(1, nbatches1 - 1)
                size1 = int(L / nbatches1)
                size2 = int(L / nbatches2)
                if size1 * nbatches1 > size2 * nbatches2:
                    size = size1
                    nbatches = nbatches1
                else:
                    size = size2
                    nbatches = nbatches2
            else:
                nbatches = 1
                size = L

            newargs = []
            for arg in args:
                newargs += [jnp.array([arg[i * size:(i + 1) * size]
                                       for i in range(nbatches)])]
            return newargs

        bRs, bVs, bFs, bRs_lead, bVs_lead = batching(Rs, Vs, Fs, Rs_lead, Vs_lead, size=min(len(Rs), batch_size))
        print("bRs shape = {}, bVs shape = {}, bFs shape = {}, bRs_lead = {}, bVs_lead = {}\n".format(bRs.shape,
                                                                                                      bVs.shape,
                                                                                                      bFs.shape,
                                                                                                      bRs_lead.shape,
                                                                                                      bVs_lead.shape))

        # training

        opt_state = opt_init(params)
        epoch = 0
        optimizer_step = -1
        larray = []
        ltarray = []
        last_loss = 1000
        larray += [loss_fn(params, Rs, Vs, Fs, Rs_lead, Vs_lead)]
        ltarray += [loss_fn(params, Rst, Vst, Fst, Rst_lead, Vst_lead)]

        def print_loss():
            print(f"Epoch: {epoch}/{epochs} Loss (mean of {error_fn}):  train={larray[-1]}, test={ltarray[-1]}")

        print_loss()

        for epoch in range(epochs):
            for data in zip(bRs, bVs, bFs, bRs_lead, bVs_lead):
                optimizer_step += 1
                opt_state, params, l_ = step(optimizer_step, (opt_state, params, 0), *data)

            # optimizer_step += 1
            # opt_state, params, l_ = step(
            #     optimizer_step, (opt_state, params, 0), Rs, Vs, Fs)

            larray += [loss_fn(params, Rs, Vs, Fs, Rs_lead, Vs_lead)]
            ltarray += [loss_fn(params, Rst, Vst, Fst, Rst_lead, Vst_lead)]

            if epoch % saveat == 0:
                print_loss()
                metadata = {
                    "savedat": epoch,
                    "mpass": mpass,
                    "grid": grid,
                    "use_drag": use_drag,
                    "trainm": trainm,
                }
                savefile(f"lgnn_trained_model_{use_drag}_{trainm}.dil",
                         params, metadata=metadata)
                savefile(f"loss_array_{use_drag}_{trainm}.dil",
                         (larray, ltarray), metadata=metadata)
                if last_loss > larray[-1]:
                    last_loss = larray[-1]
                    savefile(f"lgnn_trained_model_{use_drag}_{trainm}_low.dil",
                             params, metadata=metadata)

        fig, axs = plt.subplots(1, 1)
        plt.plot(larray, label="Training")
        plt.plot(ltarray, label="Test")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(_filename(f"training_loss_{use_drag}_{trainm}.png"))

        metadata = {
            "savedat": epoch,
            "mpass": mpass,
            "grid": grid,
            "use_drag": use_drag,
            "trainm": trainm,
        }
        params = get_params(opt_state)
        savefile(f"lgnn_trained_model_{use_drag}_{trainm}.dil",
                 params, metadata=metadata)
        savefile(f"loss_array_{use_drag}_{trainm}.dil",
                 (larray, ltarray), metadata=metadata)

        # ******************** LGNN evaluation ********************

        print("\n******************** LGNN evaluation ********************\n")

        forward_model_trained_onestep = get_forward_model(params=params, force_fn=force_fn_model, runs=1)

        print(">> Plot acceleration prediction results\n")

        label_title_ax = ["X", "Y", "Z"]

        pred_acc_train = v_acceleration_fn_model(Rs, Vs, Rs_lead, Vs_lead, params)
        # print("pred_acc_train = {}".format(pred_acc_train))
        # print("Fs = {}".format(Fs))
        print(f"pred_acc_train shape = {pred_acc_train.shape}\n")
        for i_marker in range(N):
            fig, axis = plt.subplots(3, 1)
            for i_ax, ax in enumerate(axis):
                ax.plot(train_timestamp, Fs[:, i_marker, i_ax], label="Ground Truth")
                ax.plot(train_timestamp, pred_acc_train[:, i_marker, i_ax], label="Prediction")
                ax.set_title(label_title_ax[i_ax])
                ax.legend()
            plt.suptitle(f"Training Results for Marker {i_marker}")
            plt.tight_layout()
            plt.savefig(_filename(f"training_result_{use_drag}_{trainm}_compare_Marker{i_marker}.png"))
        print("Training results saved\n")

        pred_acc_test = v_acceleration_fn_model(Rst, Vst, Rst_lead, Vst_lead, params)
        print(f"pred_acc_test shape = {pred_acc_test.shape}\n")
        for i_marker in range(N):
            fig, axis = plt.subplots(3, 1)
            for i_ax, ax in enumerate(axis):
                ax.plot(test_timestamp, Fst[:, i_marker, i_ax], label="Ground Truth")
                ax.plot(test_timestamp, pred_acc_test[:, i_marker, i_ax], label="Prediction")
                ax.set_title(label_title_ax[i_ax])
                ax.legend()
            plt.suptitle(f"Test Results for Marker {i_marker}")
            plt.tight_layout()
            plt.savefig(_filename(f"test_result_{use_drag}_{trainm}_compare_Marker{i_marker}.png"))
        print("Test results saved\n")

        if use_drag:
            print(">> Plot drag prediction results\n")

            drag_truth_train = np.zeros((train_num, N, dim))
            drag_predict_train = np.zeros((train_num, N, dim))
            for i_time in range(train_num):
                R = np.empty((N, dim))
                V = np.empty((N, dim))
                R = Rs[i_time, :, :]
                V = Vs[i_time, :, :]
                if use_VirtualCoupling:
                    R_lead = np.empty((N, dim))
                    V_lead = np.empty((N, dim))
                    R_lead = Rs_lead[i_time, :, :]
                    V_lead = Vs_lead[i_time, :, :]
                drag_truth_train[i_time, :, :] = drag_sim(R, V, R_lead, V_lead, None).reshape(N, dim)
                drag_predict_train[i_time, :, :] = drag_model(R, V, R_lead, V_lead, params).reshape(N, dim)
            for i_N in range(N):
                plot_data_example(drag_truth_train, "TrainTruth", "Drag", train_timestamp, i_N)
                plot_data_example(drag_predict_train, "TrainPredict", "Drag", train_timestamp, i_N)
            print("Training results saved\n")

            drag_truth_test = np.zeros((test_num, N, dim))
            drag_predict_test = np.zeros((test_num, N, dim))
            for i_time in range(test_num):
                R = np.empty((N, dim))
                V = np.empty((N, dim))
                R = Rst[i_time, :, :]
                V = Vst[i_time, :, :]
                if use_VirtualCoupling:
                    R_lead = np.empty((N, dim))
                    V_lead = np.empty((N, dim))
                    R_lead = Rst_lead[i_time, :, :]
                    V_lead = Vst_lead[i_time, :, :]
                drag_truth_test[i_time, :, :] = drag_sim(R, V, R_lead, V_lead, None).reshape(N, dim)
                drag_predict_test[i_time, :, :] = drag_model(R, V, R_lead, V_lead, params).reshape(N, dim)
            for i_N in range(N):
                plot_data_example(drag_truth_test, "TestTruth", "Drag", test_timestamp, i_N)
                plot_data_example(drag_predict_test, "TestPredict", "Drag", test_timestamp, i_N)
            print("Test results saved\n")

        if use_VirtualCoupling:
            print(">> Plot trajectory predictions\n")

            print("1) Training dataset\n")
            start_time = time.time()
            pred_pos_train = np.empty((train_num_i, N, dim))
            pred_vel_train = np.empty((train_num_i, N, dim))
            pred_pos_train[0, :, :] = Rs[0, :, :]
            pred_vel_train[0, :, :] = Vs[0, :, :]
            for i_time in range(train_num_i - 1):
                index = i_time
                pred_traj_temp = forward_model_trained_onestep(R=pred_pos_train[index, :, :],
                                                               V=pred_vel_train[index, :, :],
                                                               R_lead=Rs_lead[index, :],
                                                               V_lead=Vs_lead[index, :])
                pred_pos_train[index + 1, :, :] = pred_traj_temp.position
                pred_vel_train[index + 1, :, :] = pred_traj_temp.velocity
            end_time = time.time()
            execution_time = end_time - start_time
            execution_time_aver = execution_time / train_num_i
            print(f"Prediction speed (average frequency): {(1.0 / execution_time_aver):.2f} Hz\n")
            plot_trajectory(data_traj=pred_pos_train,
                            data_num=train_num_i,
                            data_time=train_timestamp[0:train_num_i],
                            data_type="TrainPred",
                            data_compare=Rs[0:train_num_i, :, :],
                            data_virtual=Rs_lead[0:train_num_i, :, :])
            print("Training results saved\n")

            print("2) Test dataset\n")
            start_time = time.time()
            pred_pos_test = np.empty((test_num, N, dim))
            pred_vel_test = np.empty((test_num, N, dim))
            pred_pos_test[0, :, :] = Rst[0, :, :]
            pred_vel_test[0, :, :] = Vst[0, :, :]
            for i_time in range(test_num - 1):
                index = i_time
                pred_traj_temp = forward_model_trained_onestep(R=pred_pos_test[index, :, :],
                                                               V=pred_vel_test[index, :, :],
                                                               R_lead=Rst_lead[index, :],
                                                               V_lead=Vst_lead[index, :])
                pred_pos_test[index + 1, :, :] = pred_traj_temp.position
                pred_vel_test[index + 1, :, :] = pred_traj_temp.velocity
            end_time = time.time()
            execution_time = end_time - start_time
            execution_time_aver = execution_time / test_num
            print(f"Prediction speed (average frequency): {(1.0 / execution_time_aver):.2f} Hz\n")
            plot_trajectory(data_traj=pred_pos_test,
                            data_num=test_num,
                            data_time=test_timestamp,
                            data_type="TestPred",
                            data_compare=Rst,
                            data_virtual=Rst_lead)
            print("Test results saved\n")
        else:

            print(">> Plot trajectory predictions\n")

            pred_traj_train = np.empty((train_num_i, N, dim))
            pred_traj_train[0, :, :] = Rs[0, :, :]
            for i in range(train_num_i - 1):
                R = Rs[i, :, :]
                V = Vs[i, :, :]
                if use_VirtualCoupling:
                    R_lead = Rs_lead[i, :, :]
                    V_lead = Vs_lead[i, :, :]
                else:
                    R_lead = None
                    V_lead = None
                pred_traj_temp = forward_model_trained_onestep(R=R, V=V, R_lead=R_lead, V_lead=V_lead)
                pred_traj_temp = pred_traj_temp.position
                pred_traj_train[i + 1, :, :] = pred_traj_temp

            # print("pred_traj_train = {}".format(pred_traj_train))

            plot_trajectory(data_traj=pred_traj_train,
                            data_num=train_num_i,
                            data_time=train_timestamp[0:train_num_i],
                            data_type="TrainPred",
                            data_compare=Rs[0:train_num_i, :, :],
                            data_virtual=Rs_lead[0:train_num_i, :, :])
            print("Training results saved\n")

            pred_traj_test = np.empty((test_num, N, dim))
            pred_traj_test[0, :, :] = Rst[0, :, :]
            for i in range(test_num - 1):
                R = Rst[i, :, :]
                V = Vst[i, :, :]
                if use_VirtualCoupling:
                    R_lead = Rst_lead[i, :, :]
                    V_lead = Vst_lead[i, :, :]
                else:
                    R_lead = None
                    V_lead = None
                pred_traj_temp = forward_model_trained_onestep(R=R, V=V, R_lead=R_lead, V_lead=V_lead)
                pred_traj_temp = pred_traj_temp.position
                pred_traj_test[i + 1, :, :] = pred_traj_temp
            plot_trajectory(data_traj=pred_traj_test,
                            data_num=test_num,
                            data_time=test_timestamp,
                            data_type="TestPred",
                            data_compare=Rst,
                            data_virtual=Rst_lead)
            print("Test results saved\n")
            print(f"pred_traj_train shape = {pred_traj_train.shape}, pred_traj_test shape = {pred_traj_test.shape}\n")

            print(">> Plot free prediction results\n")

            sim_model_free = get_forward_model(params=params, force_fn=force_fn_model, runs=train_num_i)
            pred_traj_temp = sim_model_free(Rs[0, :, :], Vs[0, :, :], Rs_lead[0, :, :], Vs_lead[0, :, :])
            pred_traj_temp = pred_traj_temp.position
            pred_traj_train_free = pred_traj_temp
            plot_trajectory(data_traj=pred_traj_train_free,
                            data_num=train_num_i,
                            data_time=train_timestamp[0:train_num_i],
                            data_type="TrainFree",
                            data_compare=Rs[0:train_num_i, :, :])
            print("Training results saved\n")

            sim_model_free = get_forward_model(params=params, force_fn=force_fn_model, runs=test_num)
            pred_traj_temp = sim_model_free(Rst[0, :, :], Vst[0, :, :], Rst_lead[0, :, :], Vst_lead[0, :, :])
            pred_traj_temp = pred_traj_temp.position
            pred_traj_test_free = pred_traj_temp
            plot_trajectory(data_traj=pred_traj_test_free,
                            data_num=test_num,
                            data_time=test_timestamp,
                            data_type="TestFree",
                            data_compare=Rst)
            print("Test results saved\n")

            print(">> Recursive prediction\n")

            pred_traj_train_recur, pred_vel_train_recur = pred_recursive(data_pos=Rs[0:train_num_i, :, :],
                                                                         data_vel=Vs[0:train_num_i, :, :],
                                                                         num_timestamp=train_num_i,
                                                                         pred_model=forward_model_trained_onestep,
                                                                         points_known=1)
            plot_trajectory(data_traj=pred_traj_train_recur,
                            data_num=train_num_i,
                            data_time=train_timestamp[0:train_num_i],
                            data_type="TrainPredRecur",
                            data_compare=Rs[0:train_num_i, :, :])
            print("Training results saved\n")

            pred_traj_test_recur, pred_vel_test_recur = pred_recursive(data_pos=Rst, data_vel=Vst,
                                                                       num_timestamp=test_num,
                                                                       pred_model=forward_model_trained_onestep,
                                                                       points_known=1)
            plot_trajectory(data_traj=pred_traj_test_recur,
                            data_num=test_num,
                            data_time=test_timestamp,
                            data_type="TestPredRecur",
                            data_compare=Rst)
            print("Test results saved\n")

    # ******************** Rendering ********************

    if execute_render:
        print("\n******************** Rendering ********************\n")

        params = src.io.loadfile(model_trained_path)[0]
        forward_model_trained_onestep = get_forward_model(params=params, force_fn=force_fn_model, runs=1)
        print(">> Trained model loaded, ready for rendering\n")

        object_position = Rst[0, :, :]
        object_velocity = Vst[0, :, :]
        com_count = -1
        last_force = 0.0
        force_exceeded = False

        # Initialize the Kalman filter
        kf = KalmanFilter3D(process_var, measurement_var, estimated_var)
        # Initial state
        kf.x[:3] = [0.0, 0.0, 0.0]

        save_history = False

        if save_history:
            hist_user_position_processed = []
            hist_user_position_orig = []
            hist_user_velocity = []
            hist_object_position = []
            hist_object_velocity = []
            hist_force = []
            # hist_force_magnitude_orig = []
            # hist_force_magnitude_processed = []
            # hist_force_max_orig = []
            # hist_force_max_processed = []

        hist_time = []
        hist_execution_time_com = []
        hist_execution_time_CalForce = []
        hist_execution_time_CalDynamic = []
        hist_execution_time_render = []

        hist_com_count = []

        start_time_global = time.time()
        while True:

            start_time_com = time.time()

            com_count += 1
            data_byte, address = sock.recvfrom(BUFLEN)
            data_received = bool(data_byte)
            # print(f"Received: {data_byte}")
            data_json = data_byte.decode('utf-8')
            data_dict = json.loads(data_json)  # dictionary
            data_position = data_dict.get('position', [])
            data_timestamp = data_dict.get('timestamp')
            if not use_KalmanFilter:
                data_velocity = data_dict.get('velocity', [])

            # print(f"Position = {data_position}", )
            # print(f"Velocity = {data_velocity}", )
            # print(f"Timestamp = {data_timestamp}")

            # cipher for stopping
            if data_timestamp < 0:
                break

            end_time_global = time.time()
            diff_time_global = end_time_global - start_time_global
            end_time_com = time.time()
            communication_time = end_time_com - start_time_com

            # compute force from position

            start_time_cal_force = time.time()

            hist_time.extend([diff_time_global])
            # hist_time.extend([data_timestamp * 1e-3])
            hist_com_count.extend([com_count])
            user_position = np.array([data_position]) * 1e-3  # mm to m
            user_position = get_coordinate_TouchX2Python(user_position)

            if save_history:
                hist_user_position_orig.append(user_position)

            if use_KalmanFilter:
                if com_count <= 1:
                    user_position = np.zeros((1, dim))
                    user_velocity = np.zeros((1, dim))
                else:
                    hist_time_np = np.array(hist_time).reshape(-1)
                    delta_time = hist_time_np[com_count] - hist_time_np[com_count - 1]
                    # delta_time = 0.001
                    kf.predict(delta_time)

                    kf.update(user_position.reshape(dim))
                    state = kf.get_state()

                    # estimated_position = np.zeros((1, dim))
                    # for i_dim in range(dim):
                    #     estimated_position[0, i_dim] = state[i_dim]
                    #
                    # estimated_velocity = np.zeros((1, dim))
                    # for i_dim in range(dim):
                    #     estimated_velocity[0, i_dim] = state[3 + i_dim]

                    estimated_position = state[:3]
                    estimated_velocity = state[3:6]
                    user_position = estimated_position.reshape(1, dim)
                    user_velocity = estimated_velocity.reshape(1, dim)
            else:
                user_velocity = np.array([data_velocity])
                user_velocity = get_coordinate_TouchX2Python(user_velocity)

            if save_history:
                hist_user_position_processed.append(user_position.tolist())
                hist_user_velocity.append(user_velocity.tolist())

            # print(f"user_position = {user_position}, user_velocity = {user_velocity}")

            # print(f"user_position shape = {user_position.shape}")
            # print(f"object_position shape = {object_position.shape}")
            # print(f"Rst_lead shape = {Rst_lead[0, :, :].shape}")

            # object_acceleration = acceleration_fn_model(x=object_position,
            #                                             v=object_velocity,
            #                                             x_lead=user_position,
            #                                             v_lead=user_velocity,
            #                                             params=params)

            force = get_force_feedback(x=object_position,
                                       v=object_velocity,
                                       x_lead=user_position,
                                       v_lead=user_velocity)

            # force test:
            # 2.0 yes
            # 2.5 ok but at corners will exist some sound
            # 3.0 ok but at limit (electrical sound), 4.0 no
            force = np.array(force) * 0.5

            force_max = np.max(np.abs(force))
            # if save_history:
            #     hist_force_max_orig.append(force_max)
            if force_max > 2.5:
                force = force / force_max * 2.5  # can't do /= force_max * 1.7, will cause some unexpected value ~0.7
            # if save_history:
            #     hist_force_max_processed.append(np.max(np.abs(force)))

            force_magnitude = np.linalg.norm(force)
            # if save_history:
            #     hist_force_magnitude_orig.append(force_magnitude.tolist())
            if force_magnitude > 7.0:
                force = force / force_magnitude * 7.0
            force_magnitude = np.linalg.norm(force)
            # if save_history:
            #     hist_force_magnitude_processed.append(force_magnitude.tolist())

            # for i_dim in range(dim):
            #     if force[i_dim] > 7.0 or force[i_dim] < -7.0:
            #         force_exceeded = True
            #         error_force.append(force.tolist())
            #         break
            #     else:
            #         force_exceeded = False
            # if force_exceeded:
            #     force = last_force
            # else:
            #     last_force = force

            # force = np.array(force)
            # np.where(force > 7.0, 7.0, force)
            # np.where(force < -7.0, -7.0, force)

            if save_history:
                hist_force.append(force.tolist())

            end_time_cal_force = time.time()
            calculation_time_force = end_time_cal_force - start_time_cal_force

            # predict next step dynamic

            start_time_cal_dynamic = time.time()

            if save_history:
                hist_object_position.append(object_position.tolist())
                hist_object_velocity.append(object_velocity.tolist())

            pred_traj = forward_model_trained_onestep(R=object_position,
                                                      V=object_velocity,
                                                      R_lead=user_position,
                                                      V_lead=user_velocity)

            # object_position = np.zeros((N, dim))
            # object_velocity = np.zeros((N, dim))
            # for i_dim in range(dim):
            #     for i_N in range(N):
            #         object_position[i_N, i_dim] = pred_traj.position[0,i_N, i_dim]
            #         object_velocity[i_N, i_dim] = pred_traj.velocity[0,i_N, i_dim]
            object_position = pred_traj.position.reshape(N, dim)  # [0,:,:] is causing time spikes, and only this line
            object_velocity = pred_traj.velocity.reshape(N, dim)  # even used [0,:,:] as well, this line doesn't cause
            end_time_cal_dynamic = time.time()

            calculation_time_dynamic = end_time_cal_dynamic - start_time_cal_dynamic

            # if calculation_time_dynamic > 0.05:
            #     print(f"pred_traj = {pred_traj}")
            #     print(f"pred_traj.position.shape = {pred_traj.position.shape}")
            #     print(f"pred_traj.velocity.shape = {pred_traj.velocity.shape}")
            #     print("Time spike encountered!!!")
            #     break

            calculation_time = calculation_time_force + calculation_time_dynamic

            # return force
            start_time_com = time.time()
            output_dict = dict(force=get_coordinate_Python2TouchX(force).tolist())
            if data_received:
                output_json = json.dumps(output_dict)
                output_byte = output_json.encode('utf-8')
                output_length = sock.sendto(output_byte, address)
            end_time_com = time.time()

            communication_time += end_time_com - start_time_com
            render_time = communication_time + calculation_time

            hist_execution_time_com.append(communication_time)
            hist_execution_time_CalForce.append(calculation_time_force)
            hist_execution_time_CalDynamic.append(calculation_time_dynamic)
            hist_execution_time_render.append(render_time)

            # print(f"force = {force}")
            # print(f"average update rate = {(1.0 / (render_time + const_numerical)):.2f} Hz")
            # print(f"communication: {(communication_time * 1000.0):.2f} ms, "
            #       f"ratio {(communication_time / (render_time + const_numerical) * 100.0):.2f}%")
            # print(f"calculation: {(calculation_time * 1000.0):.2f} ms, "
            #       f"ratio {(calculation_time / (render_time + const_numerical) * 100.0):.2f}%")
            # print(f"cal - force: {(calculation_time_force * 1000.0):.2f} ms, "
            #       f"ratio {(calculation_time_force / (calculation_time + const_numerical) * 100.0):.2f}%")
            # print(f"cal - dynamic: {(calculation_time_dynamic * 1000.0):.2f} ms, "
            #       f"ratio {(calculation_time_dynamic / (calculation_time + const_numerical) * 100.0):.2f}%\n")

        if save_history:
            hist_user_position_orig = np.array(hist_user_position_orig)
            hist_user_position_processed = np.array(hist_user_position_processed)
            hist_user_velocity = np.array(hist_user_velocity)
            hist_force = np.array(hist_force)
            hist_object_position = np.array(hist_object_position)
            hist_object_velocity = np.array(hist_object_velocity)
            print(f"hist_user_position_orig shape = {hist_user_position_orig.shape}")
            print(f"hist_user_velocity shape = {hist_user_velocity.shape}")
            print(f"hist_object_position shape = {hist_object_position.shape}")
            print(f"hist_object_velocity shape = {hist_object_velocity.shape}")
            print(f"hist_force shape = {hist_force.shape}")

            plot_trajectory(data_traj=hist_object_position,
                            data_num=len(hist_time),
                            data_time=hist_time,
                            data_type="render",
                            data_force=hist_force,
                            data_virtual=hist_user_position_processed)
            print(">> Render trajectory saved\n")

            fig, axis = plt.subplots(3, 1)
            label_title_ax = ["X", "Y", "Z"]
            for i, ax in enumerate(axis):
                ax.plot(hist_time, hist_force[:, i])
                ax.set_title(label_title_ax[i])
            plt.suptitle(f"Render Force Observation")
            plt.tight_layout()
            plt.savefig(_filename(f"render_force.png"))
            print(">> Plot render force\n")

            fig, axis = plt.subplots(3, 1)
            for i, ax in enumerate(axis):
                ax.plot(hist_time, hist_user_position_orig[:, 0, i] * 1e3, label='Origin')
                ax.plot(hist_time, hist_user_position_processed[:, 0, i] * 1e3, label='Processed')
                ax.set_title(label_title_ax[i])
                ax.legend()
            plt.suptitle(f"User Position Observation")
            plt.tight_layout()
            plt.savefig(_filename(f"render_user_position.png"))
            print(">> Plot user position\n")

            fig, axis = plt.subplots(3, 1)
            for i, ax in enumerate(axis):
                ax.plot(hist_time, hist_user_velocity[:, 0, i] * 1e3)
                ax.set_title(label_title_ax[i])
            plt.suptitle(f"User Velocity Observation")
            plt.tight_layout()
            plt.savefig(_filename(f"render_user_velocity.png"))
            print(">> Plot user velocity\n")

            # print(f"error_force = {error_force}\n")

            fig, axis = plt.subplots(2, 1)
            axis[0].plot(hist_time, hist_force_magnitude_orig)
            axis[0].set_title("Force Magnitude (Origin)")
            axis[1].plot(hist_time, hist_force_magnitude_processed)
            axis[1].set_title("Force Magnitude (Processed)")
            plt.suptitle("Force Magnitude Observation")
            plt.tight_layout()
            plt.savefig(_filename(f"render_force_magnitude.png"))
            print(">> Plot force magnitude\n")

            fig, axis = plt.subplots(2, 1)
            axis[0].plot(hist_time, hist_force_max_orig)
            axis[0].set_title("Force Max (Origin)")
            axis[1].plot(hist_time, hist_force_max_processed)
            axis[1].set_title("Force Max (Processed)")
            plt.suptitle("Force Max Observation")
            plt.tight_layout()
            plt.savefig(_filename(f"render_force_max.png"))
            print(">> Plot force max\n")

        hist_time = np.array(hist_time)
        print(f"hist_time shape = {hist_time.shape}")

        fig, axis = plt.subplots(4, 1)
        axis[0].plot(hist_com_count, hist_execution_time_com)
        axis[0].set_yscale('log')
        axis[0].set_title("Communication Time")
        axis[1].plot(hist_com_count, hist_execution_time_CalForce)
        axis[1].set_yscale('log')
        axis[1].set_title("Force Calculation Time")
        axis[2].plot(hist_com_count, hist_execution_time_CalDynamic)
        axis[2].set_yscale('log')
        axis[2].set_title("Dynamic Calculation Time")
        axis[3].plot(hist_com_count, hist_execution_time_render)
        axis[3].set_yscale('log')
        axis[3].set_title("Overall Rendering Time")
        plt.suptitle("Execution Time Observation")
        plt.tight_layout()
        plt.savefig(_filename(f"render_execution_time.png"))
        print(">> Plot execution time\n")

    # ******************** Rendering ********************

    if execute_test:
        print("\n******************** Testing ********************\n")

        params = src.io.loadfile(model_trained_path)[0]
        forward_model_trained_onestep = get_forward_model(params=params, force_fn=force_fn_model, runs=1)
        print(">> Trained model loaded, ready for testing\n")

        save_path = r"C:\YutongZhang\LGNN\tests"
        save_path += f"\\Object{use_object}"

        test_num = 5000  # 5 s
        test_timestamp = np.arange(test_num) * dt * stride
        sweep_num = 50
        save_skip = 10  # 100 Hz data

        # print(">> Frequency sweep\n")
        # freq_start = 0.1
        # freq_stop = 10.0
        # frequencies = np.logspace(np.log10(freq_start), np.log10(freq_stop), sweep_num)
        # magnitude = VirtualCoupling_sim_magnitude_xy
        # result_FreqSweep = []
        # progress = -1
        # for freq in frequencies:
        #     progress += 1
        #     if progress % (int)(sweep_num / 10) == 0:
        #         print(f"Progress {progress}: testing frequency {freq:.4f} Hz")
        #     omega = 2 * np.pi * freq
        #
        #     Rst_lead = np.zeros((test_num, 1, dim))
        #     Vst_lead = np.zeros((test_num, 1, dim))
        #     for i_time in range(test_num):
        #         Rst_lead[i_time, 0, 0] = magnitude * np.sin(omega * test_timestamp[i_time])
        #         Vst_lead[i_time, 0, 0] = omega * magnitude * np.cos(omega * test_timestamp[i_time])
        #         Rst_lead[i_time, 0, 1] = magnitude * np.sin(omega * test_timestamp[i_time])
        #         Vst_lead[i_time, 0, 1] = omega * magnitude * np.cos(omega * test_timestamp[i_time])
        #
        #     R, V = get_sim_data_initial(N, dim, random.PRNGKey(50))
        #
        #     # ground truth
        #     Rst = np.empty((test_num, N, dim))
        #     Vst = np.empty((test_num, N, dim))
        #     Rst[0, :, :] = R
        #     Vst[0, :, :] = V
        #     for i_time in range(test_num - 1):
        #         gen_data_test = get_forward_sim_data(Rst[i_time, :, :],
        #                                              Vst[i_time, :, :],
        #                                              Rst_lead[i_time, :],
        #                                              Vst_lead[i_time, :],
        #                                              1)
        #         Rst[i_time + 1, :, :] = gen_data_test.position
        #         Vst[i_time + 1, :, :] = gen_data_test.velocity
        #
        #     # model prediction
        #     Rsp = np.empty((test_num, N, dim))
        #     Vsp = np.empty((test_num, N, dim))
        #     Rsp[0, :, :] = R
        #     Vsp[0, :, :] = V
        #     for i_time in range(test_num - 1):
        #         pred_traj = forward_model_trained_onestep(R=Rsp[i_time, :, :],
        #                                                   V=Vsp[i_time, :, :],
        #                                                   R_lead=Rst_lead[i_time, :],
        #                                                   V_lead=Vst_lead[i_time, :])
        #         Rsp[i_time + 1, :, :] = pred_traj.position
        #         Vsp[i_time + 1, :, :] = pred_traj.velocity
        #
        #     error_pos = Rsp - Rst
        #     error_vel = Vsp - Vst
        #
        #     for i_time in range(test_num):
        #         if i_time % save_skip == 0:
        #             result = {'Frequency': freq}
        #             result['Time'] = test_timestamp[i_time]
        #             for i_N in range(N):
        #                 for i_dim in range(dim):
        #                     result[f'Position_Error_Node{i_N}_Dim{i_dim}'] = error_pos[i_time, i_N, i_dim]
        #                     result[f'Velocity_Error_Node{i_N}_Dim{i_dim}'] = error_vel[i_time, i_N, i_dim]
        #                     result[f'Position_Predict_Node{i_N}_Dim{i_dim}'] = Rsp[i_time, i_N, i_dim]
        #                     result[f'Velocity_Predict_Node{i_N}_Dim{i_dim}'] = Vsp[i_time, i_N, i_dim]
        #                     result[f'Position_Truth_Node{i_N}_Dim{i_dim}'] = Rst[i_time, i_N, i_dim]
        #                     result[f'Velocity_Truth_Node{i_N}_Dim{i_dim}'] = Vst[i_time, i_N, i_dim]
        #             result_FreqSweep.append(result)
        #
        #     if freq > 5.15:
        #         plot_trajectory(Rsp, test_num, test_timestamp, "Test",
        #                         data_compare=Rst, data_virtual=Rst_lead)
        #         print(freq)
        #         input("stop")
        #
        #     del Rst, Vst
        #     del Rsp, Vsp
        #     del Rst_lead, Vst_lead
        #     del R, V
        #
        # fields = result_FreqSweep[0].keys()
        # with open(save_path + f"_sweep_freq.csv", mode='w', newline='') as file:
        #     writer = csv.DictWriter(file, fieldnames=fields)
        #     writer.writeheader()
        #     writer.writerows(result_FreqSweep)
        #
        # print("\n>> Magnitude sweep\n")
        # magnitude_start = 0.0
        # magnitude_end = 0.4
        # magnitudes = np.linspace(magnitude_start, magnitude_end, sweep_num)
        # freq = 1.0 / VirtualCoupling_sim_period
        # omega = 2 * np.pi * freq
        # result_MagnitudeSweep = []
        # progress = -1
        # for magnitude in magnitudes:
        #     progress += 1
        #     if progress % (int)(sweep_num / 10) == 0:
        #         print(f"Progress {progress}: testing magnitude {magnitude:.4f} m")
        #
        #     Rst_lead = np.zeros((test_num, 1, dim))
        #     Vst_lead = np.zeros((test_num, 1, dim))
        #     for i_time in range(test_num):
        #         Rst_lead[i_time, 0, 0] = magnitude * np.sin(omega * test_timestamp[i_time])
        #         Vst_lead[i_time, 0, 0] = omega * magnitude * np.cos(omega * test_timestamp[i_time])
        #         Rst_lead[i_time, 0, 1] = magnitude * np.sin(omega * test_timestamp[i_time])
        #         Vst_lead[i_time, 0, 1] = omega * magnitude * np.cos(omega * test_timestamp[i_time])
        #
        #     R, V = get_sim_data_initial(N, dim, random.PRNGKey(50))
        #
        #     # ground truth
        #     Rst = np.empty((test_num, N, dim))
        #     Vst = np.empty((test_num, N, dim))
        #     Rst[0, :, :] = R
        #     Vst[0, :, :] = V
        #     for i_time in range(test_num - 1):
        #         gen_data_test = get_forward_sim_data(Rst[i_time, :, :],
        #                                              Vst[i_time, :, :],
        #                                              Rst_lead[i_time, :],
        #                                              Vst_lead[i_time, :],
        #                                              1)
        #         Rst[i_time + 1, :, :] = gen_data_test.position
        #         Vst[i_time + 1, :, :] = gen_data_test.velocity
        #
        #     # model prediction
        #     Rsp = np.empty((test_num, N, dim))
        #     Vsp = np.empty((test_num, N, dim))
        #     Rsp[0, :, :] = R
        #     Vsp[0, :, :] = V
        #     for i_time in range(test_num - 1):
        #         pred_traj = forward_model_trained_onestep(R=Rsp[i_time, :, :],
        #                                                   V=Vsp[i_time, :, :],
        #                                                   R_lead=Rst_lead[i_time, :],
        #                                                   V_lead=Vst_lead[i_time, :])
        #         Rsp[i_time + 1, :, :] = pred_traj.position
        #         Vsp[i_time + 1, :, :] = pred_traj.velocity
        #
        #     error_pos = Rsp - Rst
        #     error_vel = Vsp - Vst
        #
        #     for i_time in range(test_num):
        #         if i_time % save_skip == 0:
        #             result = {'Magnitude': magnitude}
        #             result['Time'] = test_timestamp[i_time]
        #             for i_N in range(N):
        #                 for i_dim in range(dim):
        #                     result[f'Position_Error_Node{i_N}_Dim{i_dim}'] = error_pos[i_time, i_N, i_dim]
        #                     result[f'Velocity_Error_Node{i_N}_Dim{i_dim}'] = error_vel[i_time, i_N, i_dim]
        #                     result[f'Position_Predict_Node{i_N}_Dim{i_dim}'] = Rsp[i_time, i_N, i_dim]
        #                     result[f'Velocity_Predict_Node{i_N}_Dim{i_dim}'] = Vsp[i_time, i_N, i_dim]
        #                     result[f'Position_Truth_Node{i_N}_Dim{i_dim}'] = Rst[i_time, i_N, i_dim]
        #                     result[f'Velocity_Truth_Node{i_N}_Dim{i_dim}'] = Vst[i_time, i_N, i_dim]
        #             result_MagnitudeSweep.append(result)
        #
        #     del Rst, Vst
        #     del Rsp, Vsp
        #     del Rst_lead, Vst_lead
        #     del R, V
        #
        # fields = result_MagnitudeSweep[0].keys()
        # with open(save_path + f"_sweep_magnitude.csv", mode='w', newline='') as file:
        #     writer = csv.DictWriter(file, fieldnames=fields)
        #     writer.writeheader()
        #     writer.writerows(result_MagnitudeSweep)

        print("\n>> Scale sweep\n")
        magnitude_start = 0.001
        magnitude_end = 0.4
        magnitudes = np.linspace(magnitude_start, magnitude_end, sweep_num)
        omegas = (2 * np.pi / VirtualCoupling_sim_period) * VirtualCoupling_sim_magnitude_xy / magnitudes
        result_ScaleSweep = []
        progress = -1
        for magnitude, omega in zip(magnitudes, omegas):
            progress += 1
            freq = omega / (2 * np.pi)
            if progress % (int)(sweep_num / 10) == 0:
                print(f"Progress {progress}: testing magnitude {magnitude:.4f} m, frequency {freq:.4f}, "
                      f"velocity {omega * magnitude:.4f} m/s")

            Rst_lead = np.zeros((test_num, 1, dim))
            Vst_lead = np.zeros((test_num, 1, dim))
            for i_time in range(test_num):
                Rst_lead[i_time, 0, 0] = magnitude * np.sin(omega * test_timestamp[i_time])
                Vst_lead[i_time, 0, 0] = omega * magnitude * np.cos(omega * test_timestamp[i_time])
                Rst_lead[i_time, 0, 1] = magnitude * np.sin(omega * test_timestamp[i_time])
                Vst_lead[i_time, 0, 1] = omega * magnitude * np.cos(omega * test_timestamp[i_time])

            R, V = get_sim_data_initial(N, dim, random.PRNGKey(50))

            # ground truth
            Rst = np.empty((test_num, N, dim))
            Vst = np.empty((test_num, N, dim))
            Rst[0, :, :] = R
            Vst[0, :, :] = V
            for i_time in range(test_num - 1):
                gen_data_test = get_forward_sim_data(Rst[i_time, :, :],
                                                     Vst[i_time, :, :],
                                                     Rst_lead[i_time, :],
                                                     Vst_lead[i_time, :],
                                                     1)
                Rst[i_time + 1, :, :] = gen_data_test.position
                Vst[i_time + 1, :, :] = gen_data_test.velocity

            # model prediction
            Rsp = np.empty((test_num, N, dim))
            Vsp = np.empty((test_num, N, dim))
            Rsp[0, :, :] = R
            Vsp[0, :, :] = V
            for i_time in range(test_num - 1):
                pred_traj = forward_model_trained_onestep(R=Rsp[i_time, :, :],
                                                          V=Vsp[i_time, :, :],
                                                          R_lead=Rst_lead[i_time, :],
                                                          V_lead=Vst_lead[i_time, :])
                Rsp[i_time + 1, :, :] = pred_traj.position
                Vsp[i_time + 1, :, :] = pred_traj.velocity

            error_pos = Rsp - Rst
            error_vel = Vsp - Vst

            for i_time in range(test_num):
                if i_time % save_skip == 0:
                    result = {'Magnitude': magnitude}
                    result['Frequency'] = freq
                    result['Time'] = test_timestamp[i_time]
                    for i_N in range(N):
                        for i_dim in range(dim):
                            result[f'Position_Error_Node{i_N}_Dim{i_dim}'] = error_pos[i_time, i_N, i_dim]
                            result[f'Velocity_Error_Node{i_N}_Dim{i_dim}'] = error_vel[i_time, i_N, i_dim]
                            result[f'Position_Predict_Node{i_N}_Dim{i_dim}'] = Rsp[i_time, i_N, i_dim]
                            result[f'Velocity_Predict_Node{i_N}_Dim{i_dim}'] = Vsp[i_time, i_N, i_dim]
                            result[f'Position_Truth_Node{i_N}_Dim{i_dim}'] = Rst[i_time, i_N, i_dim]
                            result[f'Velocity_Truth_Node{i_N}_Dim{i_dim}'] = Vst[i_time, i_N, i_dim]
                    result_ScaleSweep.append(result)

            if magnitude > 0.03:
                plot_trajectory(Rsp, test_num, test_timestamp, "Test",
                                data_compare=Rst, data_virtual=Rst_lead)
                print(freq)
                input("stop")

            del Rst, Vst
            del Rsp, Vsp
            del Rst_lead, Vst_lead
            del R, V

        fields = result_ScaleSweep[0].keys()
        with open(save_path + f"_sweep_scale.csv", mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(result_ScaleSweep)


fire.Fire(Main)
