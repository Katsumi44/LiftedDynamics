import json
import os
import socket
import sys
import time
from datetime import datetime
from functools import wraps

import fire
import matplotlib
import pandas as pd
from scipy.signal import butter, filtfilt, freqz
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import UnivariateSpline

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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
from scipy.interpolate import CubicSpline

jax.config.update("jax_enable_x64", True)

path_data_root = "C:\YutongZhang\LGNN\dataset"
path_data_train = path_data_root + "\data_20240814_115559.csv"
path_data_test = path_data_root + "\data_20240814_115527.csv"
matplotlib.rcParams['figure.max_open_warning'] = 100

use_RealData = True
use_SimData = False
use_SimNoise = False

use_drag = True

const_numerical = 2e-7  # note that linalg also has numerical problem (because of float precision)
const_gravity_acc = 9.81

use_object = 1

if use_object == 1:
    N = 5  # number of points
    masses = np.array([40, 40, 40, 40, 40])
    length = np.array([0.05, 0.05, 0.05, 0.05])

species = jnp.zeros(N, dtype=int)  # node types
object_mass = np.sum(masses)
object_length = np.sum(length)

use_stretching = False
if use_object == 1:
    object_stiffness_stretching_scale = 5000 * 1e3
    object_stiffness_stretching_sim = np.array([1.0, 1.0, 1.0, 1.0, 1.0]) * object_stiffness_stretching_scale

use_bending = True
if use_object == 1:
    object_stiffness_bending_scale = 0.1 * 1e3
    object_stiffness_bending_sim = np.array([1.0, 1.0, 1.0, 1.0, 1.0]) * object_stiffness_bending_scale

use_twisting = False
object_stiffness_twisting_scale = 0.001 * 1e3
object_stiffness_twisting_sim = object_stiffness_twisting_scale * 1.0

use_tension = False
object_stiffness_tension_scale = 1000 * 1e3
object_stiffness_tension_sim = object_stiffness_tension_scale * 5.0
object_damping_tension_scale = 10 * 1e3 * 0.0
object_damping_tension_sim = object_damping_tension_scale * 5.0

use_damping = False
if use_object == 1:
    object_damping_scale = 0.01 * 1e3
    object_damping_sim = np.array([5.0] * N) * object_damping_scale

use_gravity_energy = True
use_gravity_force = False  # proven to be the same

use_VirtualCoupling = True
if use_object == 1:
    VirtualCoupling_stiffness = 500 * 1e3
    VirtualCoupling_damping = 5 * 1e3

VirtualCoupling_sim_magnitude_xy = 0.10  # m
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
process_var = 0.001  # Process noise variance
measurement_var = 1  # Measurement noise variance
estimated_var = 0.01  # Initial estimate of error covariance

execute_StaticTest = False
execute_learn = True
message_pass_num = 1  # 1
execute_render = False
execute_test = False
model_trained_path = r"C:\YutongZhang\LGNN\trained\simulation\08-12-2024_15-48-43_None_Object1\lgnn_trained_model_True_0.dil"

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


def Main(epochs=500, seed=42, rname=True, saveat=10, error_fn="L2error",
         dt=1.0e-4, stride=100, trainm=0, grid=False, lr=0.001,
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

    # have "N=" to make this work
    # print("Configs: ")
    # pprint(N, epochs, seed, rname,
    #        dt, stride, lr, use_drag, batch_size,
    #        namespace=locals())

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

    @jit
    def get_energy_spring(x, stiffness, initial_length):
        return 0.5 * stiffness * (x - initial_length) ** 2

    def apply_KalmanFilter(timestamp, data_num, data_position, point_num):
        output_position = np.zeros((data_num, point_num, 3))
        output_velocity = np.zeros((data_num, point_num, 3))
        output_acceleration = np.zeros((data_num, point_num, 3))

        t = timestamp.reshape(-1)

        for i_N in range(point_num):

            measured_position = np.zeros((3, data_num))
            for i_time in range(data_num):
                for i_dim in range(3):
                    measured_position[i_dim, i_time] = data_position[i_time, i_N, i_dim]

            # Initialize the Kalman filter
            kf = KalmanFilter3D(process_var, measurement_var, estimated_var)

            estimated_position = np.zeros_like(measured_position)
            estimated_velocity = np.zeros_like(measured_position)
            estimated_acceleration = np.zeros_like(measured_position)

            # Process data continuously
            for i in range(len(t)):
                if i == 0:
                    # Initial state
                    kf.x[:3] = measured_position[:, i]
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
                edge_length_compare[i_time, :] = np.sqrt(
                    np.square(R - np.vstack([np.zeros_like(R[0]), R[:-1]])).sum(axis=1))

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
            ax.plot(timestamp, data[:, index, i])
            ax.set_title(label_title_ax[i])
        plt.suptitle(f"{data_meaning} (Example {index}, {data_type})")
        plt.tight_layout()
        plt.savefig(_filename(f"dataset_example{index}_{data_meaning}_{data_type}.png"))

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

    # ******************** Dataset preparation ********************

    print("\n******************** Dataset preparation ********************\n")

    if use_RealData:
        use_force_direct = True
        use_force_virtual = False

        print("Real data used\n")
        data_train = pd.read_csv(path_data_train)
        data_test = pd.read_csv(path_data_test)
        col_pos = list(range(3 * (5 + 1)))  # trajectory cols (5 markers + 1 rigid body)
        temp = [x + 3 * 4 for x in col_pos]  # 4 markers
        col_pos = temp
        col_time = 3 * (5 + 4 + 2) + 6 + 2 - 1  # 5 markers + 2 rigid body + 6 force&torque + freq + time
        train_timestamp = data_train.iloc[:, col_time].values
        test_timestamp = data_test.iloc[:, col_time].values
        col_force = [3 * (5 + 4 + 2) + 1 - 1, 3 * (5 + 4 + 2) + 2 - 1, 3 * (5 + 4 + 2) + 3 - 1]  # FX, FY, FZ
        print(f"col_pos = {col_pos}, col_force = {col_force}\n")

        # order from top to bottom
        if N == 6:
            order_list = [5, 0, 3, 2, 4, 1]
        elif N == 5:
            order_list = [0, 3, 2, 4, 1]

        Rs_raw = data_train.iloc[:, col_pos].values
        Rst_raw = data_test.iloc[:, col_pos].values
        force_lead_train_raw = data_train.iloc[:, col_force].values
        force_lead_test_raw = data_test.iloc[:, col_force].values
        print("Rs_raw shape = {}, Rst_raw shape = {}\n".format(Rs_raw.shape, Rst_raw.shape))
        print("train_timestamp shape = {}, test_timestamp shape = {}\n".format(train_timestamp.shape,
                                                                               test_timestamp.shape))

        # reformat

        train_num = len(Rs_raw)
        Rs = np.empty((train_num, N, dim))
        Rs_virtual = np.empty((train_num, 1, dim))
        force_lead_train = np.empty((train_num, 1, dim))
        for i_N in range(N):
            for i_dim in range(dim):
                i_col = 3 * order_list[i_N] + i_dim
                Rs[:, i_N, i_dim] = Rs_raw[:, i_col]
        for i_dim in range(dim):
            i_col = 3 * 5 + i_dim
            Rs_virtual[:, 0, i_dim] = Rs_raw[:, i_col]
        for i_dim in range(dim):
            force_lead_train[:, 0, i_dim] = force_lead_train_raw[:, i_dim]

        test_num = len(Rst_raw)
        Rst = np.empty((test_num, N, dim))
        Rst_virtual = np.empty((test_num, 1, dim))
        force_lead_test = np.empty((test_num, 1, dim))
        for i_N in range(N):
            for i_dim in range(dim):
                i_col = 3 * order_list[i_N] + i_dim
                Rst[:, i_N, i_dim] = Rst_raw[:, i_col]
        for i_dim in range(dim):
            i_col = 3 * 5 + i_dim
            Rst_virtual[:, 0, i_dim] = Rst_raw[:, i_col]
        for i_dim in range(dim):
            force_lead_test[:, 0, i_dim] = force_lead_test_raw[:, i_dim]

        # Force Sensor to World Coordinate

        force_lead_train_temp = np.zeros_like(force_lead_train)
        force_lead_train_temp[:, 0, 0] = force_lead_train[:, 0, 1]
        force_lead_train_temp[:, 0, 1] = force_lead_train[:, 0, 0]
        force_lead_train_temp[:, 0, 2] = force_lead_train[:, 0, 2]
        force_lead_train = force_lead_train_temp

        force_lead_test_temp = np.zeros_like(force_lead_test)
        force_lead_test_temp[:, 0, 0] = force_lead_test[:, 0, 1]
        force_lead_test_temp[:, 0, 1] = force_lead_test[:, 0, 0]
        force_lead_test_temp[:, 0, 2] = force_lead_test[:, 0, 2]
        force_lead_test = force_lead_test_temp

        # translation

        print(">> Coordinate adjustment: translation\n")
        Rs_temp = np.empty((train_num, N, dim))
        Rs_virtual_temp = np.empty((train_num, 1, dim))
        Rst_temp = np.empty((test_num, N, dim))
        Rst_virtual_temp = np.empty((test_num, 1, dim))
        for i_N in range(N):
            for i_dim in range(dim):
                Rs_temp[:, i_N, i_dim] = Rs[:, i_N, i_dim] - Rs[0, 0, i_dim]
                Rs_virtual_temp[:, 0, i_dim] = Rs_virtual[:, 0, i_dim] - Rs_virtual[0, 0, i_dim]
                Rst_temp[:, i_N, i_dim] = Rst[:, i_N, i_dim] - Rst[0, 0, i_dim]
                Rst_virtual_temp[:, 0, i_dim] = Rst_virtual[:, 0, i_dim] - Rst_virtual[0, 0, i_dim]
        Rs = Rs_temp
        Rst = Rst_temp

        if use_force_virtual:
            train_num -= 1
            test_num -= 1
            Rs = Rs_temp[1:, :, :]
            Rst = Rst_temp[1:, :, :]
            Rs_virtual = Rs_virtual_temp[:-1, :, :]
            Rst_virtual = Rst_virtual_temp[:-1, :, :]
            train_timestamp = train_timestamp[1:]
            test_timestamp = test_timestamp[1:]

        if use_force_direct:
            plot_data_example(force_lead_train, 'Train', 'ForceRaw', train_timestamp)
            plot_data_example(force_lead_test, 'Test', 'ForceRaw', test_timestamp)

        # rotation: make initial direction along z direction

        print(">> Coordinate adjustment: rotation\n")

        def get_fit_direction(fit_points):
            # Compute the centroid of the points
            fit_centroid = np.mean(fit_points, axis=0)

            # Center the points by subtracting the centroid
            fit_centered_points = fit_points - fit_centroid

            # Compute the covariance matrix of the centered points
            fit_cov_matrix = np.cov(fit_centered_points.T)

            # Perform Singular Value Decomposition (SVD) on the covariance matrix
            fit_U, fit_S, fit_Vt = np.linalg.svd(fit_cov_matrix)

            # The direction vector of the line is given by the first column of V (or the first row of Vt)
            fit_direction = fit_Vt.T[:, 0]
            return fit_direction

        def get_rotation_matrix(v, t):
            v = np.array(v)
            t = np.array(t)

            v_norm = v / np.linalg.norm(v)
            t_norm = t / np.linalg.norm(t)

            axis = np.cross(v_norm, t_norm)
            angle = np.arccos(np.clip(np.dot(v_norm, t_norm), -1.0, 1.0))

            if np.linalg.norm(axis) == 0:  # Vectors are parallel
                return np.eye(3) if np.allclose(v_norm, t_norm) else None

            axis = axis / np.linalg.norm(axis)
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])

            I = np.eye(3)
            R = I + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

            return R

        target_direction = np.array([0, 0, 1])
        target_direction = target_direction / np.linalg.norm(target_direction)

        # training dataset

        fit_points = np.zeros((N, dim))
        for i_N in range(N):
            for i_dim in range(dim):
                fit_points[i_N, i_dim] = Rs[0, i_N, i_dim]
        fit_direction = get_fit_direction(fit_points)
        # fit_data = np.empty((N, dim))
        # for i_N in range(N):
        #     fit_data[i_N, :] = Rs[0, i_N, :]  # use points along the object to fit the line
        # pca = PCA(n_components=1)
        # pca.fit(fit_data)
        # direction = pca.components_[0]
        fit_direction = fit_direction / np.linalg.norm(fit_direction)
        rot_matrix = get_rotation_matrix(fit_direction, target_direction)
        Rs_temp = np.empty((train_num, N, dim))
        for i_time in range(train_num):
            for i_N in range(N):
                Rs_temp[i_time, i_N, :] = np.dot(rot_matrix, Rs[i_time, i_N, :].reshape(dim))
        Rs = Rs_temp

        Rs_virtual_temp = np.empty((train_num, N, dim))
        for i_time in range(train_num):
            Rs_virtual_temp[i_time, 0, :] = np.dot(rot_matrix, Rs_virtual[i_time, 0, :].reshape(dim))
        Rs_virtual = Rs_virtual_temp

        # force
        # fit_direction = force_lead_train[0, 0, :].reshape((dim))
        # print(f"Force fit direction: {fit_direction}")
        # fit_direction = fit_direction / np.linalg.norm(fit_direction)
        # rot_matrix = get_rotation_matrix(fit_direction, target_direction)
        # force_lead_train_temp = np.empty((train_num, 1, dim))
        # for i_time in range(train_num):
        #     force_lead_train_temp[i_time, 0, :] = np.dot(rot_matrix, force_lead_train[i_time, 0, :].reshape(dim))
        # force_lead_train = force_lead_train_temp

        # test dataset

        fit_points = np.zeros((N, dim))
        for i_N in range(N):
            for i_dim in range(dim):
                fit_points[i_N, i_dim] = Rst[0, i_N, i_dim]
        fit_direction = get_fit_direction(fit_points)
        # fit_data = np.empty((N, dim))
        # for i_N in range(N):
        #     fit_data[i_N, :] = Rst[0, i_N, :]  # use points along the object to fit the line
        # pca = PCA(n_components=1)
        # pca.fit(fit_data)
        # direction = pca.components_[0]
        fit_direction = fit_direction / np.linalg.norm(fit_direction)
        rot_matrix = get_rotation_matrix(fit_direction, target_direction)
        Rst_temp = np.empty((test_num, N, dim))
        for i_time in range(test_num):
            for i_N in range(N):
                Rst_temp[i_time, i_N, :] = np.dot(rot_matrix, Rst[i_time, i_N, :].reshape(dim))
        Rst = Rst_temp

        Rst_virtual_temp = np.empty((test_num, N, dim))
        for i_time in range(test_num):
            Rst_virtual_temp[i_time, 0, :] = np.dot(rot_matrix, Rst_virtual[i_time, 0, :].reshape(dim))
        Rst_virtual = Rst_virtual_temp

        # force
        # fit_direction = force_lead_test[0, 0, :].reshape((dim))
        # print(f"Force fit direction: {fit_direction}")
        # fit_direction = fit_direction / np.linalg.norm(fit_direction)
        # rot_matrix = get_rotation_matrix(fit_direction, target_direction)
        # force_lead_test_temp = np.empty((test_num, 1, dim))
        # for i_time in range(test_num):
        #     force_lead_test_temp[i_time, 0, :] = np.dot(rot_matrix, force_lead_test[i_time, 0, :].reshape(dim))
        # force_lead_test = force_lead_test_temp
        #
        # plot_data_example(force_lead_train, 'Train', 'ForceRot', train_timestamp)
        # plot_data_example(force_lead_test, 'Test', 'ForceRot', test_timestamp)

        # SI units

        Rs /= 1000.0
        Rs_virtual /= 1000.0
        Rst /= 1000.0
        Rst_virtual /= 1000.0

        for i_N in range(N):
            plot_data_example(Rs, 'Train', 'PositionRaw', train_timestamp, i_N)
            plot_data_example(Rst, 'Test', 'PositionRaw', test_timestamp, i_N)

        # debug
        #
        # Rs_spline = np.zeros_like(Rs)
        # for i_N in range(N):
        #     for i_dim in range(dim):
        #         spline = UnivariateSpline(train_timestamp, Rs[:, i_N, i_dim], s=1)
        #         Rs_spline[:, i_N, i_dim] = spline(train_timestamp)
        # Rs = Rs_spline
        # for i_N in range(N):
        #     plot_data_example(Rs, "Train", f"PositionSpline", train_timestamp, index=i_N)
        #
        # Fs_pos = np.zeros((train_num, N, dim))  # Rs smooth
        # i_N = 0
        # save = []
        # for i_time in range(1, train_num-1):
        #     pos_p1 = Rs[i_time + 1, i_N, :]
        #     pos_n1 = Rs[i_time - 1, i_N, :]
        #     pos = Rs[i_time, i_N, :]
        #     delta_t = (train_timestamp[i_time + 1] - train_timestamp[i_time - 1]) / 2
        #     delta_t = 0.01
        #     # print(delta_t, pos_p1, pos_n1)
        #     # delta_t = 0.01
        #     # print(delta_t)
        #     # print(type(delta_t))
        #     # input("stop")
        #     Fs_pos[i_time, 0, :] = (100 * pos_p1 + 100*pos_n1 -200*pos) / (0.01)
        #     print(pos)
        #     save.append(pos)
        #
        #     # delta_t = (train_timestamp[i_time + 1] - train_timestamp[i_time - 1]) / 2
        #     # Fs_pos[i_time, i_N, :] = (pos_p1 + pos_n1 - 2 * pos) / (delta_t ** 2)
        #
        # fig, axis = plt.subplots(3, 1)
        # label_title_ax = ["X", "Y", "Z"]
        # for i, ax in enumerate(axis):
        #     # ax.plot(save)
        #     ax.plot(Fs_pos[:,0,0])
        #     ax.set_title(label_title_ax[i])
        # plt.tight_layout()
        # plt.savefig(_filename(f"debug.png"))
        # plt.show()
        #
        # input("Press Enter to continue...")
        #
        # for i_N in range(N):
        #     plot_data_example(Fs_pos, "Train", f"Debug", train_timestamp, index=i_N)

        # position LPF

        LPF_pos_cutoff = 10.0
        LPF_pos_fs = 100.0
        LPF_pos_order = 5

        plot_filter_frequency_response(LPF_pos_cutoff, LPF_pos_fs, LPF_pos_order, "LPF", "Position")

        Rs_LPF = np.zeros_like(Rs)
        Rst_LPF = np.zeros_like(Rst)
        for i_N in range(N):
            for i_dim in range(dim):
                Rs_LPF[:, i_N, i_dim] = filter_lowpass(Rs[:, i_N, i_dim],
                                                       LPF_pos_cutoff,
                                                       LPF_pos_fs,
                                                       LPF_pos_order)
                Rst_LPF[:, i_N, i_dim] = filter_lowpass(Rst[:, i_N, i_dim],
                                                        LPF_pos_cutoff,
                                                        LPF_pos_fs,
                                                        LPF_pos_order)

        plot_frequency_spectrum(Rs[:, 0, :], Rs_LPF[:, 0, :], LPF_pos_fs, "Train", "PositionLPF")
        plot_frequency_spectrum(Rst[:, 0, :], Rst_LPF[:, 0, :], LPF_pos_fs, "Test", "PositionLPF")
        Rs = Rs_LPF
        Rst = Rst_LPF

        for i_N in range(N):
            plot_data_example(Rs, 'Train', 'PositionLPF', train_timestamp, i_N)
            plot_data_example(Rst, 'Test', 'PositionLPF', test_timestamp, i_N)

        # force LPF

        LPF_force_cutoff = 1.0
        LPF_force_fs = 100.0
        LPF_force_order = 5

        plot_filter_frequency_response(LPF_force_cutoff, LPF_force_fs, LPF_force_order, "LPF", "Force")

        force_lead_train_LPF = np.zeros_like(force_lead_train)
        force_lead_test_LPF = np.zeros_like(force_lead_test)
        for i_dim in range(dim):
            force_lead_train_LPF[:, 0, i_dim] = filter_lowpass(force_lead_train[:, 0, i_dim],
                                                               LPF_force_cutoff,
                                                               LPF_force_fs,
                                                               LPF_force_order)
            force_lead_test_LPF[:, 0, i_dim] = filter_lowpass(force_lead_test[:, 0, i_dim],
                                                              LPF_force_cutoff,
                                                              LPF_force_fs,
                                                              LPF_force_order)

        plot_frequency_spectrum(force_lead_train[:, 0, :], force_lead_train_LPF[:, 0, :], LPF_force_fs, "Train",
                                "ForceLPF")
        plot_frequency_spectrum(force_lead_test[:, 0, :], force_lead_test_LPF[:, 0, :], LPF_force_fs, "Test",
                                "ForceLPF")
        force_lead_train = force_lead_train_LPF
        force_lead_test = force_lead_test_LPF

        plot_data_example(force_lead_train, 'Train', 'ForceLPF', train_timestamp, 0)
        plot_data_example(force_lead_test, 'Test', 'ForceLPF', test_timestamp, 0)

        print(">> Apply Kalman filter\n")

        Rs_Kalman, Vs_Kalman, Fs_Kalman = apply_KalmanFilter(train_timestamp, train_num, Rs, N)
        Rst_Kalman, Vst_Kalman, Fst_Kalman = apply_KalmanFilter(test_timestamp, test_num, Rst, N)

        if use_force_virtual:
            Rs_virtual, Vs_virtual, _ = apply_KalmanFilter(train_timestamp, train_num, Rs_virtual, 1)
            Rst_virtual, Vst_virtual, _ = apply_KalmanFilter(test_timestamp, test_num, Rst_virtual, 1)

        Rs = Rs_Kalman
        Rst = Rst_Kalman
        Vs = Vs_Kalman
        Vst = Vst_Kalman

        for i_N in range(N):
            plot_data_example(Rs, 'Train', 'PositionKalman', train_timestamp, i_N)
            plot_data_example(Rst, 'Test', 'PositionKalman', test_timestamp, i_N)
            plot_data_example(Vs, 'Train', 'VelocityKalman', train_timestamp, i_N)
            plot_data_example(Vst, 'Test', 'VelocityKalman', test_timestamp, i_N)

            if use_force_virtual:
                plot_data_example(Rs_virtual, 'Train', 'PositionVirtual', train_timestamp, i_N)
                plot_data_example(Rst_virtual, 'Test', 'PositionVirtual', test_timestamp, i_N)

        # smooth the Kalman filter estimates

        def moving_average(data, window_size):
            return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

        mov_windowsize = 10

        Rs_mov = np.zeros((train_num - mov_windowsize + 1, N, dim))
        Vs_mov = np.zeros((train_num - mov_windowsize + 1, N, dim))
        force_lead_train_mov = np.zeros((train_num - mov_windowsize + 1, 1, dim))

        Rst_mov = np.zeros((test_num - mov_windowsize + 1, N, dim))
        Vst_mov = np.zeros((test_num - mov_windowsize + 1, N, dim))
        force_lead_test_mov = np.zeros((test_num - mov_windowsize + 1, 1, dim))

        train_num -= mov_windowsize + 1
        test_num -= mov_windowsize + 1

        train_timestamp = moving_average(train_timestamp, window_size=mov_windowsize)
        test_timestamp = moving_average(test_timestamp, window_size=mov_windowsize)
        for i_dim in range(dim):
            force_lead_train_mov[:, 0, i_dim] = moving_average(force_lead_train[:, 0, i_dim],
                                                               window_size=mov_windowsize)
            force_lead_test_mov[:, 0, i_dim] = moving_average(force_lead_test[:, 0, i_dim],
                                                              window_size=mov_windowsize)
        for i_N in range(N):
            for i_dim in range(dim):
                Rs_mov[:, i_N, i_dim] = moving_average(Rs[:, i_N, i_dim], window_size=mov_windowsize)
                Rst_mov[:, i_N, i_dim] = moving_average(Rst[:, i_N, i_dim], window_size=mov_windowsize)
                Vs_mov[:, i_N, i_dim] = moving_average(Vs[:, i_N, i_dim], window_size=mov_windowsize)
                Vst_mov[:, i_N, i_dim] = moving_average(Vst[:, i_N, i_dim], window_size=mov_windowsize)

        Rs = Rs_mov
        Rst = Rst_mov
        Vs = Vs_mov
        Vst = Vst_mov
        force_lead_train = force_lead_train_mov
        force_lead_test = force_lead_test_mov

        plot_data_example(force_lead_train, 'Train', 'ForceMOV', train_timestamp, 0)
        plot_data_example(force_lead_test, 'Test', 'ForceMOV', test_timestamp, 0)
        for i_N in range(N):
            plot_data_example(Rs, 'Train', 'PositionMOV', train_timestamp, i_N)
            plot_data_example(Rst, 'Test', 'PositionMOV', test_timestamp, i_N)
            plot_data_example(Vs, 'Train', 'VelocityMOV', train_timestamp, i_N)
            plot_data_example(Vst, 'Test', 'VelocityMOV', test_timestamp, i_N)

        # compute ground truth acceleration

        def get_acceleration_velocity(data, timestamp, data_num):
            result = np.zeros_like(data)
            for i_time in range(1, data_num):
                for i_N in range(N):
                    delta_vel = data[i_time, i_N, :] - data[i_time - 1, i_N, :]
                    delta_t = timestamp[i_time] - timestamp[i_time - 1]
                    result[i_time, i_N, :] = delta_vel / delta_t
            return result

        def get_acceleration_position(data, timestamp, data_num):
            result = np.zeros_like(data)
            for i_time in range(1, data_num - 1):
                for i_N in range(N):
                    pos_p1 = data[i_time + 1, i_N, :]
                    pos_n1 = data[i_time - 1, i_N, :]
                    pos = data[i_time, i_N, :]
                    delta_t = (timestamp[i_time + 1] - timestamp[i_time - 1]) / 2
                    result[i_time, i_N, :] = (pos_p1 + pos_n1 - 2 * pos) / (delta_t ** 2)
            return result

        Fs = get_acceleration_position(Rs, train_timestamp, train_num)
        Fst = get_acceleration_position(Rst, test_timestamp, test_num)
        for i_N in range(N):
            plot_data_example(Fs, 'Train', 'Acceleration', train_timestamp, i_N)
            plot_data_example(Fst, 'Test', 'Acceleration', test_timestamp, i_N)

        # print(">> Apply low pass filter\n")
        #
        # filter_freq_cutoff = 10.0
        # filter_freq_sample = 100.0
        # filter_order = 5
        # plot_filter_frequency_response(filter_freq_cutoff,
        #                                filter_freq_sample,
        #                                filter_order,
        #                                "Test",
        #                                "AccelerationPositionLPF")
        #
        # # spline
        #
        # # Vel_spline = np.zeros_like(Rs)
        # # for i_N in range(N):
        # #     for i_dim in range(dim):
        # #         spline = UnivariateSpline(train_timestamp, Vs[:, i_N, i_dim], k=5)
        # #         Vel_spline[:, i_N, i_dim] = spline(train_timestamp)
        #
        # # for i_N in range(N):
        # #     plot_data_example(Vel_spline, "Train", f"VelocitySpline", train_timestamp, index=i_N)
        #
        # Fs_vel = get_acceleration_velocity(Vs, train_timestamp, train_num)
        # Fst_vel = get_acceleration_velocity(Vst, test_timestamp, test_num)
        # for i_N in range(N):
        #     plot_data_example(Fs_vel, "Train", f"AccelerationVelocity", train_timestamp, index=i_N)
        #     plot_data_example(Fst_vel, "Test", f"AccelerationVelocity", test_timestamp, index=i_N)
        #
        # Fs_pos = get_acceleration_position(Rs, train_timestamp, train_num)
        # Fst_pos = get_acceleration_position(Rst, test_timestamp, test_num)
        # for i_N in range(N):
        #     plot_data_example(Fs_pos, "Train", f"AccelerationPosition", train_timestamp, index=i_N)
        #     plot_data_example(Fst_pos, "Test", f"AccelerationPosition", test_timestamp, index=i_N)
        #
        # Fs_pos_LPF = np.zeros_like(Rs)
        # Fst_pos_LPF = np.zeros_like(Rst)
        # for i_N in range(N):
        #     for i_dim in range(dim):
        #         Fs_pos_LPF[:, i_N, i_dim] = filter_lowpass(Fs_pos[:, i_N, i_dim].reshape(-1),
        #                                                    filter_freq_cutoff,
        #                                                    filter_freq_sample,
        #                                                    filter_order)
        #         Fst_pos_LPF[:, i_N, i_dim] = filter_lowpass(Fst_pos[:, i_N, i_dim].reshape(-1),
        #                                                     filter_freq_cutoff,
        #                                                     filter_freq_sample,
        #                                                     filter_order)
        # plot_frequency_spectrum(Fst_pos[:, 1, :],
        #                         Fst_pos_LPF[:, 1, :],
        #                         filter_freq_sample,
        #                         "Test",
        #                         "AccelerationPosition")
        # for i_N in range(N):
        #     plot_data_example(Fst_pos_LPF, "Test", f"AccelerationPositionLPF", test_timestamp, index=i_N)
        #
        # if use_force_direct:
        #     plot_data_example(force_lead_test, 'Test', 'ForceData', test_timestamp)
        #
        #     force_lead_train_LPF = np.zeros_like(force_lead_train)
        #     force_lead_test_LPF = np.zeros_like(force_lead_test)
        #     for i_dim in range(dim):
        #         force_lead_train_LPF[:, 0, i_dim] = filter_lowpass(force_lead_train[:, 0, i_dim].reshape(-1),
        #                                                            0.5,
        #                                                            filter_freq_sample,
        #                                                            filter_order)
        #         force_lead_test_LPF[:, 0, i_dim] = filter_lowpass(force_lead_test[:, 0, i_dim].reshape(-1),
        #                                                           0.5,
        #                                                           filter_freq_sample,
        #                                                           filter_order)
        #
        #     plot_data_example(force_lead_test_LPF, 'Test', 'ForceLPF', test_timestamp)
        #     plot_frequency_spectrum(force_lead_test[:, 0, :],
        #                             force_lead_test_LPF[:, 0, :],
        #                             filter_freq_sample,
        #                             "Test",
        #                             "ForceLPF")

        # scale to DIY scale (g, and 1e3 N)

        force_lead_train *= -1000.0 * 0
        force_lead_test *= -1000.0 * 0
        force_lead_train[:, 0, 2] += object_mass * const_gravity_acc
        force_lead_test[:, 0, 2] += object_mass * const_gravity_acc

        plot_data_example(force_lead_train, 'Train', 'ForceFinal', train_timestamp)
        plot_data_example(force_lead_test, 'Test', 'ForceFinal', test_timestamp)

        # prepare for learning input

        print(">> Plot training trajectory\n")
        plot_trajectory(data_traj=Rs, data_num=train_num, data_time=train_timestamp, data_type='Train', show_skip=5,
                        show_fps=20, data_force=force_lead_train / 1000)

        print(">> Plot test trajectory\n")
        plot_trajectory(data_traj=Rst, data_num=test_num, data_time=test_timestamp, data_type='Test', show_skip=5,
                        show_fps=20, data_force=force_lead_test / 1000)

        Rs_lead = force_lead_train
        Rst_lead = force_lead_test
        Vs_lead = np.zeros_like(Rs_lead)
        Vst_lead = np.zeros_like(Rst_lead)

        R, V = Rs[0], Vs[0]
        length = jnp.sqrt(jnp.square(R[1:] - R[:-1]).sum(axis=1))
        print(f"species = {species}\n")
        print(f"masses = {masses}\n")
        print(f"length = {length}\n")

        # plot_data_example(Rs, 'Train', 'PositionFinal', train_timestamp)
        # plot_data_example(Rst, 'Test', 'PositionFinal', test_timestamp)
        # plot_data_example(Vs, 'Train', 'VelocityFinal', train_timestamp)
        # plot_data_example(Vst, 'Test', 'VelocityFinal', test_timestamp)
        # plot_data_example(Fs, 'Train', 'AccelerationFinal', train_timestamp)
        # plot_data_example(Fst, 'Test', 'AccelerationFinal', test_timestamp)
        #
        # if use_force_virtual:
        #
        #     Fs_feedback = np.empty((train_num, dim))
        #     for i in range(train_num):
        #         Fs_feedback[i, :] = get_force_feedback(x=Rs[i, :, :], v=Vs[i, :, :],
        #                                                x_lead=Rs_virtual[i, :, :], v_lead=Vs_virtual[i, :, :])
        #
        #     Fst_feedback = np.empty((test_num, dim))
        #     for i in range(train_num):
        #         Fst_feedback[i, :] = get_force_feedback(x=Rst[i, :, :], v=Vst[i, :, :],
        #                                                 x_lead=Rst_virtual[i, :, :], v_lead=Vst_virtual[i, :, :])
        #
        #     print(">> Plot training trajectory\n")
        #     plot_trajectory(data_traj=Rs, data_num=train_num, data_time=train_timestamp, data_type='Train', show_skip=5,
        #                     show_fps=20, data_virtual=Rs_virtual, data_force=Fs_feedback)
        #
        #     print(">> Plot test trajectory\n")
        #     plot_trajectory(data_traj=Rst, data_num=test_num, data_time=test_timestamp, data_type='Test', show_skip=5,
        #                     show_fps=20, data_virtual=Rst_virtual, data_force=Fst_feedback)
        #
        #     Rs_lead = Rs_virtual
        #     Rst_lead = Rst_virtual
        #     Vs_lead = Vs_virtual
        #     Vst_lead = Vst_virtual
        #     Fs = Fs_pos_LPF
        #     Fst = Fst_pos_LPF

        input("Press Enter to continue...\n")

    # ******************** System configuration ********************

    print("\n******************** System configuration ********************\n")

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

            # def nn_drag_model_pen(x_and_v, params):
            #     return models.forward_pass(params, x_and_v, activation_fn=models.SquarePlus)
            #
            # x_and_v = jnp.hstack((x, v))
            # result_general = vmap(nn_drag_model_pen, in_axes=(0, None))(x_and_v, params["drag_general"])
            # result += result_general

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
                    if use_force_direct:
                        # force_VirtualCoupling = get_force_VirtualCoupling(x=x, v=v, x_lead=x_lead, v_lead=v_lead)
                        # result_VirtualCoupling = result_VirtualCoupling.at[0, :].set(force_VirtualCoupling)
                        result_VirtualCoupling = result_VirtualCoupling.at[0, :].set(x_lead.reshape((dim)))
                        result += result_VirtualCoupling
                    else:
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

    mpass = mpass_pen
    senders = senders_pen
    receivers = receivers_pen

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

        # if use_drag:
        #     print(">> Plot drag prediction results\n")
        #
        #     drag_truth_train = np.zeros((train_num, N, dim))
        #     drag_predict_train = np.zeros((train_num, N, dim))
        #     for i_time in range(train_num):
        #         R = np.empty((N, dim))
        #         V = np.empty((N, dim))
        #         R = Rs[i_time, :, :]
        #         V = Vs[i_time, :, :]
        #         if use_VirtualCoupling:
        #             R_lead = np.empty((N, dim))
        #             V_lead = np.empty((N, dim))
        #             R_lead = Rs_lead[i_time, :, :]
        #             V_lead = Vs_lead[i_time, :, :]
        #         drag_truth_train[i_time, 0, :] = R_lead.reshape((dim))
        #         drag_predict_train[i_time, :, :] = drag_model(R, V, R_lead, V_lead, params).reshape(N, dim)
        #     for i_N in range(N):
        #         plot_data_example(drag_truth_train, "TrainTruth", "Drag", train_timestamp, i_N)
        #         plot_data_example(drag_predict_train, "TrainPredict", "Drag", train_timestamp, i_N)
        #     print("Training results saved\n")
        #
        #     drag_truth_test = np.zeros((test_num, N, dim))
        #     drag_predict_test = np.zeros((test_num, N, dim))
        #     for i_time in range(test_num):
        #         R = np.empty((N, dim))
        #         V = np.empty((N, dim))
        #         R = Rst[i_time, :, :]
        #         V = Vst[i_time, :, :]
        #         if use_VirtualCoupling:
        #             R_lead = np.empty((N, dim))
        #             V_lead = np.empty((N, dim))
        #             R_lead = Rst_lead[i_time, :, :]
        #             V_lead = Vst_lead[i_time, :, :]
        #         drag_truth_test[i_time, 0, :] = R_lead.reshape((dim))
        #         drag_predict_test[i_time, :, :] = drag_model(R, V, R_lead, V_lead, params).reshape(N, dim)
        #     for i_N in range(N):
        #         plot_data_example(drag_truth_test, "TestTruth", "Drag", test_timestamp, i_N)
        #         plot_data_example(drag_predict_test, "TestPredict", "Drag", test_timestamp, i_N)
        #     print("Test results saved\n")

        print(">> Plot trajectory predictions\n")

        print("1) Training dataset\n")
        start_time = time.time()
        pred_pos_train = np.empty((train_num, N, dim))
        pred_vel_train = np.empty((train_num, N, dim))
        pred_pos_train[0, :, :] = Rs[0, :, :]
        pred_vel_train[0, :, :] = Vs[0, :, :]
        for i_time in range(train_num - 1):
            index = i_time
            pred_traj_temp = forward_model_trained_onestep(R=pred_pos_train[index, :, :],
                                                           V=pred_vel_train[index, :, :],
                                                           R_lead=Rs_lead[index, :],
                                                           V_lead=Vs_lead[index, :])
            pred_pos_train[index + 1, :, :] = pred_traj_temp.position
            pred_vel_train[index + 1, :, :] = pred_traj_temp.velocity
        end_time = time.time()
        execution_time = end_time - start_time
        execution_time_aver = execution_time / train_num
        print(f"Prediction speed (average frequency): {(1.0 / execution_time_aver):.2f} Hz\n")
        plot_trajectory(data_traj=pred_pos_train,
                        data_num=train_num,
                        data_time=train_timestamp,
                        data_type="TrainPred",
                        data_compare=Rs,
                        show_skip=5,
                        show_fps=20)
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
                        show_skip=5,
                        show_fps=20)
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

        hist_user_position = []
        hist_user_velocity = []
        hist_object_position = []
        hist_object_velocity = []
        hist_force = []
        hist_force_magnitude_orig = []
        hist_force_magnitude_processed = []
        hist_time = []
        hist_execution_time_com = []
        hist_execution_time_CalForce = []
        hist_execution_time_CalDynamic = []
        hist_com_count = []
        error_force = []

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

            if use_KalmanFilter:
                if com_count <= 1:
                    # Initialize the Kalman filter
                    kf = KalmanFilter3D(process_var, measurement_var, estimated_var)

                    # Initial state
                    kf.x[:3] = [0.0, 0.0, 0.0]
                    user_position = np.array([0.0, 0.0, 0.0]).reshape(1, dim)
                    user_velocity = np.array([0.0, 0.0, 0.0]).reshape(1, dim)
                else:
                    hist_time_np = np.array(hist_time).reshape(-1)
                    dt = hist_time_np[com_count] - hist_time_np[com_count - 1]
                    kf.predict(dt)

                    kf.update(user_position.reshape(dim))
                    state = kf.get_state()

                    estimated_position = state[:3]
                    estimated_velocity = state[3:6]
                    user_position = estimated_position.reshape(1, dim)
                    user_velocity = estimated_velocity.reshape(1, dim)
            else:
                user_velocity = np.array([data_velocity])
                user_velocity = get_coordinate_TouchX2Python(user_velocity)

            print(f"user_position = {user_position}, user_velocity = {user_velocity}")
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
            force = np.array(force) * 0.1

            force_max = np.max(force)
            if force_max > 2.0:
                force /= force_max * 2.0

            force_magnitude = np.linalg.norm(force)
            hist_force_magnitude_orig.append(force_magnitude.tolist())
            if force_magnitude > 7.0:
                force /= force_magnitude * 7.0
            force_magnitude = np.linalg.norm(force)
            hist_force_magnitude_processed.append(force_magnitude.tolist())

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

            hist_force.append(force.tolist())

            end_time_cal_force = time.time()
            calculation_time_force = end_time_cal_force - start_time_cal_force

            # predict next step dynamic
            start_time_cal_dynamic = time.time()

            hist_user_position.append(user_position.tolist())
            hist_user_velocity.append(user_velocity.tolist())
            hist_object_position.append(object_position.tolist())
            hist_object_velocity.append(object_velocity.tolist())

            pred_traj = forward_model_trained_onestep(R=object_position,
                                                      V=object_velocity,
                                                      R_lead=user_position,
                                                      V_lead=user_velocity)
            object_position = pred_traj.position[0, :, :]
            object_velocity = pred_traj.velocity[0, :, :]
            end_time_cal_dynamic = time.time()
            calculation_time_dynamic = end_time_cal_dynamic - start_time_cal_dynamic

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

            print(f"force = {force}")
            print(f"average update rate = {(1.0 / (render_time + const_numerical)):.2f} Hz")
            print(f"communication: {(communication_time * 1000.0):.2f} ms, "
                  f"ratio {(communication_time / (render_time + const_numerical) * 100.0):.2f}%")
            print(f"calculation: {(calculation_time * 1000.0):.2f} ms, "
                  f"ratio {(calculation_time / (render_time + const_numerical) * 100.0):.2f}%")
            print(f"cal - force: {(calculation_time_force * 1000.0):.2f} ms, "
                  f"ratio {(calculation_time_force / (calculation_time + const_numerical) * 100.0):.2f}%")
            print(f"cal - dynamic: {(calculation_time_dynamic * 1000.0):.2f} ms, "
                  f"ratio {(calculation_time_dynamic / (calculation_time + const_numerical) * 100.0):.2f}%\n")

        hist_user_position = np.array(hist_user_position)
        hist_user_velocity = np.array(hist_user_velocity)
        hist_force = np.array(hist_force)
        hist_time = np.array(hist_time)
        hist_object_position = np.array(hist_object_position)
        hist_object_velocity = np.array(hist_object_velocity)

        print(f"hist_user_position shape = {hist_user_position.shape}")
        print(f"hist_user_velocity shape = {hist_user_velocity.shape}")
        print(f"hist_object_position shape = {hist_object_position.shape}")
        print(f"hist_object_velocity shape = {hist_object_velocity.shape}")
        print(f"hist_force shape = {hist_force.shape}")
        print(f"hist_time shape = {hist_time.shape}")

        plot_trajectory(data_traj=hist_object_position,
                        data_num=len(hist_time),
                        data_time=hist_time,
                        data_type="render",
                        data_force=hist_force,
                        data_virtual=hist_user_position)
        print(">> Render trajectory saved\n")

        fig, axis = plt.subplots(3, 1)
        label_title_ax = ["X", "Y", "Z"]
        for i, ax in enumerate(axis):
            ax.plot(hist_time, hist_force[:, i])
            ax.set_title(label_title_ax[i])
        plt.suptitle(f"Render Force")
        plt.tight_layout()
        plt.savefig(_filename(f"render_force.png"))
        print(">> Plot render force\n")

        fig, axis = plt.subplots(3, 1)
        for i, ax in enumerate(axis):
            ax.plot(hist_time, hist_user_position[:, 0, i])
            ax.set_title(label_title_ax[i])
        plt.suptitle(f"User Position")
        plt.tight_layout()
        plt.savefig(_filename(f"render_user_position.png"))
        print(">> Plot user position\n")

        fig, axis = plt.subplots(3, 1)
        for i, ax in enumerate(axis):
            ax.plot(hist_time, hist_user_velocity[:, 0, i])
            ax.set_title(label_title_ax[i])
        plt.suptitle(f"User Velocity")
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

        fig, axis = plt.subplots(3, 1)
        axis[0].plot(hist_com_count, hist_execution_time_com)
        axis[0].set_yscale('log')
        axis[0].set_title("Communication Time")
        axis[1].plot(hist_com_count, hist_execution_time_CalForce)
        axis[1].set_yscale('log')
        axis[1].set_title("Force Calculation Time")
        axis[2].plot(hist_com_count, hist_execution_time_CalDynamic)
        axis[2].set_yscale('log')
        axis[2].set_title("Dynamic Calculation Time")
        plt.suptitle("Execution Time Observation")
        plt.tight_layout()
        plt.savefig(_filename(f"render_execution_time.png"))
        print(">> Plot execution time\n")


fire.Fire(Main)
