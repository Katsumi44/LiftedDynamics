from typing import Callable, Tuple, TypeVar, Union

import jax.numpy as np
from jax import random
from jax_md import dataclasses, interpolate, quantity, simulate, space, util

static_cast = util.static_cast
# Types
Array = util.Array
f32 = util.f32
f64 = util.f64
ShiftFn = space.ShiftFn
T = TypeVar('T')
InitFn = Callable[..., T]
ApplyFn = Callable[[T], T]
Simulator = Tuple[InitFn, ApplyFn]
NVEState = simulate.NVEState
Schedule = Union[Callable[..., float], float]


# pylint: disable=invalid-name


def canonicalize_mass(mass):
    if isinstance(mass, float):
        return mass
    if mass.ndim == 2 and mass.shape[1] == 1:
        return mass
    elif mass.ndim == 1:
        return np.reshape(mass, (mass.shape[0], 1))
    elif mass.ndim == 0:
        return mass
    msg = (
        'Expected mass to be either a floating point number or a one-dimensional'
        'ndarray. Found {}.'.format(mass)
    )
    raise ValueError(msg)


class NVEStates():
    def __init__(self, states):
        self.position = states.position
        self.velocity = states.velocity
        self.force = states.force
        self.mass = states.mass
        self.index = 0

    def __len__(self):
        return len(self.position)

    def __getitem__(self, key):
        if isinstance(key, int):
            return NVEState(self.position[key], self.velocity[key],
                            self.position_lead[key], self.velocity_lead[key],
                            self.force[key], self.mass[key])
        else:
            return NVEState(self.position[key],
                            self.velocity[key],
                            self.position_lead[key],
                            self.velocity_lead[key],
                            self.force[key],
                            self.mass[key])

    def __iter__(self, ):
        return (self.__getitem__(i) for i in range(len(self)))


def nve(energy_or_force_fn: Callable[..., Array],
        shift_fn: ShiftFn,
        dt: float) -> Simulator:
    """Simulates a system in the NVE ensemble.
    Samples from the microcanonical ensemble in which the number of particles
    (N), the system volume (V), and the energy (E) are held constant. We use a
    standard velocity verlet integration scheme.
    Args:
      energy_or_force: A function that produces either an energy or a force from
        a set of particle positions specified as an ndarray of shape
        [n, spatial_dimension].
      shift_fn: A function that displaces positions, R, by an amount dR. Both R
        and dR should be ndarrays of shape [n, spatial_dimension].
      dt: Floating point number specifying the timescale (step size) of the
        simulation.
      quant: Either a quantity.Energy or a quantity.Force specifying whether
        energy_or_force is an energy or force respectively.
    Returns:
      See above.
    """
    force_fn = energy_or_force_fn

    dt_2 = 0.5 * dt ** 2

    def init_fun(R: Array,
                 V: Array,
                 mass=f32(1.0),
                 **kwargs) -> NVEState:
        mass = canonicalize_mass(mass)
        return NVEState(R, V, force_fn(R, V, **kwargs), mass)

    def apply_fun(state: NVEState, **kwargs) -> NVEState:
        R, V, F, mass = dataclasses.astuple(state)
        A = F / mass
        dR = V * dt + A * dt_2
        R, V = shift_fn(R, dR, V)
        F = force_fn(R, V, **kwargs)
        A_prime = F / mass
        V = V + f32(0.5) * (A + A_prime) * dt
        return NVEState(R, V, F, mass)

    return init_fun, apply_fun


@dataclasses.dataclass
class NVEState_DIY:
    position: Array
    # momentum: Array
    # this is the reason why it causes weird simulations!!! the original authors just copied the jax nve function,
    # without paying attention to the meaning of the variables: momentum instead of velocity. They use mass=1.0 so
    # there is no problem, but it is totally a misuse! (happens when you obtain the pred_traj.velocity)
    velocity: Array
    position_lead: Array
    velocity_lead: Array
    force: Array
    mass: Array

    # @property
    # def velocity(self) -> Array:
    #     return self.momentum / 1.0


class NVEStates_DIY():
    def __init__(self, states):
        self.position = states.position
        self.velocity = states.velocity
        self.position_lead = states.position_lead
        self.velocity_lead = states.velocity_lead
        self.force = states.force
        self.mass = states.mass
        self.index = 0

    def __len__(self):
        return len(self.position)

    def __getitem__(self, key):
        if isinstance(key, int):
            return NVEState(self.position[key], self.velocity[key],
                            self.position_lead[key], self.velocity_lead[key],
                            self.force[key], self.mass[key])
        else:
            return NVEState(self.position[key],
                            self.velocity[key],
                            self.position_lead[key],
                            self.velocity_lead[key],
                            self.force[key],
                            self.mass[key])

    def __iter__(self, ):
        return (self.__getitem__(i) for i in range(len(self)))


def nve_DIY(energy_or_force_fn: Callable[..., Array],
            shift_fn: ShiftFn,
            dt: float) -> Simulator:
    force_fn = energy_or_force_fn
    dt_2 = 0.5 * dt ** 2

    def init_fun(R: Array,
                 V: Array,
                 R_lead: Array,
                 V_lead: Array,
                 mass,
                 **kwargs) -> NVEState_DIY:
        mass = canonicalize_mass(mass)
        return NVEState_DIY(R, V, R_lead, V_lead, force_fn(R, V, R_lead, V_lead, **kwargs), mass)

    def apply_fun(state: NVEState_DIY, **kwargs) -> NVEState_DIY:
        R, V, R_lead, V_lead, F, mass = dataclasses.astuple(state)
        A = F / mass
        dR = V * dt + A * dt_2
        R, V = shift_fn(R, dR, V)
        F = force_fn(R, V, R_lead, V_lead, **kwargs)
        A_prime = F / mass
        V = V + f64(0.5) * (A + A_prime) * dt
        # V = V + A * dt
        return NVEState_DIY(R, V, R_lead, V_lead, F, mass)

    return init_fun, apply_fun
