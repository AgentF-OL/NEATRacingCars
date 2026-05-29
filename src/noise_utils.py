"""Sensor and actuator noise utilities.

Aligned with project spec §3.3: introduce noise in sensor readings and
actuator outputs to simulate physical-world uncertainty.
"""
import numpy as np

SENSOR_NOISE_STD = 0.03
ACTUATOR_NOISE_STD = 0.05


def add_sensor_noise(value, std=SENSOR_NOISE_STD):
    """Add Gaussian noise to a normalized sensor reading [0,1]."""
    noisy = value + np.random.normal(0.0, std)
    return float(np.clip(noisy, 0.0, 1.0))


def add_actuator_noise(value, std=ACTUATOR_NOISE_STD):
    """Add Gaussian noise to an actuator command [-1,1]."""
    noisy = value + np.random.normal(0.0, std)
    return float(np.clip(noisy, -1.0, 1.0))
