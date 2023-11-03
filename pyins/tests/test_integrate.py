import pandas as pd
from numpy.testing import assert_allclose
import numpy as np
from pyins import sim
from pyins.transform import difference_trajectories
from pyins.strapdown import compute_theta_and_dv, Integrator
from pyins.util import GYRO_COLS, ACCEL_COLS, THETA_COLS, DV_COLS


def test_coning_sculling():
    # Basically a smoke test, because the function is quite simple.
    imu = pd.DataFrame(data=np.zeros((10, 6)), columns=GYRO_COLS + ACCEL_COLS)

    imu.gyro_x = 0.01
    imu.gyro_z = -0.01
    imu.accel_z = 0.1

    dv_true = np.empty((9, 3))
    dv_true[:, 0] = 0
    dv_true[:, 1] = -0.5e-3
    dv_true[:, 2] = 0.1
    increments = compute_theta_and_dv(imu)
    assert_allclose(increments[THETA_COLS], imu[GYRO_COLS].iloc[1:], rtol=1e-10)
    assert_allclose(increments[DV_COLS], dv_true, rtol=1e-10)


def run_integration_test(reference_trajectory, imu, dt, sensor_type,
                         thresholds):
    increments = compute_theta_and_dv(imu, dt=dt if sensor_type == 'rate' else None)
    integrator = Integrator(dt, reference_trajectory.iloc[0])
    result = integrator.integrate(increments)
    diff = difference_trajectories(result, reference_trajectory).abs().max(axis=0)
    assert (diff < thresholds).all()


def test_integrate_stationary():
    total_time = 3600
    dt = 1e-1
    n = int(total_time / dt)

    lla = np.empty((n, 3))
    lla[:, 0] = 55.0
    lla[:, 1] = 37.0
    lla[:, 2] = 150.0

    rph = np.empty((n, 3))
    rph[:, 0] = -5.0
    rph[:, 1] = 10.0
    rph[:, 2] = 110.0

    thresholds = pd.Series({
        'north': 1e-3, 'east': 1e-3, 'down': 1e-2,
        'VN': 1e-6, 'VE': 1e-6, 'VD': 1e-5,
        'roll': 1e-8, 'pitch': 1e-8, 'heading': 1e-8
    })

    ref, imu = sim.generate_imu(dt, lla, rph, sensor_type='increment')
    run_integration_test(ref, imu, dt, 'increment', thresholds)

    ref, imu = sim.generate_imu(dt, lla, rph, sensor_type='rate')
    run_integration_test(ref, imu, dt, 'rate', thresholds)


def test_integrate_constant_velocity():
    dt = 1e-1
    total_time = 3600

    lla0 = [55.0, 37.0, 1500.0]
    velocity_n = [5.0, -3.0, 0.2]

    thresholds = pd.Series({
        'north': 1, 'east':1, 'down': 1,
        'VN': 1e-3, 'VE': 1e-3, 'VD': 1e-3,
        'roll': 1e-4, 'pitch': 1e-4, 'heading': 1e-4
    })

    ref, imu = sim.sinusoid_velocity_motion(dt, total_time, lla0, velocity_n,
                                            sensor_type='increment')
    run_integration_test(ref, imu, dt, 'increment', thresholds)

    ref, imu = sim.sinusoid_velocity_motion(dt, total_time, lla0, velocity_n,
                                            sensor_type='rate')
    run_integration_test(ref, imu, dt, 'rate', thresholds)
