from numpy.testing import (assert_, assert_allclose, run_module_suite,
                           assert_equal)
import numpy as np
import pandas as pd
from pyins.filt import (InertialSensor, LatLonObs, VeVnObs,
                        FeedforwardFilter, FeedbackFilter,
                        _refine_stamps, _kalman_correct, correct_traj)
from pyins.error_model import propagate_errors
from pyins import earth
from pyins import sim
from pyins.integrate import coning_sculling, Integrator
from pyins.transform import perturb_lla, difference_trajectories


def test_InertialSensor():
    s = InertialSensor()
    assert_equal(s.n_states, 0)
    assert_equal(s.n_noises, 0)
    assert_equal(len(s.states), 0)
    assert_equal(s.P.shape, (0, 0))
    assert_equal(s.q.shape, (0,))
    assert_equal(s.F.shape, (0, 0))
    assert_equal(s.G.shape, (0, 0))
    assert_equal(s.output_matrix().shape, (3, 0))

    s = InertialSensor(bias=0.1, bias_walk=0.2)
    assert_equal(s.n_states, 3)
    assert_equal(s.n_noises, 3)
    assert_equal(list(s.states.keys()), ['BIAS_1', 'BIAS_2', 'BIAS_3'])
    assert_equal(list(s.states.values()), [0, 1, 2])
    assert_allclose(s.P, 0.01 * np.identity(3))
    assert_equal(s.q, [0.2, 0.2, 0.2])
    assert_equal(s.F, np.zeros((3, 3)))
    assert_equal(s.G, np.identity(3))
    assert_equal(s.output_matrix(), np.identity(3))

    s = InertialSensor(scale=0.2, scale_walk=0.3)
    assert_equal(s.n_states, 3)
    assert_equal(s.n_noises, 3)
    assert_equal(list(s.states.keys()), ['SCALE_1', 'SCALE_2', 'SCALE_3'])
    assert_equal(list(s.states.values()), [0, 1, 2])
    assert_allclose(s.P, 0.04 * np.identity(3))
    assert_equal(s.q, [0.3, 0.3, 0.3])
    assert_equal(s.F, np.zeros((3, 3)))
    assert_equal(s.G, np.identity(3))
    assert_equal(s.output_matrix([1, 2, 3]), np.diag([1, 2, 3]))
    assert_equal(s.output_matrix([[1, -2, 2], [0.1, 2, 0.5]]),
                 np.array((np.diag([1, -2, 2]), np.diag([0.1, 2, 0.5]))))

    s = InertialSensor(corr_sd=0.1, corr_time=5)
    assert_equal(s.n_states, 3)
    assert_equal(s.n_noises, 3)
    assert_equal(list(s.states.keys()), ['CORR_1', 'CORR_2', 'CORR_3'])
    assert_equal(list(s.states.values()), [0, 1, 2])
    assert_allclose(s.P, 0.01 * np.identity(3))
    q = 0.1 * (2 / 5) ** 0.5
    assert_equal(s.q, [q, q, q])
    assert_allclose(s.F, -np.identity(3) / 5)
    assert_equal(s.G, np.identity(3))

    s = InertialSensor(bias=0.1, bias_walk=0.2, scale=0.3, scale_walk=0.4,
                       corr_sd=0.5, corr_time=10)
    assert_equal(s.n_states, 9)
    assert_equal(s.n_noises, 9)
    assert_equal(list(s.states.keys()),
                 ['BIAS_1', 'BIAS_2', 'BIAS_3', 'SCALE_1', 'SCALE_2',
                  'SCALE_3', 'CORR_1', 'CORR_2', 'CORR_3'])
    assert_equal(list(s.states.values()), np.arange(9))
    assert_allclose(s.P, np.diag([0.01, 0.01, 0.01, 0.09, 0.09, 0.09,
                                  0.25, 0.25, 0.25]))
    q_corr = 0.5 * (2 / 10) ** 0.5
    assert_equal(s.q, [0.2, 0.2, 0.2, 0.4, 0.4, 0.4, q_corr, q_corr, q_corr])
    assert_allclose(s.F, np.diag([0, 0, 0, 0, 0, 0, -1/10, -1/10, -1/10]))
    assert_equal(s.G, np.identity(9))

    H = s.output_matrix([1, 2, 3])
    assert_allclose(H, [[1, 0, 0, 1, 0, 0, 1, 0, 0],
                        [0, 1, 0, 0, 2, 0, 0, 1, 0],
                        [0, 0, 1, 0, 0, 3, 0, 0, 1]])

    H = s.output_matrix([[1, 2, 3], [-1, 2, 0.5]])
    assert_allclose(H[0], [[1, 0, 0, 1, 0, 0, 1, 0, 0],
                           [0, 1, 0, 0, 2, 0, 0, 1, 0],
                           [0, 0, 1, 0, 0, 3, 0, 0, 1]])
    assert_allclose(H[1], [[1, 0, 0, -1, 0, 0, 1, 0, 0],
                           [0, 1, 0, 0, 2, 0, 0, 1, 0],
                           [0, 0, 1, 0, 0, 0.5, 0, 0, 1]])


def test_LatLonObs():
    traj_point = pd.Series(data={
        'lat': 40,
        'lon': 30,
        'VE': 4,
        'VN': -3,
        'h': 15,
        'p': 0,
        'r': 0
    })
    obs_data = pd.DataFrame(index=[50])
    obs_data['lat'] = [40.0001]
    obs_data['lon'] = [30.0002]
    obs = LatLonObs(obs_data, 10)

    ret = obs.compute_obs(55, traj_point)
    assert_(ret is None)

    z, H, R = obs.compute_obs(50, traj_point)
    z_true = [np.deg2rad(-0.0002) * earth.R0 * np.cos(np.deg2rad(40)),
              np.deg2rad(-0.0001) * earth.R0]
    assert_allclose(z, z_true, rtol=1e-5)

    assert_allclose(H, [[1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0, 0]])

    assert_allclose(R, [[100, 0], [0, 100]])


def test_VeVnObs():
    traj_point = pd.Series(data={
        'lat': 40,
        'lon': 30,
        'VE': 4,
        'VN': -3,
        'h': 15,
        'p': 0,
        'r': 0
    })
    obs_data = pd.DataFrame(index=[50])
    obs_data['VE'] = [3]
    obs_data['VN'] = [-2]
    obs = VeVnObs(obs_data, 10)

    ret = obs.compute_obs(55, traj_point)
    assert_(ret is None)

    z, H, R = obs.compute_obs(50, traj_point)
    assert_allclose(z, [1, -1])
    assert_allclose(H, [[0, 0, 0, 1, 0, 0, 0, 0, -2],
                        [0, 0, 0, 0, 1, 0, 0, 0, -3]])


def test_refine_stamps():
    stamps = [2, 2, 5, 1, 10, 20]
    stamps = _refine_stamps(stamps, 2)
    stamps_true = [1, 2, 4, 5, 7, 9, 10, 12, 14, 16, 18, 20]
    assert_equal(stamps, stamps_true)


def test_kalman_correct():
    # As the implementation of standard Kalman correction formulas is
    # straightforward we use a sanity check, when the correct answer is
    # computed without complete formulas.
    P0 = np.array([[2, 0], [0, 1]], dtype=float)
    x0 = np.array([0, 0], dtype=float)

    z = np.array([1, 2])
    R = np.array([[3, 0], [0, 2]])
    H = np.identity(2)

    x_true = np.array([1 * 2 / (2 + 3), 2 * 1 / (1 + 2)])
    P_true = np.diag([1 / (1/2 + 1/3), 1 / (1/1 + 1/2)])

    x = x0.copy()
    P = P0.copy()
    _kalman_correct(x, P, z, H, R, None, None)
    assert_allclose(x, x_true)
    assert_allclose(P, P_true)

    x = x0.copy()
    P = P0.copy()
    _kalman_correct(x, P, z, H, R, np.array([1 / 2, 1 / 3]), None)
    x_true = np.array([1 * 2 / (2 + 3) / 2, 2 * 1 / (1 + 2) / 3])

    K1 = 0.4 * 1 / 2
    K2 = 1 / 3 * 1 / 3
    P_true = np.diag([(1 - K1)**2 * 2 + K1**2 * 3,
                      (1 - K2)**2 * 1 + K2**2 * 2])

    assert_allclose(x, x_true)
    assert_allclose(P, P_true)

    def create_gain_curve(params):
        L, F, C = params

        def gain_curve(q):
            if q > C:
                return 0
            if F < q <= C:
                return L * F * (C - q) / ((C - F) * q)
            elif L < q <= F:
                return L
            else:
                return q

        return gain_curve

    curve = create_gain_curve([0.5, 1, 5])
    x = x0.copy()
    P = P0.copy()
    _kalman_correct(x, P, z, H, R, None, curve)

    K1 = 0.4 * 0.5 / (23 / 30)**0.5
    K2 = 1/3 * 0.5 / (23 / 30)**0.5
    P_true = np.diag([(1 - K1)**2 * 2 + K1**2 * 3,
                      (1 - K2)**2 * 1 + K2**2 * 2])

    assert_allclose(x, [K1 * z[0], K2 * z[1]])
    assert_allclose(P, P_true)


def test_FeedforwardFilter():
    # Test that the results are reasonable on a static bench.
    dt = 1
    traj = pd.DataFrame(index=np.arange(1 * 3600))
    traj['lat'] = 50
    traj['lon'] = 60
    traj['alt'] = 100
    traj['VE'] = 0
    traj['VN'] = 0
    traj['VU'] = 0
    traj['h'] = 0
    traj['p'] = 0
    traj['r'] = 0

    np.random.seed(1)
    obs_data = pd.DataFrame(index=traj.index[::10])
    lla_obs = perturb_lla(traj.loc[::10, ['lat', 'lon', 'alt']],
                          10 * np.random.randn(len(obs_data), 3))
    obs_data[['lat', 'lon']] = lla_obs[:, :2]
    obs = LatLonObs(obs_data, 10)

    d_lat = 5
    d_lon = -3
    d_alt = 0
    d_VE = 1
    d_VN = -1
    d_VU = 0
    d_h = 0.1
    d_p = 0.03
    d_r = -0.02

    errors = propagate_errors(dt, traj, d_lat, d_lon, d_alt, d_VE, d_VN, d_VU,
                              d_h, d_p, d_r)
    traj_error = correct_traj(traj, -errors)

    f = FeedforwardFilter(dt, traj, 5, 1, 0.2, 0.05)
    res = f.run(traj_error, [obs])

    x = errors.loc[3000:]
    y = res.err.loc[3000:]

    assert_allclose(x.lat, y.lat, rtol=0, atol=10)
    assert_allclose(x.lon, y.lon, rtol=0, atol=10)
    assert_allclose(x.VE, y.VE, rtol=0, atol=1e-2)
    assert_allclose(x.VE, y.VE, rtol=0, atol=1e-2)
    assert_allclose(x.h, y.h, rtol=0, atol=1.5e-3)
    assert_allclose(x.p, y.p, rtol=0, atol=1e-4)
    assert_allclose(x.r, y.r, rtol=0, atol=1e-4)
    assert_(np.all(np.abs(res.residuals[0] < 4)))

    res = f.run_smoother(traj_error, [obs])

    # This smoother we don't need to wait until the filter converges,
    # the estimation accuracy is also improved some
    x = errors
    y = res.err

    assert_allclose(x.lat, y.lat, rtol=0, atol=10)
    assert_allclose(x.lon, y.lon, rtol=0, atol=10)
    assert_allclose(x.VE, y.VE, rtol=0, atol=1e-2)
    assert_allclose(x.VE, y.VE, rtol=0, atol=1e-2)
    assert_allclose(x.h, y.h, rtol=0, atol=1.5e-3)
    assert_allclose(x.p, y.p, rtol=0, atol=1e-4)
    assert_allclose(x.r, y.r, rtol=0, atol=1e-4)
    assert_(np.all(np.abs(res.residuals[0] < 4)))


def test_FeedbackFilter():
    dt = 0.9
    traj = pd.DataFrame(index=np.arange(1 * 3600))
    traj['lat'] = 50
    traj['lon'] = 60
    traj['alt'] = 100
    traj['VE'] = 0
    traj['VN'] = 0
    traj['VU'] = 0
    traj['h'] = 0
    traj['p'] = 0
    traj['r'] = 0

    _, gyro, accel = sim.from_position(dt, traj[['lat', 'lon', 'alt']],
                                       traj[['h', 'p', 'r']])
    theta, dv = coning_sculling(gyro, accel)

    np.random.seed(0)
    obs_data = pd.DataFrame(index=traj.index[::10])
    lla_obs = perturb_lla(traj.loc[::10, ['lat', 'lon', 'alt']],
                          10 * np.random.randn(len(obs_data), 3))
    obs_data[['lat', 'lon']] = lla_obs[:, :2]
    obs = LatLonObs(obs_data, 10)

    f = FeedbackFilter(dt, 5, 1, 0.2, 0.05)

    d_lat = 5
    d_lon = -3
    d_alt = 0
    d_VE = 1
    d_VN = -1
    d_VU = 0
    d_h = 0.1
    d_p = 0.03
    d_r = -0.02

    lla0 = perturb_lla(traj.loc[0, ['lat', 'lon', 'alt']],
                       [d_lon, d_lat, d_alt])
    integrator = Integrator(dt, lla0, [d_VE, d_VN, d_VU],
                            [d_h, d_p, d_r])
    res = f.run(integrator, theta, dv, observations=[obs])
    error = difference_trajectories(res.traj, traj)
    error = error.iloc[3000:]

    assert_allclose(error.lat, 0, rtol=0, atol=10)
    assert_allclose(error.lon, 0, rtol=0, atol=10)
    assert_allclose(error.VE, 0, rtol=0, atol=1e-2)
    assert_allclose(error.VN, 0, rtol=0, atol=2e-2)
    assert_allclose(error.h, 0, rtol=0, atol=1.5e-3)
    assert_allclose(error.p, 0, rtol=0, atol=1e-4)
    assert_allclose(error.r, 0, rtol=0, atol=1e-4)
    assert_(np.all(np.abs(res.residuals[0] < 4)))

    res = f.run_smoother(integrator, theta, dv, [obs])
    error = difference_trajectories(res.traj, traj)
    assert_allclose(error.lat, 0, rtol=0, atol=10)
    assert_allclose(error.lon, 0, rtol=0, atol=10)
    assert_allclose(error.VE, 0, rtol=0, atol=1e-2)
    assert_allclose(error.VN, 0, rtol=0, atol=1e-2)
    assert_allclose(error.h, 0, rtol=0, atol=1.5e-3)
    assert_allclose(error.p, 0, rtol=0, atol=1e-4)
    assert_allclose(error.r, 0, rtol=0, atol=1e-4)
    assert_(np.all(np.abs(res.residuals[0] < 4)))


if __name__ == '__main__':
    run_module_suite()
