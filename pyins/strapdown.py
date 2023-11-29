"""Strapdown INS integration algorithms."""
import numpy as np
import pandas as pd
from . import transform
import math
from numba import jit
import numpy as np
from . import earth
from . transform import DEG_TO_RAD, RAD_TO_DEG

from ._integrate import integrate_fast
# from ._integrate_py import integrate_fast
EARTH_F = (1 - earth.E2) ** 0.5 * earth.GP / earth.GE - 1


@jit
def _dcm_from_rv(rv, out):
    rv1, rv2, rv3 = rv
    rv11 = rv1 * rv1
    rv12 = rv1 * rv2
    rv13 = rv1 * rv3
    rv22 = rv2 * rv2
    rv23 = rv2 * rv3
    rv33 = rv3 * rv3

    norm2 = rv11 + rv22 + rv33
    if norm2 > 1e-6:
        norm = norm2 ** 0.5
        k1 = math.sin(norm) / norm
        k2 = (1.0 - math.cos(norm)) / norm2
    else:
        norm4 = norm2 * norm2
        k1 = 1.0 - norm2 / 6.0 + norm4 / 120.0
        k2 = 0.5 - norm2 / 24.0 + norm4 / 720.0

    out[0, :] = 1.0 - k2*(rv33 + rv22), -k1*rv3 + k2*rv12, k1*rv2 + k2*rv13
    out[1, :] = k1*rv3 + k2*rv12, 1.0 - k2*(rv33 + rv11), -k1*rv1 + k2*rv23
    out[2, :] = -k1*rv2 + k2*rv13, k1*rv1 + k2*rv23, 1.0 - k2*(rv22 + rv11)
    return out


@jit
def _mv_dot3(A, b):
    b1, b2, b3 = b
    v1 = A[0, 0] * b1 + A[0, 1] * b2 + A[0, 2] * b3
    v2 = A[1, 0] * b1 + A[1, 1] * b2 + A[1, 2] * b3
    v3 = A[2, 0] * b1 + A[2, 1] * b2 + A[2, 2] * b3
    return v1, v2, v3


@jit
def _v_add3(a, b):
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


@jit
def _v_cross3(a, b):
    a1, a2, a3 = a
    b1, b2, b3 = b
    c1 = (-a3 * b2 + a2 * b3)
    c2 = ( a3 * b1 - a1 * b3)
    c3 = (-a2 * b1 + a1 * b2)

    return c1, c2, c3


@jit
def _gravity(lat, alt):
    sin_lat = math.sin(lat * DEG_TO_RAD)
    sin_lat2 = sin_lat * sin_lat
    return (earth.GE * (1 + EARTH_F * sin_lat2)
            / (1 - earth.E2 * sin_lat2)**0.5
            * (1 - 2 * alt / earth.R0))


@jit
def integrate_fast_py(dt, lla, Vn, Cnb, theta, dv, offset=0):
    """Mechanization in a rotating navigation frame.

    Parameters
    ----------
    lla : array_like, shape: (N, 3)
        Geodetic coordinates [rad].
    V : array_like, shape: (N, 3)
        Velocity in local level frame [m/s].
    Cnb : array_like, shape: (N, 3, 3)
        Direct cosine matrix.
    theta : array_like, shape: (N, 3)
        Rotation increments in body frame.
    dv : array_like, shape: (N, 3)
        Velocity increments in body frame.
    dt : float
        Time step.
    offset : int
        Offset index for intital conditions in arrays: lla, Vn, Cnb.
    """
    C = np.empty((3,3))
    dCn = np.empty((3,3))
    dCb = np.empty((3,3))
    V = np.empty(3)
    V_new = np.empty(3)

    for i in range(theta.shape[0]):
        j = i + offset

        lat, lon, alt = lla[j]
        sin_lat = math.sin(lat * DEG_TO_RAD)
        sin_lat2 = sin_lat * sin_lat
        cos_lat = (1.0 - sin_lat2) ** 0.5
        tan_lat = sin_lat / cos_lat

        V = Vn[j]
        VN, VE, VD = V

        x = 1 - earth.E2 * sin_lat2
        re = earth.R0 / (x ** 0.5)
        rn = re * (1 - earth.E2) / x + alt
        re += alt

        Omega = (earth.RATE * cos_lat, 0, -earth.RATE * sin_lat)
        rho = (VE / re, -VN / rn, -VE / re * tan_lat)
        chi = _v_add3(Omega, rho)
        w = _v_add3(Omega, chi)

        dv_n = _mv_dot3(Cnb[j], dv[i])
        corriolis1 = _v_cross3(w, V)
        corriolis2 = _v_cross3(chi, dv_n)

        VN_new = VN + (dv_n[0] - dt * (corriolis1[0] + 0.5 * corriolis2[0]))
        VE_new = VE + (dv_n[1] - dt * (corriolis1[1] + 0.5 * corriolis2[1]))
        VD_new = VD + (dv_n[2] - dt * (corriolis1[2] + 0.5 * corriolis2[2]))
        VD_new += dt * _gravity(lat, (alt - dt * 0.5 * VD))

        Vn[j + 1] = VN_new, VE_new, VD_new
        # print(Vn[j + 1])
        # return

        VN = 0.5 * (VN + VN_new)
        VE = 0.5 * (VE + VE_new)
        VD = 0.5 * (VD + VD_new)

        rho = (VE / re, -VN / rn, -VE / re * tan_lat)
        chi = _v_add3(Omega, rho)

        lla[j + 1, 0] = lat - RAD_TO_DEG * rho[1] * dt
        lla[j + 1, 1] = lon + RAD_TO_DEG * rho[0] / cos_lat * dt
        lla[j + 1, 2] = alt - VD * dt

        xi = (-dt * chi[0], -dt * chi[1], -dt * chi[2])
        dCn = _dcm_from_rv(xi, dCn)
        dCb = _dcm_from_rv(theta[i], dCb)

        C[:, 0] = _mv_dot3(dCn, Cnb[j, :, 0])
        C[:, 1] = _mv_dot3(dCn, Cnb[j, :, 1])
        C[:, 2] = _mv_dot3(dCn, Cnb[j, :, 2])

        Cnb[j+1, :, 0] = _mv_dot3(C, dCb[:, 0])
        Cnb[j+1, :, 1] = _mv_dot3(C, dCb[:, 1])
        Cnb[j+1, :, 2] = _mv_dot3(C, dCb[:, 2])


def compute_theta_and_dv(gyro, accel, dt=None):
    """Compute attitude and velocity increments from IMU readings.

    This function transforms raw gyro and accelerometer readings into
    rotation vectors and velocity increments by applying coning and sculling
    corrections and accounting for IMU rotation during a sampling period.

    The algorithm assumes a linear model for the angular velocity and the
    specific force described in [1]_ and [2]_.

    Parameters
    ----------
    gyro : array_like, shape (n_readings, 3)
        Gyro readings.
    accel : array_like, shape (n_readings, 3)
        Accelerometer readings.
    dt : float or None, optional
        If None (default), `gyro` and `accel` are assumed to contain integral
        increments. Float is interpreted as the sampling rate of rate sensors.

    Returns
    -------
    theta : ndarray, shape (n_readings, 3)
        Estimated rotation vectors.
    dv : ndarray, shape (n_readings, 3)
        Estimated velocity increments.

    References
    ----------
    .. [1] P. G. Savage, "Strapdown Inertial Navigation Integration Algorithm
           Design Part 1: Attitude Algorithms", Journal of Guidance, Control,
           and Dynamics 1998, Vol. 21, no. 2.
    .. [2] P. G. Savage, "Strapdown Inertial Navigation Integration Algorithm
           Design Part 2: Velocity and Position Algorithms", Journal of
           Guidance, Control, and Dynamics 1998, Vol. 21, no. 2.
    """
    gyro = np.asarray(gyro)
    accel = np.asarray(accel)

    if dt is not None:
        a_gyro = gyro[:-1]
        b_gyro = gyro[1:] - gyro[:-1]
        a_accel = accel[:-1]
        b_accel = accel[1:] - accel[:-1]
        alpha = (a_gyro + 0.5 * b_gyro) * dt
        dv = (a_accel + 0.5 * b_accel) * dt

        coning = np.cross(a_gyro, b_gyro) * dt**2 / 12
        sculling = (np.cross(a_gyro, b_accel) +
                    np.cross(a_accel, b_gyro)) * dt**2/12

        return alpha + coning, dv + sculling + 0.5 * np.cross(alpha, dv)

    coning = np.vstack((np.zeros(3), np.cross(gyro[:-1], gyro[1:]) / 12))
    sculling = np.vstack((np.zeros(3),
                          (np.cross(gyro[:-1], accel[1:]) +
                           np.cross(accel[:-1], gyro[1:])) / 12))

    return gyro + coning, accel + sculling + 0.5 * np.cross(gyro, accel)


class StrapdownIntegrator:
    """Integrate inertial readings by strapdown algorithm.

    The algorithm described in [1]_ and [2]_ is used with slight
    simplifications. The position is updated using the trapezoid rule.

    Parameters
    ----------
    dt : float
        Sensors sampling period.
    lla : array_like, shape (3,)
        Initial latitude, longitude and altitude.
    velocity_n: array_like, shape (3,)
        Initial velocity in NED frame.
    rph : array_like, shape (3,)
        Initial heading, pitch and roll.
    with_altitude : bool, optional
        Whether to compute altitude and vertical velocity. Default is True.
        If False, then vertical velocity is set to zero and altitude is kept
        as constant.

    Attributes
    ----------
    trajectory : DataFrame
        Computed trajectory so far.

    See Also
    --------
    coning_sculling : Apply coning and sculling corrections.

    References
    ----------
    .. [1] P. G. Savage, "Strapdown Inertial Navigation Integration Algorithm
           Design Part 1: Attitude Algorithms", Journal of Guidance, Control,
           and Dynamics 1998, Vol. 21, no. 2.
    .. [2] P. G. Savage, "Strapdown Inertial Navigation Integration Algorithm
           Design Part 2: Velocity and Position Algorithms", Journal of
           Guidance, Control, and Dynamics 1998, Vol. 21, no. 2.
    """
    TRAJECTORY_COLUMNS = ['lat', 'lon', 'alt', 'VN', 'VE', 'VD',
                          'roll', 'pitch', 'heading']
    INITIAL_SIZE = 10000

    def __init__(self, dt, lla, velocity_n, rph, with_altitude=True):
        self.dt = dt
        self.with_altitude = with_altitude
        if not with_altitude:
            velocity_n[2] = 0.0

        self.lla = np.empty((self.INITIAL_SIZE, 3))
        self.velocity_n = np.empty((self.INITIAL_SIZE, 3))
        self.Cnb = np.empty((self.INITIAL_SIZE, 3, 3))

        self.trajectory = None

        self._init_values = [lla, velocity_n, rph]
        self.reset()

    def reset(self):
        """Clear computed trajectory except the initial point."""
        lla, velocity_n, rph = self._init_values
        self.lla[0] = lla
        self.velocity_n[0] = velocity_n
        self.Cnb[0] = transform.mat_from_rph(rph)
        self.trajectory = pd.DataFrame(
            data=np.atleast_2d(np.hstack((lla, velocity_n, rph))),
            columns=self.TRAJECTORY_COLUMNS,
            index=pd.Index([0], name='stamp'))

    def integrate(self, theta, dv):
        """Integrate inertial readings.

        The integration continues from the last computed value.

        Parameters
        ----------
        theta, dv : array_like, shape (n_readings, 3)
            Rotation vectors and velocity increments computed from gyro and
            accelerometer readings after applying coning and sculling
            corrections.

        Returns
        -------
        traj_last : DataFrame
            Added chunk of the trajectory. It contains n_readings + 1 rows
            including the last point before `theta` and `dv` where integrated.
        """
        theta = np.asarray(theta)
        dv = np.asarray(dv)

        n_data = self.trajectory.shape[0]
        n_readings = theta.shape[0]
        size = self.lla.shape[0]

        required_size = n_data + n_readings
        if required_size > size:
            new_size = max(2 * size, required_size)
            self.lla.resize((new_size, 3), refcheck=False)
            self.velocity_n.resize((new_size, 3), refcheck=False)
            self.Cnb.resize((new_size, 3, 3), refcheck=False)
        for i in range(40):
            self.dt += i * 1e-20
            integrate_fast(self.dt, self.lla, self.velocity_n, self.Cnb,
                              theta, dv, n_data-1)
        rph = transform.mat_to_rph(self.Cnb[n_data:n_data + n_readings])
        index = pd.Index(self.trajectory.index[-1] + 1 + np.arange(n_readings),
                         name='stamp')
        trajectory = pd.DataFrame(index=index)
        trajectory[['lat', 'lon', 'alt']] = self.lla[n_data:
                                                     n_data + n_readings]
        trajectory[['VN', 'VE', 'VD']] = self.velocity_n[n_data:
                                                         n_data + n_readings]
        trajectory[['roll', 'pitch', 'heading']] = rph

        self.trajectory = pd.concat([self.trajectory, trajectory])

        return self.trajectory.iloc[-n_readings - 1:]

    def get_state(self):
        """Get current integrator state.

        Returns
        -------
        trajectory_point : pd.Series
            Trajectory point.
        """
        return self.trajectory.iloc[-1]

    def set_state(self, trajectory_point):
        """Set (overwrite) the current integrator state.

        Parameters
        ----------
        trajectory_point : pd.Series
            Trajectory point.
        """
        i = len(self.trajectory) - 1
        self.lla[i] = trajectory_point[['lat', 'lon', 'alt']]
        self.velocity_n[i] = trajectory_point[['VN', 'VE', 'VD']]
        self.Cnb[i] = transform.mat_from_rph(
            trajectory_point[['roll', 'pitch', 'heading']])
        self.trajectory.iloc[-1] = trajectory_point
