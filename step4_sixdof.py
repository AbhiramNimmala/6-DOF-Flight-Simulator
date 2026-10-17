"""
Flight dynamics simulator - Step 4
Six degrees of freedom. Quaternion attitude, rigid body rotation,
aerodynamic torque from the CP-CG offset.

This is where the simulator stops assuming the rocket points along its
velocity vector and starts working out where it actually points.

STATE VECTOR (13 elements)
    [0:3]   position in inertial frame       (m)
    [3:6]   velocity in inertial frame       (m/s)
    [6:10]  attitude quaternion [w,x,y,z]    body -> inertial
    [10:13] angular velocity in BODY frame   (rad/s)

Body frame: +z along the long axis toward the nose, origin at the CG.

WHY QUATERNIONS
    Euler angles have singularities. Pitch through 90 degrees and yaw and
    roll collapse onto the same axis - gimbal lock - and the equations blow
    up. A rocket that pitches over does exactly that. TEST 2 below
    demonstrates the failure numerically so you can see it rather than
    take it on faith.

    Quaternions have no singularity. The cost is that they drift off unit
    norm through accumulated floating point error, so we renormalize every
    step. That is a cheap, well-understood fix.

Run:  python step4_sixdof.py
"""

import numpy as np
import matplotlib.pyplot as plt

G0 = 9.80665
R_AIR = 287.05287

I_POS = slice(0, 3)
I_VEL = slice(3, 6)
I_QUAT = slice(6, 10)
I_OMEGA = slice(10, 13)


# ----------------------------------------------------------------------
# Atmosphere (from step 2)
# ----------------------------------------------------------------------

_LAYERS = [
    (0.0, 288.15, 101325.0, -0.0065), (11000.0, 216.65, 22632.06, 0.0),
    (20000.0, 216.65, 5474.889, 0.001), (32000.0, 228.65, 868.0187, 0.0028),
    (47000.0, 270.65, 110.9063, 0.0), (51000.0, 270.65, 66.93887, -0.0028),
    (71000.0, 214.65, 3.956420, -0.002), (84852.0, 186.946, 0.3734, 0.0),
]


def atmosphere(h):
    h = float(np.clip(h, 0.0, 86000.0))
    for i in range(len(_LAYERS) - 1, -1, -1):
        hb, tb, pb, lapse = _LAYERS[i]
        if h >= hb:
            break
    dh = h - hb
    if lapse == 0.0:
        t, p = tb, pb * np.exp(-G0 * dh / (R_AIR * tb))
    else:
        t = tb + lapse * dh
        p = pb * (t / tb) ** (-G0 / (lapse * R_AIR))
    rho = p / (R_AIR * t)
    return t, p, rho, np.sqrt(1.4 * R_AIR * t)


# ----------------------------------------------------------------------
# Quaternion algebra
# ----------------------------------------------------------------------
# Convention: q = [w, x, y, z], unit norm, rotating BODY vectors into the
# INERTIAL frame. q = [1,0,0,0] means the body frame is aligned with the
# inertial frame.

def quat_normalize(q):
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_multiply(a, b):
    """Hamilton product. Applying b then a."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_to_matrix(q):
    """Rotation matrix taking body-frame vectors to the inertial frame."""
    w, x, y, z = quat_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quat_derivative(q, omega_body):
    """
    dq/dt = 0.5 * q (x) [0, omega_body]

    The angular velocity is expressed in the body frame, which is why the
    pure quaternion goes on the right of the product.
    """
    return 0.5 * quat_multiply(q, np.concatenate([[0.0], omega_body]))


def quat_from_axis_angle(axis, angle):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    return np.concatenate([[np.cos(angle / 2)], axis * np.sin(angle / 2)])


def quat_to_euler(q):
    """ZYX Euler angles (yaw, pitch, roll) in radians. For reporting only."""
    w, x, y, z = quat_normalize(q)
    sin_p = 2 * (w * y - z * x)
    sin_p = np.clip(sin_p, -1.0, 1.0)
    return np.array([
        np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)),  # yaw
        np.arcsin(sin_p),                                           # pitch
        np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),  # roll
    ])


# ----------------------------------------------------------------------
# Vehicle
# ----------------------------------------------------------------------

class Body:
    """
    A rigid axisymmetric vehicle.

    cp_offset is the signed distance along body +z from the CG to the
    centre of pressure. NEGATIVE means the CP sits aft of the CG, which is
    the stable configuration. Positive is unstable and will tumble, which
    TEST 4 demonstrates on purpose.
    """

    def __init__(self, mass, length, diameter, cd, cn_alpha, cp_offset):
        self.mass = mass
        self.length = length
        self.diameter = diameter
        self.radius = diameter / 2.0
        self.area = np.pi * self.radius ** 2
        self.cd = cd
        self.cn_alpha = cn_alpha          # per radian
        self.cp_offset = cp_offset

        # Inertia tensor for a uniform cylinder about its CG, body axes.
        i_long = 0.5 * mass * self.radius ** 2
        i_trans = mass * (3 * self.radius ** 2 + length ** 2) / 12.0
        self.inertia = np.diag([i_trans, i_trans, i_long])
        self.inertia_inv = np.linalg.inv(self.inertia)

    @property
    def static_margin(self):
        """Stability in calibers. Positive is stable."""
        return -self.cp_offset / self.diameter


# ----------------------------------------------------------------------
# 6-DOF dynamics
# ----------------------------------------------------------------------

def make_derivative(body, gravity=True, thrust_fn=None):
    """Build the 13-element derivative function."""

    def derivative(t, state):
        pos = state[I_POS]
        vel = state[I_VEL]
        q = quat_normalize(state[I_QUAT])
        omega = state[I_OMEGA]

        R = quat_to_matrix(q)              # body -> inertial
        force_inertial = np.zeros(3)
        torque_body = np.zeros(3)

        if gravity:
            force_inertial += np.array([0.0, 0.0, -body.mass * G0])

        if thrust_fn is not None:
            # Thrust acts along the body axis, through the CG: no torque.
            force_inertial += R @ (np.array([0.0, 0.0, 1.0]) * thrust_fn(t))

        speed = np.linalg.norm(vel)
        if speed > 1e-6:
            _, _, rho, _ = atmosphere(pos[2])
            q_dyn = 0.5 * rho * speed ** 2

            # Velocity in body coordinates. R is orthonormal so R.T inverts it.
            v_body = R.T @ vel
            v_axial = v_body[2]
            v_lat = np.array([v_body[0], v_body[1], 0.0])
            lat_mag = np.linalg.norm(v_lat)

            # Angle of attack: angle between the body axis and the velocity.
            alpha = np.arctan2(lat_mag, abs(v_axial))

            # Axial drag, opposing the inertial velocity.
            drag = q_dyn * body.cd * body.area
            force_inertial -= drag * (vel / speed)

            if lat_mag > 1e-9:
                # Normal force: perpendicular to the body axis, opposing the
                # lateral component of velocity. Applied at the CP.
                normal_mag = q_dyn * body.area * body.cn_alpha * np.sin(alpha)
                f_normal_body = -normal_mag * (v_lat / lat_mag)

                force_inertial += R @ f_normal_body

                # Torque = r x F, with r from CG to CP along the body axis.
                r_cp = np.array([0.0, 0.0, body.cp_offset])
                torque_body += np.cross(r_cp, f_normal_body)

            # Pitch damping. A rotating body drags through the air and that
            # opposes the rotation. Without this a stable rocket oscillates
            # forever instead of settling.
            damping = q_dyn * body.area * body.cn_alpha * (body.length ** 2) / (
                2.0 * max(speed, 1e-6))
            torque_body -= damping * np.array([omega[0], omega[1], 0.0])

        # Euler's rigid body equations, body frame:
        #   I * omega_dot = torque - omega x (I * omega)
        # The cross term is what makes a spinning body precess.
        omega_dot = body.inertia_inv @ (
            torque_body - np.cross(omega, body.inertia @ omega))

        deriv = np.zeros(13)
        deriv[I_POS] = vel
        deriv[I_VEL] = force_inertial / body.mass
        deriv[I_QUAT] = quat_derivative(q, omega)
        deriv[I_OMEGA] = omega_dot
        return deriv

    return derivative


def step_rk4(f, t, state, dt):
    k1 = f(t, state)
    k2 = f(t + dt / 2, state + dt / 2 * k1)
    k3 = f(t + dt / 2, state + dt / 2 * k2)
    k4 = f(t + dt, state + dt * k3)
    out = state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    # Renormalize the quaternion. RK4 does not preserve unit norm, so this
    # correction has to happen every step or attitude slowly corrupts.
    out[I_QUAT] = quat_normalize(out[I_QUAT])
    return out


def integrate(f, state0, t0, t_end, dt, event=None):
    n = int(round((t_end - t0) / dt))
    ts, states = [t0], [np.array(state0, float)]
    state = np.array(state0, float)
    for i in range(n):
        t = t0 + i * dt
        nxt = step_rk4(f, t, state, dt)
        t_next = t0 + (i + 1) * dt
        if event is not None and event(t, state) > 0.0 >= event(t_next, nxt):
            ts.append(t_next)
            states.append(nxt)
            break
        state = nxt
        ts.append(t_next)
        states.append(state.copy())
    return np.array(ts), np.array(states)


def make_state(pos, vel, quat=None, omega=None):
    s = np.zeros(13)
    s[I_POS] = pos
    s[I_VEL] = vel
    s[I_QUAT] = quat if quat is not None else [1.0, 0.0, 0.0, 0.0]
    s[I_OMEGA] = omega if omega is not None else [0.0, 0.0, 0.0]
    return s


def angle_of_attack(state):
    """Angle between the body axis and the velocity vector, in degrees."""
    vel = state[I_VEL]
    speed = np.linalg.norm(vel)
    if speed < 1e-6:
        return 0.0
    axis = quat_to_matrix(state[I_QUAT]) @ np.array([0.0, 0.0, 1.0])
    cos_a = np.clip(np.dot(axis, vel / speed), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


# ======================================================================
# TESTS
# ======================================================================

def test_quaternion_norm():
    """Norm must stay at 1. Drift here corrupts every attitude downstream."""
    print("TEST 1 - quaternion norm preservation")
    body = Body(mass=0.6, length=1.0, diameter=0.054, cd=0.45,
                cn_alpha=10.0, cp_offset=-0.10)
    f = make_derivative(body, gravity=False)
    # Spin about a tilted axis so all four components stay active.
    s0 = make_state([0, 0, 10000], [0, 0, 0],
                    omega=[3.0, 2.0, 8.0])
    ts, states = integrate(f, s0, 0.0, 60.0, 0.002)
    norms = np.linalg.norm(states[:, I_QUAT], axis=1)
    dev = np.abs(norms - 1.0).max()
    print(f"  steps               : {len(ts)}")
    print(f"  max |‖q‖ - 1|       : {dev:.3e}")
    assert dev < 1e-9, "quaternion norm is drifting"
    print("  PASS\n")


def euler_to_quat(yaw, pitch, roll):
    """ZYX Euler angles to quaternion, for the gimbal lock demonstration."""
    qz = quat_from_axis_angle([0, 0, 1], yaw)
    qy = quat_from_axis_angle([0, 1, 0], pitch)
    qx = quat_from_axis_angle([1, 0, 0], roll)
    return quat_multiply(qz, quat_multiply(qy, qx))


def test_gimbal_lock():
    """
    Demonstrate the Euler-angle singularity numerically.

    At 90 degrees of pitch the yaw axis and the roll axis line up, so yaw
    and roll stop being independent. With this ZYX convention, pitching 90
    degrees maps the body x-axis onto -z, so a roll acts about the same axis
    as yaw. Only the combination (yaw - roll) survives: (+d, +d) and
    (-d, -d) are two different angle triples describing exactly the same
    orientation. The mapping has collapsed and can no longer be inverted.

    Below, `attitude gap` is how far apart the two attitudes actually are.
    As pitch approaches 90 it goes to zero - the angles still differ by a
    full 0.2 rad each, but they now describe the same thing, so you cannot
    invert the mapping and a controller reading those angles is blind.
    """
    print("TEST 2 - gimbal lock, demonstrated rather than asserted")
    d = 0.1
    print(f"  {'pitch (deg)':>12} {'attitude gap':>15} {'euler sens.':>14}")
    prev_gap = None
    for pitch_deg in [0.0, 45.0, 80.0, 88.0, 89.5, 89.99, 90.0]:
        p = np.radians(pitch_deg)
        qa = euler_to_quat(+d, p, +d)
        qb = euler_to_quat(-d, p, -d)
        # Distance between the two attitudes, accounting for q and -q being
        # the same rotation.
        gap = min(np.linalg.norm(qa - qb), np.linalg.norm(qa + qb))

        # Conditioning of recovering angles from attitude: perturb the
        # attitude slightly, see how violently yaw and roll respond.
        eps = 1e-6
        q0 = euler_to_quat(0.0, p, 0.0)
        q1 = quat_normalize(q0 + eps * np.array([0, 1, 0, 0]))
        e0, e1 = quat_to_euler(q0), quat_to_euler(q1)
        sens = (abs(e1[0] - e0[0]) + abs(e1[2] - e0[2])) / eps

        print(f"  {pitch_deg:>12.2f} {gap:>15.3e} {sens:>14.1f}")
        prev_gap = gap

    print("  At 90 deg the gap hits zero: (yaw +0.1, roll +0.1) and")
    print("  (yaw -0.1, roll -0.1) are the SAME attitude despite differing by")
    print("  0.2 rad in both angles. And the sensitivity column shows")
    print("  recovering those angles going from 2 to over three million.")
    assert prev_gap < 1e-12, "the two attitudes should coincide exactly at 90 deg"
    print("  PASS - this is why the state carries a quaternion.\n")


def test_free_rotation_conserves():
    """Torque-free rotation must conserve angular momentum and energy."""
    print("TEST 3 - torque-free rotation conserves momentum and energy")
    body = Body(mass=0.6, length=1.0, diameter=0.054, cd=0.0,
                cn_alpha=0.0, cp_offset=-0.10)
    f = make_derivative(body, gravity=False)
    s0 = make_state([0, 0, 50000], [0, 0, 0], omega=[2.0, 5.0, 1.0])
    ts, states = integrate(f, s0, 0.0, 100.0, 0.001)

    L0 = E0 = None
    worstL = worstE = 0.0
    for s in states:
        R = quat_to_matrix(s[I_QUAT])
        w = s[I_OMEGA]
        L = np.linalg.norm(R @ (body.inertia @ w))   # inertial frame
        E = 0.5 * float(w @ (body.inertia @ w))
        if L0 is None:
            L0, E0 = L, E
        worstL = max(worstL, abs(L - L0) / L0)
        worstE = max(worstE, abs(E - E0) / E0)

    print(f"  angular momentum drift : {worstL:.3e}")
    print(f"  kinetic energy drift   : {worstE:.3e}")
    assert worstL < 1e-9 and worstE < 1e-9, "rigid body integration is lossy"
    print("  PASS\n")


def test_weathercock():
    """A stable rocket should turn into the wind on its own."""
    print("TEST 4 - stability: stable weathercocks, unstable tumbles")

    for label, cp_offset in [("stable  (CP aft of CG)", -0.15),
                             ("neutral (CP at CG)   ", 0.0),
                             ("unstable(CP fwd of CG)", +0.15)]:
        body = Body(mass=0.6, length=1.2, diameter=0.054, cd=0.45,
                    cn_alpha=12.0, cp_offset=cp_offset)
        f = make_derivative(body, gravity=False)
        # Flying fast, nose tilted 20 degrees off the velocity vector.
        q0 = quat_from_axis_angle([0, 1, 0], np.radians(20.0))
        s0 = make_state([0, 0, 1000.0], [0, 0, 150.0], quat=q0)
        ts, states = integrate(f, s0, 0.0, 6.0, 0.0005)

        aoa = np.array([angle_of_attack(s) for s in states])
        # Separate the two ways AoA can close: the BODY rotating to meet the
        # flow, or the VELOCITY being pushed around to meet the body. Only
        # the first is weathercocking.
        body_turn = np.degrees(np.abs(quat_to_euler(states[-1, I_QUAT])[1]
                                      - quat_to_euler(states[0, I_QUAT])[1]))
        print(f"  {label}  margin {body.static_margin:+5.2f} cal   "
              f"AoA {aoa[0]:5.1f} -> {aoa[-1]:6.1f} deg   "
              f"body rotated {body_turn:5.1f} deg")

        if cp_offset < 0:
            assert aoa[-1] < aoa[0] / 2, "stable rocket should align with flow"
        if cp_offset > 0:
            assert aoa[-1] > aoa[0], "unstable rocket should diverge"

    print("  Note the neutral case: AoA still closes, but the body barely")
    print("  rotates. The normal force is pushing the VELOCITY around to meet")
    print("  the body rather than the body turning into the wind. Same AoA")
    print("  reading, completely different mechanism - which is why the body")
    print("  rotation column matters.")
    print("  PASS - the sign of the CP-CG offset decides everything\n")
    return None


def test_matches_3dof():
    """
    With the nose exactly on the velocity vector and no disturbance, 6-DOF
    must reproduce the 3-DOF answer. If it does not, the new force
    bookkeeping broke something that used to work.
    """
    print("TEST 5 - 6-DOF reduces to 3-DOF at zero angle of attack")
    body = Body(mass=0.6, length=1.2, diameter=0.054, cd=0.45,
                cn_alpha=12.0, cp_offset=-0.15)

    f6 = make_derivative(body)
    s0 = make_state([0, 0, 0], [0, 0, 200.0])
    ts6, st6 = integrate(f6, s0, 0.0, 40.0, 0.001,
                         event=lambda t, s: s[5])
    apogee6 = st6[-1, 2]

    # Equivalent 3-DOF: same mass, drag, no rotation at all.
    def f3(t, s):
        vel = s[I_VEL]
        speed = np.linalg.norm(vel)
        acc = np.array([0.0, 0.0, -G0])
        if speed > 1e-9:
            _, _, rho, _ = atmosphere(s[2])
            d = 0.5 * rho * speed ** 2 * body.cd * body.area
            acc = acc - (d / body.mass) * (vel / speed)
        out = np.zeros(13)
        out[I_POS] = vel
        out[I_VEL] = acc
        return out

    ts3, st3 = integrate(f3, s0.copy(), 0.0, 40.0, 0.001,
                         event=lambda t, s: s[5])
    apogee3 = st3[-1, 2]

    diff = abs(apogee6 - apogee3)
    print(f"  6-DOF apogee : {apogee6:.6f} m")
    print(f"  3-DOF apogee : {apogee3:.6f} m")
    print(f"  difference   : {diff:.3e} m")
    assert diff < 1e-6, "6-DOF must reduce to 3-DOF when AoA is zero"
    print("  PASS\n")


def demo_disturbance():
    """A stable rocket hit by a gust, showing the oscillation settle."""
    body = Body(mass=0.6, length=1.2, diameter=0.054, cd=0.45,
                cn_alpha=12.0, cp_offset=-0.12)
    f = make_derivative(body, gravity=False)
    q0 = quat_from_axis_angle([0, 1, 0], np.radians(25.0))
    s0 = make_state([0, 0, 2000.0], [0, 0, 180.0], quat=q0)
    ts, states = integrate(f, s0, 0.0, 5.0, 0.0005)
    aoa = np.array([angle_of_attack(s) for s in states])
    # The initial tilt is about body y, so the pitching motion lives in
    # omega[1] - state index 11. Index 10 is omega_x and stays at zero here.
    pitch_rate = np.degrees(states[:, 11])
    return ts, aoa, pitch_rate, body


def plot_all():
    ts, aoa, rate, body = demo_disturbance()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    axes[0].plot(ts, aoa, lw=2)
    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("angle of attack (deg)")
    axes[0].set_title(f"Weathercocking, margin {body.static_margin:.2f} cal")
    axes[0].grid(alpha=0.3)

    axes[1].plot(ts, rate, lw=2, color="tab:orange")
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("pitch rate (deg/s)")
    axes[1].set_title("Body pitch rate")
    axes[1].grid(alpha=0.3)

    # Stability sweep
    margins, finals = [], []
    for cp in np.linspace(-0.25, 0.05, 25):
        b = Body(mass=0.6, length=1.2, diameter=0.054, cd=0.45,
                 cn_alpha=12.0, cp_offset=cp)
        f = make_derivative(b, gravity=False)
        q0 = quat_from_axis_angle([0, 1, 0], np.radians(15.0))
        s0 = make_state([0, 0, 2000.0], [0, 0, 180.0], quat=q0)
        _, st = integrate(f, s0, 0.0, 3.0, 0.001)
        margins.append(b.static_margin)
        finals.append(min(angle_of_attack(st[-1]), 180.0))

    axes[2].plot(margins, finals, "o-", lw=1.8, ms=4, color="tab:green")
    axes[2].axvline(0, ls="--", lw=1.2, color="tab:red", label="neutral")
    axes[2].axhline(15.0, ls=":", lw=1.2, color="gray", label="initial AoA")
    axes[2].set_xlabel("static margin (calibers)")
    axes[2].set_ylabel("AoA after 3 s (deg)")
    axes[2].set_title("Stability sweep")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("step4_sixdof.png", dpi=150)
    print("Saved step4_sixdof.png")


if __name__ == "__main__":
    test_quaternion_norm()
    test_gimbal_lock()
    test_free_rotation_conserves()
    test_weathercock()
    test_matches_3dof()
    plot_all()
