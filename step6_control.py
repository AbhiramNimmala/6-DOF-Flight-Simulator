"""
Flight dynamics simulator - Step 6
Closed-loop attitude control. PID first, then LQR on a linearized model.

The vehicle so far is passively stable: fins drag the nose back into the
wind and that is the only thing holding attitude. Here we add a thrust
vector control gimbal and command the attitude directly.

WHY THIS IS HARDER THAN IT LOOKS
    A fin-stabilized rocket has a restoring moment that fights you. A
    TVC rocket generally does not - it is an inverted pendulum on a
    thrust vector, unstable open-loop, and the controller is the only
    thing keeping it pointed. Turn the loop off and it falls over. TEST 1
    demonstrates exactly that.

WHAT GETS MODELLED HONESTLY
    Real actuators are not ideal. Three limits matter and all three are
    here, because a controller tuned without them will look excellent in
    simulation and fail on hardware:
      - deflection limit      (the gimbal only moves so far)
      - rate limit            (it only moves so fast)
      - first-order lag       (it does not move instantly)
    TEST 4 shows a gain that is stable with an ideal actuator going
    unstable once the lag is included.

Run:  python step6_control.py
"""

import numpy as np
import matplotlib.pyplot as plt

G0 = 9.80665


# ======================================================================
# Plant: pitch-plane rigid body on a gimballed motor
# ======================================================================
# Reduced to the pitch plane, which is where control design actually
# happens. State:
#
#   [0] theta      pitch attitude          (rad)
#   [1] theta_dot  pitch rate              (rad/s)
#   [2] delta      actual gimbal angle     (rad)
#
# The gimbal is a state, not an input, because it lags the command.

class Vehicle:
    def __init__(self, mass=25.0, inertia=12.0, thrust=600.0,
                 l_gimbal=1.10, l_cp=-0.30, cn_alpha=11.0,
                 area=0.0230, rho=1.150, speed=90.0):
        self.mass = mass
        self.inertia = inertia          # pitch inertia about CG, kg m^2
        self.thrust = thrust            # N
        self.l_gimbal = l_gimbal        # CG to gimbal, m (positive = aft)
        self.l_cp = l_cp                # CG to CP, m (negative = CP fwd, unstable)
        self.cn_alpha = cn_alpha
        self.area = area
        self.rho = rho
        self.speed = speed

    @property
    def q_dyn(self):
        return 0.5 * self.rho * self.speed ** 2

    @property
    def control_authority(self):
        """Angular acceleration per radian of gimbal, 1/s^2."""
        return self.thrust * self.l_gimbal / self.inertia

    @property
    def aero_moment(self):
        """
        Angular acceleration per radian of attitude error, 1/s^2.

        Positive means DEstabilizing. With the CP forward of the CG this is
        positive, and the open-loop system has a real positive eigenvalue -
        an inverted pendulum.
        """
        return -(self.q_dyn * self.area * self.cn_alpha * self.l_cp) / self.inertia

    def open_loop_eigenvalues(self):
        """Eigenvalues of the 2-state attitude dynamics, ignoring actuator."""
        A = np.array([[0.0, 1.0], [self.aero_moment, 0.0]])
        return np.linalg.eigvals(A)

    def state_matrices(self, tau):
        """
        Continuous-time A, B for the 3-state plant including actuator lag.

            theta_ddot = aero_moment * theta - control_authority * delta
            delta_dot  = (delta_cmd - delta) / tau
        """
        A = np.array([
            [0.0, 1.0, 0.0],
            [self.aero_moment, 0.0, -self.control_authority],
            [0.0, 0.0, -1.0 / tau],
        ])
        B = np.array([[0.0], [0.0], [1.0 / tau]])
        return A, B


class Actuator:
    """Gimbal with deflection limit, rate limit, and first-order lag."""

    def __init__(self, max_deflection_deg=8.0, max_rate_deg_s=120.0,
                 tau=0.030, ideal=False):
        self.max_deflection = np.radians(max_deflection_deg)
        self.max_rate = np.radians(max_rate_deg_s)
        self.tau = tau
        self.ideal = ideal

    def clamp(self, delta_cmd):
        return float(np.clip(delta_cmd, -self.max_deflection,
                             self.max_deflection))

    def derivative(self, delta, delta_cmd):
        """
        Rate of change of actual deflection toward the commanded one.

        An earlier version modelled the ideal actuator as a 1 microsecond
        time constant. That is a stiff ODE - eigenvalue -1e6 against a 1 ms
        timestep - and RK4 cannot integrate it. The ideal case appeared to
        diverge at every gain, which looked like a physics result and was
        pure numerical blowup. The ideal actuator is now handled by setting
        the deflection directly rather than integrating toward it.
        """
        rate = (self.clamp(delta_cmd) - delta) / self.tau
        return float(np.clip(rate, -self.max_rate, self.max_rate))


# ======================================================================
# Controllers
# ======================================================================

class PID:
    def __init__(self, kp, ki, kd, integral_limit=0.5):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral_limit = integral_limit
        self.reset()

    def reset(self):
        self.integral = 0.0

    def command(self, error, error_rate, dt):
        # Clamping the integral is not optional. Without it, a saturated
        # actuator lets the integral wind up and the vehicle overshoots
        # wildly on recovery.
        self.integral = float(np.clip(self.integral + error * dt,
                                      -self.integral_limit,
                                      self.integral_limit))
        return self.kp * error + self.ki * self.integral + self.kd * error_rate


class LQR:
    """
    Infinite-horizon LQR. Solves the continuous algebraic Riccati equation
    by integrating the Riccati differential equation backwards to steady
    state - no scipy needed, and it makes the mechanism visible.

    LQR is not magic. It trades state error against control effort using
    weights you choose, so the design work moves from tuning three gains by
    feel to picking Q and R by what you actually care about.
    """

    def __init__(self, A, B, Q, R):
        self.K = self._solve(A, B, Q, R)

    @staticmethod
    def _solve(A, B, Q, R, dt=1e-4, max_steps=200000, tol=1e-11):
        P = np.array(Q, dtype=float)
        Rinv = np.linalg.inv(np.atleast_2d(R))
        for i in range(max_steps):
            P_dot = A.T @ P + P @ A - P @ B @ Rinv @ B.T @ P + Q
            step = dt * P_dot
            P = P + step
            # Convergence on the increment, checked every step. Cheap for a
            # 3x3 and it stops as soon as the Riccati equation balances.
            if np.max(np.abs(step)) < tol * dt / 1e-4:
                break
        else:
            raise RuntimeError("Riccati iteration did not converge")
        return Rinv @ B.T @ P

    def command(self, state):
        return float(np.ravel(-(self.K @ state))[0])


# ======================================================================
# Simulation
# ======================================================================

def simulate(vehicle, actuator, controller, theta0_deg=10.0,
             t_end=6.0, dt=0.001, disturbance=None, kind="pid",
             target_deg=0.0):
    theta = np.radians(theta0_deg)
    theta_dot = 0.0
    delta = 0.0
    target = np.radians(target_deg)

    if kind == "pid":
        controller.reset()

    n = int(t_end / dt)
    log = np.zeros((n + 1, 5))   # t, theta, theta_dot, delta, delta_cmd

    for i in range(n + 1):
        t = i * dt
        # Sign convention matters here and it is easy to get backwards.
        # The plant is  theta_ddot = aero*theta - authority*delta, so a
        # POSITIVE gimbal deflection produces a NEGATIVE pitch moment. To
        # correct a positive attitude error we therefore need a positive
        # delta, which means the error fed to the controller is
        # (theta - target), not (target - theta). Getting this backwards
        # turns the loop into positive feedback and the vehicle departs
        # faster than it would with no controller at all.
        error = theta - target

        if kind == "pid":
            delta_cmd = controller.command(error, theta_dot, dt)
        else:
            delta_cmd = controller.command(np.array([theta - target,
                                                     theta_dot, delta]))

        log[i] = [t, theta, theta_dot, delta, delta_cmd]
        if i == n:
            break

        dist = disturbance(t) if disturbance else 0.0

        def deriv(state):
            th, thd, dl = state
            if actuator.ideal:
                dl = actuator.clamp(delta_cmd)     # no lag: tracks exactly
            thdd = (vehicle.aero_moment * th
                    - vehicle.control_authority * dl
                    + dist)
            ddl = 0.0 if actuator.ideal else actuator.derivative(dl, delta_cmd)
            return np.array([thd, thdd, ddl])

        s = np.array([theta, theta_dot, delta])
        k1 = deriv(s)
        k2 = deriv(s + dt / 2 * k1)
        k3 = deriv(s + dt / 2 * k2)
        k4 = deriv(s + dt * k3)
        theta, theta_dot, delta = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if actuator.ideal:
            delta = actuator.clamp(delta_cmd)
        delta = float(np.clip(delta, -actuator.max_deflection,
                              actuator.max_deflection))

    return log


def metrics(log, target_deg=0.0):
    t = log[:, 0]
    theta = np.degrees(log[:, 1])
    target = target_deg
    err = np.abs(theta - target)
    initial = abs(theta[0] - target)

    band = max(0.02 * initial, 0.1)
    settled = float("nan")          # nan = never settled inside the window
    for i in range(len(t) - 1, -1, -1):
        if err[i] > band:
            if i + 1 < len(t):
                settled = t[i + 1]
            break
    else:
        settled = t[0]

    overshoot = 0.0
    if initial > 1e-9:
        beyond = theta - target
        if beyond[0] > 0:
            overshoot = max(0.0, -beyond.min()) / initial * 100
        else:
            overshoot = max(0.0, beyond.max()) / initial * 100

    return dict(settling_time=settled,
                overshoot_pct=overshoot,
                max_gimbal_deg=np.degrees(np.abs(log[:, 3]).max()),
                final_error_deg=err[-1],
                diverged=bool(err[-1] > initial or not np.isfinite(err[-1])))


# ======================================================================
# TESTS
# ======================================================================

def test_open_loop_unstable():
    """The plant must be unstable, otherwise the controller proves nothing."""
    print("TEST 1 - open-loop plant is unstable (inverted pendulum)")
    v = Vehicle()
    eig = v.open_loop_eigenvalues()
    print(f"  aero moment coefficient : {v.aero_moment:+.3f} 1/s^2  "
          f"(positive = destabilizing)")
    print(f"  open-loop eigenvalues   : {eig[0]:+.4f}, {eig[1]:+.4f}")
    print(f"  time to double amplitude: {np.log(2)/max(eig.real):.3f} s")

    # Let it run with the loop open.
    zero = PID(0, 0, 0)
    log = simulate(v, Actuator(), zero, theta0_deg=2.0, t_end=3.0)
    print(f"  2 deg initial tilt -> {np.degrees(log[-1,1]):.1f} deg after 3 s")
    assert max(eig.real) > 0, "plant should have an unstable eigenvalue"
    assert abs(np.degrees(log[-1, 1])) > 20, "should diverge without control"
    print("  PASS - it falls over on its own, so the loop is doing real work\n")


def test_pid_stabilizes():
    print("TEST 2 - PID holds attitude")
    v = Vehicle()
    # kp must exceed aero_moment/control_authority = 0.54 for static
    # stability. kp=6 commands 60 deg of gimbal on a 10 deg error and
    # spends the whole transient saturated against an 8 deg limit.
    pid = PID(kp=1.5, ki=0.9, kd=0.35)
    log = simulate(v, Actuator(), pid, theta0_deg=10.0, t_end=6.0)
    m = metrics(log)
    st = m["settling_time"]
    print(f"  settling time (2%)   : "
          + ("did not settle in window" if np.isnan(st) else f"{st:.3f} s"))
    print(f"  overshoot            : {m['overshoot_pct']:.1f} %")
    print(f"  peak gimbal          : {m['max_gimbal_deg']:.2f} deg "
          f"of 8.00 available")
    print(f"  final error          : {m['final_error_deg']:.4f} deg")
    assert not m["diverged"], "PID should stabilize"
    assert m["final_error_deg"] < 0.5, "should settle near zero"
    print("  PASS\n")
    return log


def test_lqr_stabilizes():
    print("TEST 3 - LQR on the linearized plant")
    v = Vehicle()
    act = Actuator()
    A, B = v.state_matrices(act.tau)

    # Q penalizes state error, R penalizes control effort. Weighting
    # attitude 100x rate says "I care about pointing, not about how
    # smoothly I get there."
    Q = np.diag([400.0, 20.0, 1.0])
    R = np.array([[1.0]])
    lqr = LQR(A, B, Q, R)
    print(f"  gain matrix K        : [{lqr.K[0,0]:7.3f} {lqr.K[0,1]:7.3f} "
          f"{lqr.K[0,2]:7.3f}]")

    closed = A - B @ lqr.K
    eig = np.linalg.eigvals(closed)
    print(f"  closed-loop eigenvalues:")
    for e in eig:
        print(f"      {e:+.4f}")
    assert np.all(np.real(eig) < 0), "closed loop must be stable"

    log = simulate(v, act, lqr, theta0_deg=10.0, t_end=6.0, kind="lqr")
    m = metrics(log)
    st = m["settling_time"]
    print(f"  settling time (2%)   : "
          + ("did not settle in window" if np.isnan(st) else f"{st:.3f} s"))
    print(f"  overshoot            : {m['overshoot_pct']:.1f} %")
    print(f"  peak gimbal          : {m['max_gimbal_deg']:.2f} deg")
    assert not m["diverged"], "LQR should stabilize"
    print("  PASS\n")
    return log


def test_actuator_lag_matters():
    """
    The point of this test: a gain that looks fine with a perfect actuator
    can be unstable with a real one. Neglecting lag is how a controller
    passes simulation and breaks hardware.
    """
    print("TEST 4 - actuator lag changes what is stable")
    v = Vehicle()
    print(f"  {'kp':>6} {'ideal actuator':>18} {'30 ms lag':>18}")

    found_divergence = False
    for kp in [2.0, 10.0, 30.0, 60.0, 120.0, 240.0]:
        ideal = simulate(v, Actuator(ideal=True), PID(kp, 0.0, 1.6),
                         theta0_deg=5.0, t_end=5.0)
        real = simulate(v, Actuator(tau=0.030), PID(kp, 0.0, 1.6),
                        theta0_deg=5.0, t_end=5.0)
        mi, mr = metrics(ideal), metrics(real)
        si = "diverged" if mi["diverged"] else f"{mi['final_error_deg']:.3f} deg"
        sr = "diverged" if mr["diverged"] else f"{mr['final_error_deg']:.3f} deg"
        print(f"  {kp:>6.1f} {si:>18} {sr:>18}")
        if not mi["diverged"] and mr["diverged"]:
            found_divergence = True

    assert found_divergence, ("expected some gain that is stable ideally but "
                              "unstable with lag")
    print("  There is a gain that is stable with a perfect actuator and")
    print("  unstable with a 30 ms one. Tune against the real actuator.\n")


def test_disturbance_rejection():
    print("TEST 5 - wind gust rejection")
    v = Vehicle()

    def gust(t):
        # 0.4 s gust starting at t = 1.5 s, 3 rad/s^2 of angular accel.
        return 3.0 if 1.5 <= t < 1.9 else 0.0

    results = {}
    for name, ctrl, kind in [
        ("PID", PID(kp=1.5, ki=0.9, kd=0.35), "pid"),
        ("LQR", LQR(*v.state_matrices(0.030),
                    np.diag([400.0, 20.0, 1.0]), np.array([[1.0]])), "lqr"),
    ]:
        log = simulate(v, Actuator(), ctrl, theta0_deg=0.0, t_end=6.0,
                       disturbance=gust, kind=kind)
        peak = np.degrees(np.abs(log[:, 1]).max())
        final = abs(np.degrees(log[-1, 1]))
        results[name] = (log, peak, final)
        print(f"  {name}: peak excursion {peak:5.2f} deg, "
              f"recovered to {final:.3f} deg")
        assert final < 0.5, f"{name} should recover from the gust"

    print("  PASS\n")
    return results


# ======================================================================

def plot_all(pid_log, lqr_log, gust_results):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    axes[0].plot(pid_log[:, 0], np.degrees(pid_log[:, 1]), lw=2, label="PID")
    axes[0].plot(lqr_log[:, 0], np.degrees(lqr_log[:, 1]), lw=2, label="LQR")
    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("pitch attitude (deg)")
    axes[0].set_title("10 deg step recovery")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(pid_log[:, 0], np.degrees(pid_log[:, 3]), lw=2, label="PID")
    axes[1].plot(lqr_log[:, 0], np.degrees(lqr_log[:, 3]), lw=2, label="LQR")
    axes[1].axhline(8, ls="--", lw=1.1, color="tab:red", label="gimbal limit")
    axes[1].axhline(-8, ls="--", lw=1.1, color="tab:red")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("gimbal deflection (deg)")
    axes[1].set_title("Control effort")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    for name, (log, _, _) in gust_results.items():
        axes[2].plot(log[:, 0], np.degrees(log[:, 1]), lw=2, label=name)
    axes[2].axvspan(1.5, 1.9, alpha=0.15, color="tab:orange", label="gust")
    axes[2].axhline(0, color="k", lw=0.6)
    axes[2].set_xlabel("time (s)")
    axes[2].set_ylabel("pitch attitude (deg)")
    axes[2].set_title("Gust rejection")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("step6_control.png", dpi=150)
    print("Saved step6_control.png")


if __name__ == "__main__":
    v = Vehicle()
    print("PLANT")
    print(f"  mass {v.mass} kg, pitch inertia {v.inertia} kg m^2")
    print(f"  thrust {v.thrust} N at {v.l_gimbal} m from CG")
    print(f"  control authority : {v.control_authority:.2f} 1/s^2 per rad")
    print(f"  aero moment       : {v.aero_moment:+.2f} 1/s^2 per rad\n")

    test_open_loop_unstable()
    pid_log = test_pid_stabilizes()
    lqr_log = test_lqr_stabilizes()
    test_actuator_lag_matters()
    gust = test_disturbance_rejection()
    plot_all(pid_log, lqr_log, gust)
