"""
Flight dynamics simulator - Step 1
RK4 integrator + 3-DOF point mass under gravity.

The only goal here is proving the integrator is correct. No drag, no atmosphere,
no rotation. Those come later and they all sit on top of this, so if this is
wrong everything above it is wrong in ways that look like physics bugs.

Run:  python step1_integrator.py
"""

import numpy as np
import matplotlib.pyplot as plt

G0 = 9.80665  # m/s^2, standard gravity


# ----------------------------------------------------------------------
# State vector
# ----------------------------------------------------------------------
# We pack the state as a flat numpy array so the integrator never has to
# know what the numbers mean. That separation is the whole point: swap in
# a 6-DOF derivative function later and RK4 doesn't change at all.
#
#   state = [x, y, z, vx, vy, vz]
#
# z is up. Position in meters, velocity in m/s.
# ----------------------------------------------------------------------

IDX_POS = slice(0, 3)
IDX_VEL = slice(3, 6)


def make_state(pos, vel):
    """Build a state vector from position and velocity triples."""
    return np.concatenate([np.asarray(pos, dtype=float),
                           np.asarray(vel, dtype=float)])


# ----------------------------------------------------------------------
# Derivative function
# ----------------------------------------------------------------------

def derivative_gravity_only(t, state):
    """
    dState/dt for a point mass under constant gravity.

    Returns the time derivative of the state vector. Position's derivative
    is velocity; velocity's derivative is acceleration.

    Note this takes `t` even though it doesn't use it. Later derivative
    functions will need it (thrust curves are functions of time), and
    keeping the signature fixed means RK4 never changes.
    """
    deriv = np.zeros_like(state)
    deriv[IDX_POS] = state[IDX_VEL]
    deriv[IDX_VEL] = np.array([0.0, 0.0, -G0])
    return deriv


# ----------------------------------------------------------------------
# Integrators
# ----------------------------------------------------------------------

def step_euler(f, t, state, dt):
    """Forward Euler. First-order. Here only as a comparison baseline."""
    return state + dt * f(t, state)


def step_rk4(f, t, state, dt):
    """
    Classical fourth-order Runge-Kutta.

    Four derivative evaluations per step: one at the start, two at the
    midpoint, one at the end, combined with weights 1/6, 1/3, 1/3, 1/6.
    Costs 4x what Euler does per step and buys roughly three extra orders
    of accuracy, which is a trade worth making almost always.
    """
    k1 = f(t, state)
    k2 = f(t + dt / 2.0, state + dt / 2.0 * k1)
    k3 = f(t + dt / 2.0, state + dt / 2.0 * k2)
    k4 = f(t + dt, state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate(f, state0, t0, t_end, dt, stepper=step_rk4, stop=None):
    """
    March the state forward and return (times, states).

    `stop` is an optional predicate taking (t, state) and returning True
    when the run should end early - used here to stop at ground impact.

    NOTE on the step count. The obvious way to write this loop is
    `while t < t_end: t += dt`, and it is wrong. Repeatedly adding a float
    accumulates rounding error, so the final time lands slightly off t_end
    and by a different amount for every dt. That makes a convergence study
    meaningless: you end up comparing the solution at slightly different
    times, and that mismatch swamps the integrator's actual error.

    Fix: compute the step count as an integer up front, and derive each
    time from the step index rather than accumulating.
    """
    n_steps = int(round((t_end - t0) / dt))
    ts = [t0]
    states = [np.array(state0, dtype=float)]
    state = np.array(state0, dtype=float)

    for i in range(n_steps):
        t = t0 + i * dt
        state = stepper(f, t, state, dt)
        ts.append(t0 + (i + 1) * dt)
        states.append(state.copy())
        if stop is not None and stop(ts[-1], state):
            break

    return np.array(ts), np.array(states)


# ----------------------------------------------------------------------
# Analytic truth
# ----------------------------------------------------------------------
# The reason we start with gravity only is that the exact answer is known.
# You cannot verify an integrator against a case you can't solve by hand.

def analytic_projectile(t, pos0, vel0):
    """Closed-form position under constant gravity, no drag."""
    t = np.asarray(t).reshape(-1, 1)
    pos0 = np.asarray(pos0, dtype=float)
    vel0 = np.asarray(vel0, dtype=float)
    accel = np.array([0.0, 0.0, -G0])
    return pos0 + vel0 * t + 0.5 * accel * t ** 2


# ----------------------------------------------------------------------
# Test 1 - does it match the closed form?
# ----------------------------------------------------------------------

def test_matches_analytic():
    pos0 = [0.0, 0.0, 0.0]
    vel0 = [30.0, 0.0, 40.0]       # 50 m/s at ~53 degrees
    dt = 0.01
    t_end = 2.0 * vel0[2] / G0     # exact time to return to z=0

    ts, states = integrate(derivative_gravity_only, make_state(pos0, vel0),
                           0.0, t_end, dt)
    exact = analytic_projectile(ts, pos0, vel0)
    err = np.linalg.norm(states[:, IDX_POS] - exact, axis=1)

    print("TEST 1 - RK4 against closed-form projectile")
    print(f"  steps taken          : {len(ts)}")
    print(f"  max position error   : {err.max():.3e} m")
    print(f"  final position error : {err[-1]:.3e} m")

    # Analytic range and apogee, for a second independent check.
    range_exact = vel0[0] * t_end
    apogee_exact = vel0[2] ** 2 / (2 * G0)
    range_sim = states[-1, 0]
    apogee_sim = states[:, 2].max()

    print(f"  range   exact/sim    : {range_exact:.4f} / {range_sim:.4f} m")
    print(f"  apogee  exact/sim    : {apogee_exact:.4f} / {apogee_sim:.4f} m")

    assert err.max() < 1e-6, "RK4 should be near-exact on a quadratic trajectory"
    print("  PASS\n")
    return ts, states, exact


# ----------------------------------------------------------------------
# Test 2 - is it actually fourth order?
# ----------------------------------------------------------------------
# This is the test that catches a subtly wrong RK4. An implementation with
# the wrong coefficients will still look "close enough" on a single run,
# but its error will shrink at the wrong rate when you refine dt.
#
# Halving dt should cut RK4 error by ~2^4 = 16x, and Euler error by ~2x.
#
# Gravity-only motion is quadratic in t, which RK4 integrates exactly, so
# for the order study we use a case with a non-trivial exact solution:
# a linear spring, whose solution is a cosine.

def derivative_spring(t, state):
    """1-D harmonic oscillator packed into the same 6-element state."""
    omega2 = 4.0  # omega = 2 rad/s
    deriv = np.zeros_like(state)
    deriv[IDX_POS] = state[IDX_VEL]
    deriv[IDX_VEL] = np.array([-omega2 * state[0], 0.0, 0.0])
    return deriv


def test_convergence_order():
    omega = 2.0
    t_end = 5.0
    state0 = make_state([1.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    exact_final_x = np.cos(omega * t_end)

    print("TEST 2 - convergence order (halving dt)")
    print(f"  {'dt':>10} {'RK4 error':>14} {'ratio':>8} "
          f"{'Euler error':>14} {'ratio':>8}")

    prev_rk4 = prev_eul = None
    for dt in [0.1, 0.05, 0.025, 0.0125, 0.00625]:
        _, s_rk4 = integrate(derivative_spring, state0, 0.0, t_end, dt, step_rk4)
        _, s_eul = integrate(derivative_spring, state0, 0.0, t_end, dt, step_euler)
        e_rk4 = abs(s_rk4[-1, 0] - exact_final_x)
        e_eul = abs(s_eul[-1, 0] - exact_final_x)

        r_rk4 = f"{prev_rk4 / e_rk4:7.1f}x" if prev_rk4 else "      -"
        r_eul = f"{prev_eul / e_eul:7.1f}x" if prev_eul else "      -"
        print(f"  {dt:>10.5f} {e_rk4:>14.3e} {r_rk4:>8} "
              f"{e_eul:>14.3e} {r_eul:>8}")

        prev_rk4, prev_eul = e_rk4, e_eul

    print("  Expect RK4 ratio near 16x and Euler near 2x.")
    print("  If RK4 is coming out near 2x or 4x, your coefficients are wrong.\n")


# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------

def plot_trajectory(ts, states, exact):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(states[:, 0], states[:, 2], lw=2, label="RK4")
    ax1.plot(exact[:, 0], exact[:, 2], "--", lw=1.4, label="Analytic")
    ax1.set_xlabel("downrange (m)")
    ax1.set_ylabel("altitude (m)")
    ax1.set_title("Trajectory")
    ax1.legend()
    ax1.grid(alpha=0.3)

    err = np.linalg.norm(states[:, IDX_POS] - exact, axis=1)
    ax2.semilogy(ts, np.maximum(err, 1e-18), lw=1.6)
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("position error (m)")
    ax2.set_title("Error vs analytic solution")
    ax2.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig("step1_validation.png", dpi=150)
    print("Saved step1_validation.png")


if __name__ == "__main__":
    ts, states, exact = test_matches_analytic()
    test_convergence_order()
    plot_trajectory(ts, states, exact)
