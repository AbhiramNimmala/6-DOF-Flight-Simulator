"""
Flight dynamics simulator - Step 2
Atmosphere + drag, and a ground-impact event with interpolation.

Three things get added here:

  1. US Standard Atmosphere 1976, up to 86 km. Density falls off with
     altitude, which is what makes a high-altitude trajectory differ from
     a textbook parabola.

  2. Quadratic drag, F = -1/2 * rho * v^2 * Cd * A * v_hat. Note what this
     buys you: terminal velocity is now an emergent property of the model
     rather than a number you typed in. That is the test.

  3. Event detection. Step 1 overshot the ground by part of a timestep and
     reported a range of 244.80 m against an exact 244.73 m. That error had
     nothing to do with RK4 - it was the last step landing past the event.
     Fixed here by detecting the sign change and interpolating back to the
     crossing.

Run:  python step2_atmosphere_drag.py
"""

import numpy as np
import matplotlib.pyplot as plt

G0 = 9.80665      # m/s^2
R_AIR = 287.05287  # J/(kg K), specific gas constant for dry air

IDX_POS = slice(0, 3)
IDX_VEL = slice(3, 6)


def make_state(pos, vel):
    return np.concatenate([np.asarray(pos, float), np.asarray(vel, float)])


# ----------------------------------------------------------------------
# US Standard Atmosphere 1976
# ----------------------------------------------------------------------
# Piecewise model. Each layer has a base altitude, base temperature, base
# pressure, and a lapse rate (dT/dh). Within a layer, temperature is linear
# in altitude and pressure follows from hydrostatic equilibrium:
#
#   lapse != 0:  P = Pb * (T/Tb) ^ (-g / (L * R))
#   lapse == 0:  P = Pb * exp(-g * (h - hb) / (R * T))
#
# Density then comes from the ideal gas law, rho = P / (R * T).
#
# Layer table: (base altitude m, base temp K, base pressure Pa, lapse K/m)
# ----------------------------------------------------------------------

_LAYERS = [
    (0.0,     288.15,   101325.0,    -0.0065),
    (11000.0, 216.65,   22632.06,     0.0),
    (20000.0, 216.65,   5474.889,     0.001),
    (32000.0, 228.65,   868.0187,     0.0028),
    (47000.0, 270.65,   110.9063,     0.0),
    (51000.0, 270.65,   66.93887,    -0.0028),
    (71000.0, 214.65,   3.956420,    -0.002),
    (84852.0, 186.946,  0.3734,       0.0),
]


def atmosphere(h):
    """
    US Standard Atmosphere 1976.

    Returns (temperature K, pressure Pa, density kg/m^3, speed of sound m/s)
    for geometric altitude h in meters. Clamped at the ends: below sea level
    returns sea-level values, above 86 km returns the top-layer values.
    """
    h = float(np.clip(h, 0.0, 86000.0))

    for i in range(len(_LAYERS) - 1, -1, -1):
        hb, tb, pb, lapse = _LAYERS[i]
        if h >= hb:
            break

    dh = h - hb
    if lapse == 0.0:
        t = tb
        p = pb * np.exp(-G0 * dh / (R_AIR * tb))
    else:
        t = tb + lapse * dh
        p = pb * (t / tb) ** (-G0 / (lapse * R_AIR))

    rho = p / (R_AIR * t)
    a = np.sqrt(1.4 * R_AIR * t)
    return t, p, rho, a


# ----------------------------------------------------------------------
# Vehicle
# ----------------------------------------------------------------------

class Vehicle:
    """A point mass with a drag coefficient and a reference area."""

    def __init__(self, mass, cd, area):
        self.mass = mass      # kg
        self.cd = cd          # dimensionless
        self.area = area      # m^2, frontal reference area

    def terminal_velocity(self, rho):
        """Closed-form terminal velocity, for checking the simulation."""
        return np.sqrt(2.0 * self.mass * G0 / (rho * self.cd * self.area))


# ----------------------------------------------------------------------
# Derivative
# ----------------------------------------------------------------------

def make_derivative(vehicle, with_drag=True, rho_fixed=None):
    """
    Build a derivative function closed over a vehicle.

    `rho_fixed` pins density to a constant instead of reading the
    atmosphere. That exists so drag can be tested on its own: with density
    varying, the closed-form terminal velocity is a moving target and you
    cannot tell a drag bug from an atmosphere bug. Test one thing at a time.
    """

    def derivative(t, state):
        vel = state[IDX_VEL]
        accel = np.array([0.0, 0.0, -G0])

        if with_drag:
            speed = np.linalg.norm(vel)
            if speed > 1e-9:
                if rho_fixed is not None:
                    rho = rho_fixed
                else:
                    _, _, rho, _ = atmosphere(state[2])
                # Drag magnitude, opposing the velocity direction.
                drag = 0.5 * rho * speed ** 2 * vehicle.cd * vehicle.area
                accel = accel - (drag / vehicle.mass) * (vel / speed)

        deriv = np.zeros_like(state)
        deriv[IDX_POS] = vel
        deriv[IDX_VEL] = accel
        return deriv

    return derivative


# ----------------------------------------------------------------------
# Integrator (unchanged from step 1, plus event handling)
# ----------------------------------------------------------------------

def step_rk4(f, t, state, dt):
    k1 = f(t, state)
    k2 = f(t + dt / 2.0, state + dt / 2.0 * k1)
    k3 = f(t + dt / 2.0, state + dt / 2.0 * k2)
    k4 = f(t + dt, state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate(f, state0, t0, t_end, dt, stepper=step_rk4, event=None):
    """
    March forward, with optional event detection.

    `event` is a scalar function of (t, state). When it changes sign between
    two steps, we bisect to find the crossing, append that interpolated
    point, and stop. That is what fixes the overshoot from step 1.

    Integer step counting is kept from step 1 - accumulating `t += dt`
    corrupts any convergence study.
    """
    n_steps = int(round((t_end - t0) / dt))
    ts = [t0]
    states = [np.array(state0, float)]
    state = np.array(state0, float)

    for i in range(n_steps):
        t = t0 + i * dt
        nxt = stepper(f, t, state, dt)
        t_next = t0 + (i + 1) * dt

        if event is not None:
            e_now, e_next = event(t, state), event(t_next, nxt)
            if e_now > 0.0 >= e_next:
                t_hit, s_hit = _bisect_event(f, stepper, event, t, state, dt)
                ts.append(t_hit)
                states.append(s_hit)
                return np.array(ts), np.array(states)

        state = nxt
        ts.append(t_next)
        states.append(state.copy())

    return np.array(ts), np.array(states)


def _bisect_event(f, stepper, event, t0, s0, dt, tol=1e-12, iters=60):
    """Bisect within one step to land on the event crossing."""
    lo, hi = 0.0, dt
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        s_mid = stepper(f, t0, s0, mid)
        if event(t0 + mid, s_mid) > 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    t_hit = 0.5 * (lo + hi)
    return t0 + t_hit, stepper(f, t0, s0, t_hit)


def ground_event(t, state):
    """Positive while above ground, crosses zero at impact."""
    return state[2]


# ----------------------------------------------------------------------
# TEST 1 - atmosphere against published values
# ----------------------------------------------------------------------

def test_atmosphere():
    print("TEST 1 - US Standard Atmosphere 1976 against published values")
    # (altitude m, expected T K, expected rho kg/m^3)
    refs = [
        (0,     288.15, 1.2250),
        (5000,  255.68, 0.73643),
        (11000, 216.65, 0.36392),
        (20000, 216.65, 0.088035),
        (32000, 228.65, 0.013225),
    ]
    print(f"  {'alt (m)':>9} {'T sim':>9} {'T ref':>9} "
          f"{'rho sim':>11} {'rho ref':>11} {'err %':>8}")
    worst = 0.0
    for h, t_ref, rho_ref in refs:
        t, _, rho, _ = atmosphere(h)
        err = abs(rho - rho_ref) / rho_ref * 100.0
        worst = max(worst, err)
        print(f"  {h:>9} {t:>9.2f} {t_ref:>9.2f} "
              f"{rho:>11.5f} {rho_ref:>11.5f} {err:>7.3f}%")
    assert worst < 0.5, "atmosphere density should match tables to well under 1%"
    print(f"  worst density error: {worst:.3f}%  PASS\n")


# ----------------------------------------------------------------------
# TEST 2 - terminal velocity emerges from the model
# ----------------------------------------------------------------------
# This is the real test of the drag implementation. We never tell the
# simulation what terminal velocity is. We drop a sphere from high up and
# check that it converges to the closed-form value on its own.

def test_terminal_velocity():
    print("TEST 2 - terminal velocity emerges (not coded in)")
    # 10 cm sphere, 4 kg, Cd 0.47.
    r = 0.05
    sphere = Vehicle(mass=4.0, cd=0.47, area=np.pi * r ** 2)
    _, _, rho_sl, _ = atmosphere(0.0)
    v_exact = sphere.terminal_velocity(rho_sl)

    # 2a. Constant density. This is the clean check of the drag term: with
    #     rho pinned, the closed-form terminal velocity is a fixed target.
    f_iso = make_derivative(sphere, rho_fixed=rho_sl)
    state0 = make_state([0, 0, 0], [0, 0, 0])
    ts, states = integrate(f_iso, state0, 0.0, 200.0, 0.005)
    v_iso = abs(states[-1, 5])

    print("  2a. constant density (isolates drag from atmosphere)")
    print(f"      closed form      : {v_exact:.4f} m/s")
    print(f"      simulated        : {v_iso:.4f} m/s")
    print(f"      difference       : {abs(v_iso - v_exact):.2e} m/s")
    assert abs(v_iso - v_exact) < 1e-3, "drag term is wrong"
    print("      PASS")

    # 2b. Real atmosphere, dropped from 2 km. It does NOT reach sea-level
    #     terminal velocity, and that is correct rather than a bug: air is
    #     thinner up there, so the local terminal velocity starts higher and
    #     falls as the sphere descends. 2 km is not enough distance to
    #     converge. Showing this is the point.
    f_atm = make_derivative(sphere)
    ts2, st2 = integrate(f_atm, make_state([0, 0, 2000.0], [0, 0, 0]),
                         0.0, 200.0, 0.005, event=ground_event)
    _, _, rho_2km, _ = atmosphere(2000.0)

    print("  2b. real atmosphere, dropped from 2 km")
    print(f"      v_term at 2 km   : {sphere.terminal_velocity(rho_2km):.2f} m/s")
    print(f"      v_term at 0 km   : {v_exact:.2f} m/s")
    print(f"      impact speed     : {abs(st2[-1, 5]):.2f} m/s")
    print(f"      fall time        : {ts2[-1]:.2f} s")
    print("      Still accelerating at impact - 2 km is too short to converge.")
    print("      Expected, not a bug.\n")

    return ts, states, v_exact


# ----------------------------------------------------------------------
# TEST 3 - the event fixes step 1's overshoot
# ----------------------------------------------------------------------

def test_event_accuracy():
    print("TEST 3 - ground event interpolation (fixes step 1 overshoot)")
    pos0, vel0 = [0.0, 0.0, 0.0], [30.0, 0.0, 40.0]
    ballistic = Vehicle(mass=1.0, cd=0.0, area=0.0)   # drag disabled
    f = make_derivative(ballistic, with_drag=False)

    t_exact = 2.0 * vel0[2] / G0
    range_exact = vel0[0] * t_exact

    ts, states = integrate(f, make_state(pos0, vel0), 0.0, 20.0, 0.01,
                           event=ground_event)

    print(f"  exact flight time  : {t_exact:.9f} s")
    print(f"  sim   flight time  : {ts[-1]:.9f} s")
    print(f"  exact range        : {range_exact:.6f} m")
    print(f"  sim   range        : {states[-1, 0]:.6f} m")
    print(f"  range error        : {abs(states[-1, 0] - range_exact):.3e} m")
    print("  (step 1 reported 244.800 against 244.732 - that gap is now gone)")
    assert abs(states[-1, 0] - range_exact) < 1e-6
    print("  PASS\n")


# ----------------------------------------------------------------------
# TEST 4 - drag actually changes the trajectory
# ----------------------------------------------------------------------

def compare_drag_vs_vacuum():
    print("TEST 4 - drag vs vacuum, same launch")
    # A rough cannonball: 0.08 m radius, 10 kg
    r = 0.08
    ball = Vehicle(mass=10.0, cd=0.47, area=np.pi * r ** 2)
    pos0, vel0 = [0, 0, 0], [200.0, 0.0, 200.0]   # ~283 m/s at 45 deg

    f_drag = make_derivative(ball, with_drag=True)
    f_vac = make_derivative(ball, with_drag=False)

    ts_d, s_d = integrate(f_drag, make_state(pos0, vel0), 0, 120, 0.002,
                          event=ground_event)
    ts_v, s_v = integrate(f_vac, make_state(pos0, vel0), 0, 120, 0.002,
                          event=ground_event)

    print(f"  vacuum : range {s_v[-1,0]:9.1f} m   apogee {s_v[:,2].max():8.1f} m"
          f"   time {ts_v[-1]:6.2f} s")
    print(f"  drag   : range {s_d[-1,0]:9.1f} m   apogee {s_d[:,2].max():8.1f} m"
          f"   time {ts_d[-1]:6.2f} s")
    print(f"  drag removes {100*(1-s_d[-1,0]/s_v[-1,0]):.1f}% of the range\n")
    return (ts_v, s_v), (ts_d, s_d)


# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------

def plot_all(fall, vac, drag, v_term):
    ts_f, s_f = fall
    (ts_v, s_v), (ts_d, s_d) = vac, drag

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # 1. atmosphere profile
    hs = np.linspace(0, 84000, 400)
    rhos = [atmosphere(h)[2] for h in hs]
    axes[0].semilogx(rhos, hs / 1000.0, lw=2)
    axes[0].set_xlabel("density (kg/m$^3$)")
    axes[0].set_ylabel("altitude (km)")
    axes[0].set_title("US Standard Atmosphere 1976")
    axes[0].grid(alpha=0.3, which="both")

    # 2. terminal velocity convergence
    axes[1].plot(ts_f, np.abs(s_f[:, 5]), lw=2, label="simulated")
    axes[1].axhline(v_term, ls="--", lw=1.4, color="tab:orange",
                    label=f"closed form {v_term:.1f} m/s")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("speed (m/s)")
    axes[1].set_title("Terminal velocity emerges")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # 3. drag vs vacuum
    axes[2].plot(s_v[:, 0] / 1000, s_v[:, 2] / 1000, "--", lw=1.6, label="vacuum")
    axes[2].plot(s_d[:, 0] / 1000, s_d[:, 2] / 1000, lw=2, label="with drag")
    axes[2].set_xlabel("downrange (km)")
    axes[2].set_ylabel("altitude (km)")
    axes[2].set_title("Drag vs vacuum, same launch")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("step2_validation.png", dpi=150)
    print("Saved step2_validation.png")


if __name__ == "__main__":
    test_atmosphere()
    ts_f, s_f, v_term = test_terminal_velocity()
    test_event_accuracy()
    vac, drag = compare_drag_vs_vacuum()
    plot_all((ts_f, s_f), vac, drag, v_term)
