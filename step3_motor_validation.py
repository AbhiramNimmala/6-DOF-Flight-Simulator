"""
Flight dynamics simulator - Step 3
Solid motor, mass depletion, launch rail, apogee detection.
Plus the harness for comparing against OpenRocket.

New here:

  1. RASP (.eng) thrust curve parsing. This is the standard format used by
     thrustcurve.org and read by OpenRocket, so the exact same motor file
     drives both simulators. That matters - if you hand-type a thrust curve
     you are comparing two different rockets and learning nothing.

  2. Mass depletion. The rocket gets lighter as propellant burns, which is
     most of why a rocket accelerates the way it does. Constant mass is a
     surprisingly bad approximation.

  3. Launch rail. Until the rocket clears the rail it is constrained to the
     rail direction and cannot rotate. After that, 3-DOF assumes the vehicle
     points along its velocity vector - the zero angle of attack
     approximation, valid for a stable rocket. Step 4 removes this.

  4. Apogee event, so we get a clean apogee number instead of the maximum
     of a sampled array.

WHAT YOU NEED TO DO
-------------------
Open OpenRocket, build the same rocket described in `REFERENCE_ROCKET`
below, fly it, and put its numbers into `OPENROCKET_RESULTS`. Then re-run.
The comparison table prints itself.

The sample motor here is a plausible H-class curve, NOT a real certified
motor. Download the real .eng from thrustcurve.org and point MOTOR_FILE at
it before you trust any comparison.

Run:  python step3_motor_validation.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os

G0 = 9.80665
R_AIR = 287.05287

IDX_POS = slice(0, 3)
IDX_VEL = slice(3, 6)

MOTOR_FILE = None   # <- set to a real .eng path from thrustcurve.org


# ----------------------------------------------------------------------
# Atmosphere (carried over from step 2)
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
        t = tb
        p = pb * np.exp(-G0 * dh / (R_AIR * tb))
    else:
        t = tb + lapse * dh
        p = pb * (t / tb) ** (-G0 / (lapse * R_AIR))
    rho = p / (R_AIR * t)
    return t, p, rho, np.sqrt(1.4 * R_AIR * t)


# ----------------------------------------------------------------------
# Motor
# ----------------------------------------------------------------------

class Motor:
    """
    Solid motor defined by a thrust-vs-time curve.

    Propellant mass is depleted in proportion to delivered impulse, not
    linearly in time. That is the physically right way to do it: mass flow
    is proportional to thrust, so a regressive curve burns propellant faster
    early on.
    """

    def __init__(self, times, thrusts, prop_mass, total_mass, name="motor"):
        self.times = np.asarray(times, float)
        self.thrusts = np.asarray(thrusts, float)
        self.prop_mass = float(prop_mass)
        self.total_mass = float(total_mass)
        self.name = name
        self.burn_time = float(self.times[-1])

        # Cumulative impulse, used for mass depletion.
        self._cum_impulse = np.concatenate([
            [0.0],
            np.cumsum(0.5 * (self.thrusts[1:] + self.thrusts[:-1])
                      * np.diff(self.times))
        ])
        self.total_impulse = float(self._cum_impulse[-1])

    @classmethod
    def from_eng(cls, path):
        """Parse a RASP .eng file - the format OpenRocket reads."""
        times, thrusts = [], []
        header = None
        with open(path) as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith(";"):
                    continue
                if header is None:
                    header = line.split()
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    times.append(float(parts[0]))
                    thrusts.append(float(parts[1]))
        # header: name diameter length delays prop_mass total_mass manufacturer
        name = header[0]
        prop_mass = float(header[4])
        total_mass = float(header[5])
        if times[0] > 0.0:
            times.insert(0, 0.0)
            thrusts.insert(0, 0.0)
        return cls(times, thrusts, prop_mass, total_mass, name)

    def thrust(self, t):
        if t < 0.0 or t > self.burn_time:
            return 0.0
        return float(np.interp(t, self.times, self.thrusts))

    def propellant_remaining(self, t):
        if t <= 0.0:
            return self.prop_mass
        if t >= self.burn_time:
            return 0.0
        burned = np.interp(t, self.times, self._cum_impulse) / self.total_impulse
        return self.prop_mass * (1.0 - burned)

    def mass(self, t):
        """Current motor mass: casing plus whatever propellant is left."""
        return (self.total_mass - self.prop_mass) + self.propellant_remaining(t)

    def summary(self):
        avg = self.total_impulse / self.burn_time
        # NAR class letter: each letter doubles, A = 2.5 Ns max
        cls_idx = int(np.floor(np.log2(max(self.total_impulse, 1e-9) / 1.25)))
        letter = "ABCDEFGHIJKLMNO"[max(0, min(cls_idx, 14))]
        return (f"{self.name}: {self.total_impulse:.1f} Ns total, "
                f"{avg:.1f} N avg, {self.thrusts.max():.1f} N peak, "
                f"{self.burn_time:.2f} s burn -> class {letter}")


def sample_motor():
    """
    A plausible H-class curve for smoke-testing the pipeline.
    NOT a certified motor. Replace with a real .eng before comparing.
    """
    times = [0.0, 0.05, 0.10, 0.20, 0.40, 0.80, 1.20, 1.60, 1.90, 2.00]
    thrusts = [0.0, 190.0, 210.0, 195.0, 175.0, 150.0, 125.0, 95.0, 40.0, 0.0]
    return Motor(times, thrusts, prop_mass=0.0625, total_mass=0.1240,
                 name="SAMPLE-H128")


# ----------------------------------------------------------------------
# Rocket
# ----------------------------------------------------------------------

class Rocket:
    def __init__(self, dry_mass, diameter, cd, motor):
        self.dry_mass = dry_mass                     # kg, without motor
        self.diameter = diameter                     # m
        self.area = np.pi * (diameter / 2.0) ** 2    # m^2
        self.cd = cd
        self.motor = motor

    def mass(self, t):
        return self.dry_mass + self.motor.mass(t)

    def describe(self):
        return (f"  dry mass (no motor) : {self.dry_mass*1000:.0f} g\n"
                f"  body diameter       : {self.diameter*1000:.1f} mm\n"
                f"  reference area      : {self.area*1e4:.2f} cm^2\n"
                f"  drag coefficient    : {self.cd}\n"
                f"  liftoff mass        : {self.mass(0)*1000:.0f} g\n"
                f"  burnout mass        : {self.mass(1e9)*1000:.0f} g\n"
                f"  {self.motor.summary()}")


REFERENCE_ROCKET = dict(
    dry_mass=0.550,     # 550 g airframe, recovery, electronics - no motor
    diameter=0.054,     # 54 mm body tube
    cd=0.45,            # typical for a clean 4:1 ogive high-power airframe
)


# ----------------------------------------------------------------------
# Dynamics
# ----------------------------------------------------------------------

def make_derivative(rocket, rail_length, rail_angle_deg=0.0, rail_azimuth_deg=0.0):
    """
    3-DOF rocket dynamics.

    Thrust direction:
      - on the rail, along the rail
      - after rail exit, along the velocity vector

    That second assumption is the zero angle of attack approximation. It is
    good for a stable rocket in still air and it is exactly what step 4
    replaces when we add rotation.
    """
    tilt = np.radians(rail_angle_deg)
    az = np.radians(rail_azimuth_deg)
    rail_dir = np.array([np.sin(tilt) * np.cos(az),
                         np.sin(tilt) * np.sin(az),
                         np.cos(tilt)])

    def derivative(t, state):
        pos, vel = state[IDX_POS], state[IDX_VEL]
        mass = rocket.mass(t)
        speed = np.linalg.norm(vel)
        on_rail = np.linalg.norm(pos) < rail_length

        thrust_mag = rocket.motor.thrust(t)
        if on_rail or speed < 1e-6:
            direction = rail_dir
        else:
            direction = vel / speed

        force = thrust_mag * direction
        force = force + np.array([0.0, 0.0, -mass * G0])

        if speed > 1e-9:
            _, _, rho, _ = atmosphere(pos[2])
            drag = 0.5 * rho * speed ** 2 * rocket.cd * rocket.area
            force = force - drag * (vel / speed)

        accel = force / mass
        if on_rail:
            # Constrained to the rail: keep only the along-rail component,
            # and do not let it slide backwards down the rail.
            along = float(np.dot(accel, rail_dir))
            accel = max(along, 0.0) * rail_dir

        deriv = np.zeros_like(state)
        deriv[IDX_POS] = vel
        deriv[IDX_VEL] = accel
        return deriv

    return derivative


def step_rk4(f, t, state, dt):
    k1 = f(t, state)
    k2 = f(t + dt / 2, state + dt / 2 * k1)
    k3 = f(t + dt / 2, state + dt / 2 * k2)
    k4 = f(t + dt, state + dt * k3)
    return state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate(f, state0, t0, t_end, dt, event=None):
    n = int(round((t_end - t0) / dt))
    ts, states = [t0], [np.array(state0, float)]
    state = np.array(state0, float)
    for i in range(n):
        t = t0 + i * dt
        nxt = step_rk4(f, t, state, dt)
        t_next = t0 + (i + 1) * dt
        if event is not None and event(t, state) > 0.0 >= event(t_next, nxt):
            lo, hi = 0.0, dt
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if event(t + mid, step_rk4(f, t, state, mid)) > 0.0:
                    lo = mid
                else:
                    hi = mid
            th = 0.5 * (lo + hi)
            ts.append(t + th)
            states.append(step_rk4(f, t, state, th))
            return np.array(ts), np.array(states)
        state = nxt
        ts.append(t_next)
        states.append(state.copy())
    return np.array(ts), np.array(states)


def apogee_event(t, state):
    """Positive while climbing, crosses zero at apogee."""
    return state[5]


# ----------------------------------------------------------------------
# Flight
# ----------------------------------------------------------------------

def fly(rocket, rail_length=1.83, rail_angle_deg=0.0, dt=0.001):
    f = make_derivative(rocket, rail_length, rail_angle_deg)
    state0 = np.zeros(6)
    ts, states = integrate(f, state0, 0.0, 60.0, dt, event=apogee_event)

    speeds = np.linalg.norm(states[:, IDX_VEL], axis=1)
    accels = np.gradient(speeds, ts)

    # Rail exit: first sample where distance from pad exceeds rail length.
    dists = np.linalg.norm(states[:, IDX_POS], axis=1)
    rail_idx = int(np.argmax(dists >= rail_length))
    rail_exit_v = speeds[rail_idx] if dists[-1] >= rail_length else float("nan")

    burnout_idx = int(np.searchsorted(ts, rocket.motor.burn_time))
    burnout_idx = min(burnout_idx, len(ts) - 1)

    mach = []
    for s in states:
        _, _, _, a = atmosphere(s[2])
        mach.append(np.linalg.norm(s[IDX_VEL]) / a)
    mach = np.array(mach)

    return dict(
        ts=ts, states=states, speeds=speeds,
        apogee=float(states[-1, 2]),
        time_to_apogee=float(ts[-1]),
        max_velocity=float(speeds.max()),
        max_mach=float(mach.max()),
        max_accel_g=float(accels.max() / G0),
        rail_exit_velocity=float(rail_exit_v),
        burnout_velocity=float(speeds[burnout_idx]),
        burnout_altitude=float(states[burnout_idx, 2]),
    )


# ----------------------------------------------------------------------
# THE HANDOFF - fill this in from OpenRocket
# ----------------------------------------------------------------------
# Build the same rocket in OpenRocket:
#   - body diameter and dry mass from REFERENCE_ROCKET above
#   - the same motor file
#   - same rail length and launch angle
#   - no wind, standard sea-level conditions
# Then read these numbers off its flight summary and paste them here.

OPENROCKET_RESULTS = {
    "apogee": None,               # m
    "time_to_apogee": None,       # s
    "max_velocity": None,         # m/s
    "max_mach": None,
    "max_accel_g": None,
    "rail_exit_velocity": None,   # m/s
    "burnout_velocity": None,     # m/s
    "burnout_altitude": None,     # m
}

LABELS = {
    "apogee": ("apogee", "m"),
    "time_to_apogee": ("time to apogee", "s"),
    "max_velocity": ("max velocity", "m/s"),
    "max_mach": ("max Mach", ""),
    "max_accel_g": ("max acceleration", "g"),
    "rail_exit_velocity": ("rail exit velocity", "m/s"),
    "burnout_velocity": ("burnout velocity", "m/s"),
    "burnout_altitude": ("burnout altitude", "m"),
}


def compare(sim):
    print("\nCOMPARISON AGAINST OPENROCKET")
    have_any = any(v is not None for v in OPENROCKET_RESULTS.values())
    if not have_any:
        print("  No OpenRocket numbers entered yet.")
        print("  Build this rocket in OpenRocket, fly it, and paste its")
        print("  results into OPENROCKET_RESULTS at the bottom of this file.")
        print("\n  Simulated values to compare against:")
        for key, (label, unit) in LABELS.items():
            print(f"    {label:<22}: {sim[key]:>10.3f} {unit}")
        return

    print(f"  {'quantity':<22} {'sim':>10} {'OpenRocket':>12} {'diff':>10} {'%':>8}")
    worst = 0.0
    for key, (label, unit) in LABELS.items():
        ref = OPENROCKET_RESULTS.get(key)
        if ref is None:
            print(f"  {label:<22} {sim[key]:>10.3f} {'-':>12} {'-':>10} {'-':>8}")
            continue
        diff = sim[key] - ref
        pct = abs(diff) / abs(ref) * 100.0 if ref else float("nan")
        worst = max(worst, pct)
        print(f"  {label:<22} {sim[key]:>10.3f} {ref:>12.3f} "
              f"{diff:>+10.3f} {pct:>7.2f}%")

    print(f"\n  worst disagreement: {worst:.2f}%")
    if worst < 3.0:
        print("  Under 3% is a genuine match for a 3-DOF model. Step 3 done.")
    elif worst < 10.0:
        print("  3-10% is close but worth chasing. Usual suspects, in order:")
        print("    - Cd is a guess in this model and computed in OpenRocket.")
        print("      Read OpenRocket's actual Cd and put it in REFERENCE_ROCKET.")
        print("    - Different motor file.")
        print("    - OpenRocket may be modelling a launch lug or fin drag you aren't.")
    else:
        print("  Over 10% means something structural is different, not a tuning")
        print("  issue. Check that both are flying the same mass and motor first.")


# ----------------------------------------------------------------------

def plot(sim, rocket):
    ts, states, speeds = sim["ts"], sim["states"], sim["speeds"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    mt = np.linspace(0, rocket.motor.burn_time, 300)
    axes[0].plot(mt, [rocket.motor.thrust(x) for x in mt], lw=2)
    axes[0].fill_between(mt, [rocket.motor.thrust(x) for x in mt], alpha=0.15)
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("thrust (N)")
    axes[0].set_title(rocket.motor.name)
    axes[0].grid(alpha=0.3)

    axes[1].plot(ts, states[:, 2], lw=2)
    axes[1].axvline(rocket.motor.burn_time, ls="--", lw=1.2,
                    color="tab:orange", label="burnout")
    axes[1].scatter([ts[-1]], [states[-1, 2]], zorder=5, color="tab:red",
                    label=f"apogee {sim['apogee']:.0f} m")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("altitude (m)")
    axes[1].set_title("Altitude")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(ts, speeds, lw=2)
    axes[2].axvline(rocket.motor.burn_time, ls="--", lw=1.2, color="tab:orange")
    axes[2].set_xlabel("time (s)")
    axes[2].set_ylabel("speed (m/s)")
    axes[2].set_title(f"Velocity (peak {sim['max_velocity']:.0f} m/s)")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("step3_flight.png", dpi=150)
    print("\nSaved step3_flight.png")


if __name__ == "__main__":
    if MOTOR_FILE and os.path.exists(MOTOR_FILE):
        motor = Motor.from_eng(MOTOR_FILE)
        print(f"Loaded motor from {MOTOR_FILE}")
    else:
        motor = sample_motor()
        print("Using the built-in SAMPLE motor.")
        print("This is a plausible curve, not a certified one. Download the")
        print("real .eng from thrustcurve.org and set MOTOR_FILE before you")
        print("treat any comparison as meaningful.\n")

    rocket = Rocket(motor=motor, **REFERENCE_ROCKET)
    print("ROCKET")
    print(rocket.describe())

    sim = fly(rocket, rail_length=1.83, rail_angle_deg=0.0)

    print("\nSIMULATED FLIGHT")
    for key, (label, unit) in LABELS.items():
        print(f"  {label:<22}: {sim[key]:>10.3f} {unit}")

    # Sanity checks that do not need OpenRocket.
    print("\nSANITY CHECKS")
    twr = motor.thrusts.max() / (rocket.mass(0) * G0)
    print(f"  peak thrust-to-weight : {twr:.1f}   (want > 5 for a safe rail exit)")
    print(f"  rail exit velocity    : {sim['rail_exit_velocity']:.1f} m/s"
          f"   (want > 15 m/s for stability)")
    assert sim["apogee"] > 0, "apogee must be positive"
    assert sim["max_velocity"] > 0, "velocity must be positive"
    assert sim["time_to_apogee"] > motor.burn_time, "apogee must follow burnout"
    print("  basic physics ordering: PASS")

    compare(sim)
    plot(sim, rocket)
