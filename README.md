# 6-DOF Rocket Flight Dynamics Simulator

A six-degree-of-freedom rocket flight simulator built from scratch in Python, in six
validated layers from a bare numerical integrator up to a closed-loop flight controller.

**Author:** Abhiram Nimmala · solo build · Python / NumPy / Matplotlib · ~2,600 lines

## The idea

Each layer is validated against a known answer — a closed-form solution or published
data — before the next layer is built on top of it. Every step ships with a test suite
that checks physics, not just that the code runs. A bug in a lower layer produces output
that still looks like plausible physics higher up, so it never gets caught; the staged
validation is what stops that.

## Run it

Each step is self-contained and runnable on its own:

```bash
pip install numpy matplotlib
python step1_integrator.py        # RK4 integrator, convergence study
python step2_atmosphere_drag.py   # US Standard Atmosphere 1976 + drag
python step3_motor_validation.py  # Solid motor, powered flight to apogee
python step4_sixdof.py            # Quaternion 6-DOF rigid-body dynamics
python step5_aero.py              # Barrowman aerodynamics, static margin
python step6_control.py           # PID vs LQR pitch control
```

Each writes its own validation figure (`stepN_*.png`).

## The six layers

| Step | Adds | Validated against |
|------|------|-------------------|
| 1 | RK4 integrator, 3-DOF point mass | Closed-form projectile (error ~5e-12 m) |
| 2 | US Standard Atmosphere 1976, drag, event detection | Published density tables (0.043%) |
| 3 | Solid motor, mass depletion, launch rail | Total-impulse conservation |
| 4 | 6-DOF quaternion attitude, rigid-body rotation | 3-DOF apogee regression; momentum conservation |
| 5 | Barrowman aerodynamics, Mach-dependent drag | Static-margin envelope |
| 6 | Closed-loop PID + LQR, actuator limits | Step response, gust rejection |

## Selected results

- RK4 convergence ratios 13.5–15.7 against the theoretical 16 (fourth order, confirmed).
- Atmosphere density within 0.043% of published tables.
- Quaternion norm holds to 2e-16 over 30,000 integration steps.
- LQR settles a 10° step in 1.13 s with zero overshoot; PID takes 3.37 s with 30%.
- LQR holds a wind gust to 0.55° peak vs the PID's 2.27°.

## Design notes

- **Generic integrator.** State is a flat NumPy array; RK4 never knows what the numbers
  mean, so going from 3-DOF to 13-element 6-DOF required no change to the integrator.
- **Quaternions, not Euler angles** — gimbal lock is demonstrated numerically, not assumed.
- **Integer step counts**, never `t += dt` accumulation.

See `flight-sim-handoff.md` for the full engineering log, including the eight bugs that
each looked like physics and weren't.

## Limitations

Barrowman assumes small angle of attack, subsonic, slender body, thin fins. No wind model
in the trajectory integrator. Pitch-plane control only. Validated against closed-form
solutions and published tables — not yet against OpenRocket or real flight data.
