# Flight Dynamics Simulator — Handoff

**Abhiram Nimmala** · solo build · Python + NumPy + Matplotlib
Upload this file plus the step files at the start of a new chat.

---

## What this is

A 6-DOF rocket flight dynamics simulator, built from scratch in stages. It's an
Elite-tier item off my extracurricular board — chosen because it needs no faculty
advisor, no team, and no lab, and because it's the clearest single artifact showing I
can do both halves of a CS and aerospace degree.

The guiding rule of the whole build: **validate each layer against something with a
known answer before adding the next one.** Every step has a test suite that checks
physics, not just that the code runs.

---

## Progress: 6 of 7 steps done

| Step | What | Status |
|------|------|--------|
| 1 | RK4 integrator + 3-DOF point mass | done, `step1_integrator.py` |
| 2 | US Standard Atmosphere 1976 + drag + event detection | done, `step2_atmosphere_drag.py` |
| 3 | Solid motor, mass depletion, launch rail | done, `step3_motor_validation.py` |
| 4 | 6-DOF: quaternions, rigid body rotation, aero torque | done, `step4_sixdof.py` |
| 5 | Barrowman aerodynamics, Mach-dependent Cd, moving CP | done, `step5_aero.py` |
| 6 | Closed-loop control: PID + LQR, actuator limits | done, `step6_control.py` |
| 7 | Writeup, validation plots, 3D visualization | **NOT STARTED** |

**Also outstanding:** step 3 has an OpenRocket comparison harness with empty slots.
I haven't run OpenRocket yet. `OPENROCKET_RESULTS` in that file is still all `None`.

---

## What each file contains

**`step1_integrator.py`** — RK4 and forward Euler, a flat state vector so the
integrator never knows what the numbers mean, and a convergence study. Verifies RK4
error drops ~16x per timestep halving and Euler ~2x.

**`step2_atmosphere_drag.py`** — Piecewise US Standard Atmosphere to 86 km, quadratic
drag, and event detection with bisection so the sim lands exactly on ground impact
instead of overshooting by part of a step.

**`step3_motor_validation.py`** — RASP `.eng` thrust curve parser (the format
thrustcurve.org publishes and OpenRocket reads), impulse-proportional mass depletion,
launch rail constraint, apogee event. Contains the OpenRocket comparison harness.

**`step4_sixdof.py`** — 13-element state: position, velocity, attitude quaternion,
body angular velocity. Quaternion algebra, Euler rigid-body equations, aerodynamic
torque from the CP–CG offset, pitch damping.

**`step5_aero.py`** — Barrowman normal-force coefficients computed from nose cone,
body tube and fin geometry. Prandtl–Glauert compressibility correction, transonic
drag rise, CG that moves as propellant burns.

**`step6_control.py`** — Pitch-plane plant with a gimballed motor, deliberately
unstable open-loop. PID and LQR controllers, with the Riccati equation solved by
backward integration (no scipy). Actuator model with deflection limit, rate limit and
first-order lag.

---

## Design decisions worth not re-litigating

- **State is a flat NumPy array**, and the integrator is generic. Swapping a 6-element
  3-DOF state for a 13-element 6-DOF one required no change to RK4 at all.
- **Quaternions, not Euler angles.** Step 4 demonstrates the singularity numerically
  rather than asserting it.
- **Integer step counts**, never `t += dt` accumulation.
- **Everything gets validated against a closed form or published data** before it's
  trusted.

---

## Bugs found, and why each mattered

These are the interesting part of the project and the material for the writeup.

1. **Float accumulation in the time loop (step 1).** `while t < t_end: t += dt` makes
   the final time land differently for each dt, so a convergence study compares
   solutions at *different times* and that mismatch swamps the integrator error. The
   smallest timestep looked worse than a larger one. Fix: integer step count.

2. **Terminal velocity test failure (step 2).** A sphere dropped from 2 km hit at
   127 m/s against a closed-form 131.7. Looked like a drag bug; wasn't. Air is thinner
   at altitude, so local terminal velocity starts higher and falls as the sphere
   descends — and 2 km isn't enough distance to converge. Fix: split into a
   constant-density test that isolates drag, plus a real-atmosphere case that
   demonstrates non-convergence deliberately.

3. **Wrong Euler degeneracy pairing (step 4).** My first gimbal-lock demo used
   (yaw +d, roll −d) vs (yaw −d, roll +d), which does *not* collapse. With ZYX
   ordering, pitching 90° maps body x onto −z, so it's yaw *minus* roll that
   degenerates. The specific degeneracy depends on the Euler convention.

4. **One-sided assertion hid a bad design (step 5).** The stability test asserted
   `margin > 1.0`, which a 5.9-caliber rocket passes without complaint — and 5.9
   calibers is unflyable. Making it two-sided caught it immediately.

5. **Prandtl–Glauert artifact (step 5).** Capping the compressibility correction at
   Mach 0.8 gave a 1.67x fin boost that reported a 3.6-caliber margin at burnout — a
   modeling artifact that looked like a stability result. Capped at 0.7 now, with the
   code saying plainly that past there it's a held constant, not a prediction.

6. **Discontinuous drag curve (step 5).** The transonic rise ended at 2.06x and the
   supersonic decay started at 2.50x, leaving a visible step at Mach 1.2. Caught by
   looking at the plot, not by any assertion. Discontinuous force models make
   integrators thrash.

7. **Sign error in the PID loop (step 6).** Defined error as `target − theta` when the
   plant needs positive gimbal to correct positive pitch, making the loop positive
   feedback — the vehicle departed faster than with no controller at all.

8. **Stiff ODE masquerading as physics (step 6).** Modeled the "ideal" actuator as a
   1 µs time constant. Eigenvalue −1e6 against a 1 ms timestep is unintegrable by
   RK4, so the ideal case appeared to diverge at every gain. Looked like a control
   result; was pure numerical blowup. Fixed by setting deflection directly rather
   than integrating toward it.

---

## Results so far

- RK4 matches the closed-form projectile to ~5e−12 m; convergence ratios 13.5, 14.9,
  15.5, 15.7 against a theoretical 16.
- Atmosphere density matches published tables to within 0.043%.
- Terminal velocity emerges from the drag model to 3e−11 m/s of closed form.
- 6-DOF reproduces the 3-DOF apogee to exactly zero difference at zero angle of
  attack — the regression test proving rotation didn't break translation.
- Quaternion norm holds to 2e−16 over 30,000 steps.
- Torque-free rotation conserves angular momentum and energy to ~1e−14.
- Gimbal-lock sensitivity: 2.0 at zero pitch, 3,141,593 at ninety degrees.
- Transonic drag rise peaks at 2.5x subsonic at Mach 1.1.
- Reference rocket stability envelope: 1.44 to 2.85 calibers, inside the 1–2 rule of
  thumb, least stable at liftoff.
- LQR settles a 10° step in 1.13 s with zero overshoot; PID takes 3.37 s with 30%
  overshoot. LQR holds a wind gust to 0.55° peak against PID's 2.27°.

---

## Step 7 — what's left

1. **Run the OpenRocket comparison.** Build the reference rocket in OpenRocket, fly
   it, paste results into `OPENROCKET_RESULTS` in step 3. Download a real `.eng` from
   thrustcurve.org first — the built-in motor is a plausible curve I generated, not a
   certified one. Also read OpenRocket's computed Cd rather than using my 0.45 guess.
2. **Writeup with validation plots**, not screenshots. The convergence study, the
   atmosphere comparison, the OpenRocket agreement, the stability envelope.
3. **Honest limitations section** — small-angle Barrowman assumptions, no wind model,
   no roll dynamics, pitch-plane-only control design.
4. **3D visualization.** The easy part given my React and Skia background.

---

## Known limitations to state in the writeup

- Barrowman assumes small angle of attack (<10°), subsonic, slender body, thin fins.
- Prandtl–Glauert is held constant above Mach 0.7 rather than extrapolated.
- The transonic drag model is a shape fit, not a first-principles calculation.
- Control design is pitch-plane only; no roll coupling.
- No wind or gust model in the trajectory sim (only as a disturbance in step 6).
- The simulator has never been checked against real flight data, only against
  OpenRocket and closed-form solutions.

---

## How to start the new chat

Upload this file plus the six step files, and say something like:

> Here's my flight dynamics simulator. Handoff doc explains where I am — step 6 of 7
> done. I want to work on [step 7 / the OpenRocket validation / the writeup].
