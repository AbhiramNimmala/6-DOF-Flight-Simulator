"""
Flight dynamics simulator - Step 5
Real aerodynamics. Barrowman normal-force coefficients from actual geometry,
Mach-dependent drag with a transonic rise, and a centre of pressure that
moves with flight condition.

In step 4, cn_alpha and cp_offset were numbers you typed in. That is fine
for demonstrating that stable rockets weathercock, and useless for
predicting whether YOUR rocket is stable. Here they are computed from nose
cone, body tube and fin dimensions.

WHAT CHANGES PHYSICALLY

  1. The centre of pressure moves. Barrowman gives CP as a function of
     geometry, and the fin contribution grows with Mach, so the CP shifts
     aft through the transonic region. A rocket can be stable at Mach 0.3
     and marginal at Mach 0.9.

  2. The CG moves too, because propellant burns off from the aft end. Most
     rockets are least stable right at liftoff and get more stable as they
     burn - the opposite of what people assume.

  3. Drag is no longer one number. Below Mach 0.8 it is roughly constant.
     Through the transonic region it can more than double as shock waves
     form on the body. That drag rise is why a marginal motor never reaches
     the altitude a constant-Cd model predicts.

THE BARROWMAN ASSUMPTIONS, stated honestly
  - small angle of attack (under about 10 degrees)
  - subsonic, though we extend it with a Mach correction
  - slender body, axisymmetric, thin fins
  Outside those, the numbers are indicative rather than trustworthy.

Run:  python step5_aero.py
"""

import numpy as np
import matplotlib.pyplot as plt

G0 = 9.80665
R_AIR = 287.05287


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
# Geometry
# ----------------------------------------------------------------------
# All axial stations measured from the NOSE TIP, positive aft. This is the
# rocketry convention and it is worth matching, because every reference and
# every OpenRocket readout uses it. The body frame from step 4 has +z toward
# the nose, so converting between them flips sign - handled in one place.

class NoseCone:
    """Nose cone. `shape` is one of: conical, ogive, parabolic, vonkarman."""

    _CP_FRACTION = {          # CP station as a fraction of nose length
        "conical": 2.0 / 3.0,
        "ogive": 0.466,
        "parabolic": 0.5,
        "vonkarman": 0.5,
    }

    def __init__(self, length, base_diameter, shape="ogive"):
        self.length = length
        self.diameter = base_diameter
        self.shape = shape

    def cn_alpha(self):
        """
        Barrowman: every nose cone contributes exactly 2 per radian,
        regardless of shape. Shape only moves the CP, not the force.
        """
        return 2.0

    def cp_station(self):
        return self._CP_FRACTION[self.shape] * self.length


class BodyTube:
    """
    A cylindrical section.

    A pure cylinder contributes no normal force in Barrowman theory. Real
    body tubes do produce a little at angle of attack, which is why a long
    rocket is slightly less stable than Barrowman predicts, but the standard
    method ignores it and so do we.
    """

    def __init__(self, length, diameter, station):
        self.length = length
        self.diameter = diameter
        self.station = station          # forward end, from nose tip

    def cn_alpha(self):
        return 0.0

    def cp_station(self):
        return self.station + self.length / 2.0


class FinSet:
    """
    Trapezoidal fins. This is where nearly all the stability comes from.

    root   - root chord, at the body
    tip    - tip chord
    span   - one fin's exposed semi-span
    sweep  - axial distance from root leading edge to tip leading edge
    """

    def __init__(self, count, root, tip, span, sweep, body_radius, station):
        self.count = count
        self.root = root
        self.tip = tip
        self.span = span
        self.sweep = sweep
        self.body_radius = body_radius
        self.station = station          # root leading edge, from nose tip

        # Mid-chord sweep length, used in the Barrowman denominator.
        self.mid_chord = np.sqrt(
            span ** 2 + (sweep + tip / 2.0 - root / 2.0) ** 2)

    def cn_alpha(self, mach=0.0):
        """
        Barrowman fin normal-force slope, with body interference.

        The Prandtl-Glauert factor 1/sqrt(1-M^2) is applied below Mach 0.8.
        It diverges at Mach 1, which is a known limitation of the correction
        rather than real physics, so we cap it and hold it through the
        transonic region.
        """
        r, s = self.body_radius, self.span
        # Fin interference factor: the body carries some of the fin's load.
        kfb = 1.0 + r / (s + r)

        base = (4.0 * self.count * (s / (2 * r)) ** 2) / (
            1.0 + np.sqrt(1.0 + (2.0 * self.mid_chord / (self.root + self.tip)) ** 2))

        cn = kfb * base
        # Prandtl-Glauert compressibility correction, 1/sqrt(1-M^2).
        # It diverges at Mach 1 and is already unreliable well before that.
        # Capping at M 0.7 is the honest choice: past there this is a held
        # constant, not a prediction. An earlier version capped at M 0.8 and
        # produced a 1.67x boost that pushed the CP far enough aft to report
        # a 3.6-caliber margin at burnout - a modelling artifact that looked
        # like a real stability result.
        m_eff = min(abs(mach), 0.7)
        cn /= np.sqrt(1.0 - m_eff ** 2)
        return cn

    def cp_station(self):
        xr, ct, cr = self.sweep, self.tip, self.root
        term1 = xr * (cr + 2 * ct) / (3.0 * (cr + ct))
        term2 = (1.0 / 6.0) * (cr + ct - cr * ct / (cr + ct))
        return self.station + term1 + term2


# ----------------------------------------------------------------------
# Drag
# ----------------------------------------------------------------------

def drag_coefficient(mach, cd_subsonic=0.45):
    """
    Cd as a function of Mach.

    A simple but honest model of the transonic drag rise: flat below M 0.8,
    climbing steeply to a peak just above M 1, then falling slowly. Real
    rockets show a rise of roughly 2-3x through this region, and a model
    without it will overpredict apogee for anything that goes supersonic.

    This is a shape fit, not a first-principles calculation. It is the right
    level of fidelity for trajectory work and the wrong level for design.
    """
    m = abs(mach)
    if m < 0.8:
        return cd_subsonic

    # The two branches below are written so they MEET at Mach 1.1. An
    # earlier version had the rise end at 2.06x and the decay start at
    # 2.50x, leaving a visible step in the curve. A discontinuous force
    # model will make an adaptive integrator thrash and can inject
    # nonphysical transients, so continuity is worth checking - the plot
    # is what caught it.
    if m < 1.1:
        frac = (m - 0.8) / 0.3
        return cd_subsonic * (1.0 + 1.5 * np.sin(frac * np.pi / 2.0))

    # Supersonic decay from the 2.5x peak toward about 1.4x.
    return cd_subsonic * (1.4 + 1.1 * np.exp(-(m - 1.1) * 1.2))


# ----------------------------------------------------------------------
# Assembled rocket
# ----------------------------------------------------------------------

class Rocket:
    def __init__(self, components, dry_mass, dry_cg_station,
                 prop_mass=0.0, prop_cg_station=None, cd_subsonic=0.45):
        self.components = components
        self.dry_mass = dry_mass
        self.dry_cg_station = dry_cg_station
        self.prop_mass = prop_mass
        self.prop_cg_station = (prop_cg_station if prop_cg_station is not None
                                else dry_cg_station)
        self.cd_subsonic = cd_subsonic

        self.nose = next(c for c in components if isinstance(c, NoseCone))
        self.fins = next((c for c in components if isinstance(c, FinSet)), None)
        self.diameter = self.nose.diameter
        self.area = np.pi * (self.diameter / 2.0) ** 2

    # -- aerodynamics ---------------------------------------------------

    def cn_alpha(self, mach=0.0):
        total = 0.0
        for c in self.components:
            total += c.cn_alpha(mach) if isinstance(c, FinSet) else c.cn_alpha()
        return total

    def cp_station(self, mach=0.0):
        """Normal-force-weighted average of component CPs."""
        num = den = 0.0
        for c in self.components:
            cn = c.cn_alpha(mach) if isinstance(c, FinSet) else c.cn_alpha()
            num += cn * c.cp_station()
            den += cn
        return num / den if den > 1e-12 else self.nose.cp_station()

    def cd(self, mach):
        return drag_coefficient(mach, self.cd_subsonic)

    # -- mass properties ------------------------------------------------

    def mass(self, prop_fraction=1.0):
        return self.dry_mass + self.prop_mass * prop_fraction

    def cg_station(self, prop_fraction=1.0):
        """CG moves forward as aft-mounted propellant burns off."""
        mp = self.prop_mass * prop_fraction
        total = self.dry_mass + mp
        return (self.dry_mass * self.dry_cg_station
                + mp * self.prop_cg_station) / total

    # -- stability ------------------------------------------------------

    def static_margin(self, mach=0.0, prop_fraction=1.0):
        """Calibers between CG and CP. Positive is stable."""
        return (self.cp_station(mach) - self.cg_station(prop_fraction)) / self.diameter

    def report(self, mach=0.0):
        lines = [f"  {'component':<14} {'CN_alpha':>10} {'CP (cm)':>10}"]
        for c in self.components:
            cn = c.cn_alpha(mach) if isinstance(c, FinSet) else c.cn_alpha()
            lines.append(f"  {type(c).__name__:<14} {cn:>10.3f} "
                         f"{c.cp_station()*100:>10.2f}")
        lines.append(f"  {'TOTAL':<14} {self.cn_alpha(mach):>10.3f} "
                     f"{self.cp_station(mach)*100:>10.2f}")
        return "\n".join(lines)


def reference_rocket():
    """A 54 mm high-power airframe, dimensionally plausible."""
    d = 0.054
    nose = NoseCone(length=0.216, base_diameter=d, shape="ogive")
    tube = BodyTube(length=0.984, diameter=d, station=0.216)
    fins = FinSet(count=3, root=0.150, tip=0.070, span=0.055, sweep=0.080,
                  body_radius=d / 2, station=1.050)
    # CG at 0.80 m of a 1.20 m airframe: two thirds aft, which is where a
    # heavy motor and aft electronics actually put it. An earlier version
    # used 0.62 m and 75 mm fins, giving nearly 6 calibers of margin -
    # dimensionally possible and nothing anyone would fly. The 55 mm span
    # here is what the stability test drove it to.
    return Rocket(components=[nose, tube, fins],
                  dry_mass=0.550, dry_cg_station=0.800,
                  prop_mass=0.0625, prop_cg_station=1.050,
                  cd_subsonic=0.45)


# ======================================================================
# TESTS
# ======================================================================

def test_barrowman_known_values():
    """Check the pieces against values you can verify by hand."""
    print("TEST 1 - Barrowman components against hand calculations")
    rocket = reference_rocket()

    nose_cn = rocket.nose.cn_alpha()
    print(f"  nose CN_alpha        : {nose_cn:.4f}  (Barrowman says exactly 2.0)")
    assert abs(nose_cn - 2.0) < 1e-12

    tube_cn = next(c for c in rocket.components
                   if isinstance(c, BodyTube)).cn_alpha()
    print(f"  body tube CN_alpha   : {tube_cn:.4f}  (a cylinder contributes 0)")
    assert tube_cn == 0.0

    ogive_cp = rocket.nose.cp_station()
    print(f"  ogive CP             : {ogive_cp*100:.2f} cm  "
          f"(0.466 x {rocket.nose.length*100:.1f} cm)")
    assert abs(ogive_cp - 0.466 * rocket.nose.length) < 1e-12

    conical = NoseCone(0.216, 0.054, "conical")
    print(f"  conical CP           : {conical.cp_station()*100:.2f} cm  "
          f"(2/3 of length, further aft than an ogive)")
    assert conical.cp_station() > ogive_cp
    print("  PASS\n")


def test_cp_moves_with_mach():
    """The CP is not a constant. Fin effectiveness rises with Mach."""
    print("TEST 2 - centre of pressure moves with Mach")
    rocket = reference_rocket()
    print(f"  {'Mach':>6} {'CN_alpha':>10} {'CP (cm)':>10} {'margin (cal)':>14}")
    prev_cp = None
    for mach in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        cp = rocket.cp_station(mach)
        print(f"  {mach:>6.2f} {rocket.cn_alpha(mach):>10.3f} {cp*100:>10.2f} "
              f"{rocket.static_margin(mach, 1.0):>14.3f}")
        if prev_cp is not None and mach <= 0.8:
            assert cp >= prev_cp - 1e-9, "CP should move aft as Mach rises"
        prev_cp = cp
    print("  CP shifts aft as fins become more effective, so the rocket gets")
    print("  MORE stable with speed. That is the subsonic behaviour.\n")


def test_cg_moves_with_burn():
    """Propellant burns from the aft end, so the CG moves forward."""
    print("TEST 3 - CG moves as propellant burns")
    rocket = reference_rocket()
    print(f"  {'prop left':>10} {'mass (g)':>10} {'CG (cm)':>10} "
          f"{'margin (cal)':>14}")
    m_start = rocket.static_margin(0.3, 1.0)
    m_end = rocket.static_margin(0.3, 0.0)
    for frac in [1.0, 0.75, 0.5, 0.25, 0.0]:
        print(f"  {frac*100:>9.0f}% {rocket.mass(frac)*1000:>10.1f} "
              f"{rocket.cg_station(frac)*100:>10.2f} "
              f"{rocket.static_margin(0.3, frac):>14.3f}")
    print(f"  margin at liftoff {m_start:.3f} cal -> burnout {m_end:.3f} cal")
    assert m_end > m_start, "burning aft propellant should move CG forward"
    print("  The rocket is LEAST stable at liftoff, which is the opposite of")
    print("  what most people assume, and why rail exit velocity matters.\n")


def test_transonic_drag_rise():
    """Drag is not one number."""
    print("TEST 4 - transonic drag rise")
    print(f"  {'Mach':>6} {'Cd':>8} {'x subsonic':>12}")
    base = drag_coefficient(0.3)
    peak = 0.0
    for mach in [0.3, 0.7, 0.85, 0.95, 1.05, 1.2, 1.5, 2.0]:
        cd = drag_coefficient(mach)
        peak = max(peak, cd)
        print(f"  {mach:>6.2f} {cd:>8.3f} {cd/base:>11.2f}x")
    assert peak > 2.0 * base, "transonic rise should be substantial"
    print("  A constant-Cd model will overpredict apogee for anything that")
    print("  goes near Mach 1. This is the single biggest error in naive sims.\n")


def test_stability_criterion():
    """Check the reference rocket against the 1-2 caliber rule of thumb."""
    print("TEST 5 - reference rocket stability across the flight envelope")
    rocket = reference_rocket()
    worst, best = 1e9, -1e9
    print(f"  {'condition':<28} {'margin (cal)':>14}")
    for label, mach, frac in [
        ("liftoff, wet, M 0.05", 0.05, 1.0),
        ("mid-burn, M 0.4", 0.4, 0.5),
        ("burnout, dry, M 0.85", 0.85, 0.0),
        ("coast, M 0.3", 0.3, 0.0),
    ]:
        m = rocket.static_margin(mach, frac)
        worst = min(worst, m)
        best = max(best, m)
        if m < 1.0:
            flag = "   <- marginal, under 1 cal"
        elif m > 3.0:
            flag = "   <- overstable, weathercocks badly"
        else:
            flag = ""
        print(f"  {label:<28} {m:>14.3f}{flag}")
    print(f"  envelope: {worst:.2f} to {best:.2f} calibers")

    # Both ends matter. An earlier version of this test only checked
    # `worst > 1.0`, which a 5.8-caliber rocket passes without complaint -
    # and 5.8 calibers is a rocket that turns hard into any crosswind and
    # throws away altitude. A one-sided assertion is barely an assertion.
    assert worst > 1.0, "under 1 caliber is marginal"
    assert best < 3.0, "over 3 calibers weathercocks and loses altitude"
    print("  Rule of thumb is 1-2 calibers, and this stays inside it across")
    print("  the whole envelope. Under 1 is marginal; over about 3 and the")
    print("  rocket turns hard into wind and throws away altitude.\n")


# ----------------------------------------------------------------------

def plot_all():
    rocket = reference_rocket()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    machs = np.linspace(0.01, 2.0, 300)

    axes[0].plot(machs, [drag_coefficient(m) for m in machs], lw=2)
    axes[0].axvline(1.0, ls="--", lw=1.1, color="tab:red", label="Mach 1")
    axes[0].set_xlabel("Mach")
    axes[0].set_ylabel("$C_d$")
    axes[0].set_title("Transonic drag rise")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    sub = np.linspace(0.01, 0.95, 200)
    axes[1].plot(sub, [rocket.cp_station(m) * 100 for m in sub], lw=2,
                 label="CP")
    for frac, style in [(1.0, "--"), (0.0, ":")]:
        axes[1].axhline(rocket.cg_station(frac) * 100, ls=style, lw=1.4,
                        label=f"CG ({'wet' if frac else 'dry'})")
    axes[1].set_xlabel("Mach")
    axes[1].set_ylabel("station from nose (cm)")
    axes[1].set_title("CP and CG")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    axes[1].invert_yaxis()

    for frac, lbl in [(1.0, "wet (liftoff)"), (0.0, "dry (burnout)")]:
        axes[2].plot(sub, [rocket.static_margin(m, frac) for m in sub],
                     lw=2, label=lbl)
    axes[2].axhspan(1.0, 2.0, alpha=0.12, color="tab:green",
                    label="1-2 cal target")
    axes[2].axhline(1.0, ls="--", lw=1.1, color="tab:red")
    axes[2].set_xlabel("Mach")
    axes[2].set_ylabel("static margin (calibers)")
    axes[2].set_title("Stability envelope")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("step5_aero.png", dpi=150)
    print("Saved step5_aero.png")


if __name__ == "__main__":
    rocket = reference_rocket()
    print("REFERENCE ROCKET - Barrowman breakdown at Mach 0.3")
    print(rocket.report(0.3))
    print(f"  CG wet {rocket.cg_station(1.0)*100:.2f} cm, "
          f"dry {rocket.cg_station(0.0)*100:.2f} cm")
    print(f"  static margin: {rocket.static_margin(0.3, 1.0):.2f} cal wet, "
          f"{rocket.static_margin(0.3, 0.0):.2f} cal dry\n")

    test_barrowman_known_values()
    test_cp_moves_with_mach()
    test_cg_moves_with_burn()
    test_transonic_drag_rise()
    test_stability_criterion()
    plot_all()
