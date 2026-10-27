/*
 * Headless OpenRocket driver for validating an external flight simulator.
 *
 * Builds the reference rocket programmatically from the same dimensions and
 * lumped masses that sim/vehicle.py uses, installs a thrust curve read from a
 * RASP .eng file, runs OpenRocket's own simulation engine with no GUI, and
 * prints the results plus OpenRocket's derived drag coefficient.
 *
 * Building the rocket in code rather than writing .ork XML by hand removes a
 * whole class of silent mismatch: there is no format to guess at, and the
 * geometry that OpenRocket flies is the geometry set here.
 *
 * Usage:
 *   java ORValidate <eng-file> <designation> <out.properties> [cdOverride]
 *
 * cdOverride, when given, forces the whole rocket to a fixed drag coefficient.
 * That is what isolates the drag model from everything else: with both tools
 * pinned to the same Cd, any remaining difference is integration, atmosphere,
 * mass or motor handling.
 */

import java.io.File;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.List;

import com.google.inject.Guice;
import com.google.inject.Injector;

import net.sf.openrocket.document.OpenRocketDocument;
import net.sf.openrocket.document.OpenRocketDocumentFactory;
import net.sf.openrocket.document.Simulation;
import net.sf.openrocket.document.StorageOptions;
import net.sf.openrocket.file.openrocket.OpenRocketSaver;
import net.sf.openrocket.file.motor.RASPMotorLoader;
import net.sf.openrocket.motor.MotorConfiguration;
import net.sf.openrocket.motor.ThrustCurveMotor;
import net.sf.openrocket.plugin.PluginModule;
import net.sf.openrocket.rocketcomponent.AxialStage;
import net.sf.openrocket.rocketcomponent.BodyTube;
import net.sf.openrocket.rocketcomponent.FlightConfiguration;
import net.sf.openrocket.rocketcomponent.FlightConfigurationId;
import net.sf.openrocket.rocketcomponent.InnerTube;
import net.sf.openrocket.rocketcomponent.NoseCone;
import net.sf.openrocket.rocketcomponent.Rocket;
import net.sf.openrocket.rocketcomponent.Transition;
import net.sf.openrocket.rocketcomponent.TrapezoidFinSet;
import net.sf.openrocket.rocketcomponent.position.AxialMethod;
import net.sf.openrocket.simulation.FlightData;
import net.sf.openrocket.simulation.FlightDataBranch;
import net.sf.openrocket.simulation.FlightDataType;
import net.sf.openrocket.simulation.SimulationOptions;
import net.sf.openrocket.startup.Application;
import net.sf.openrocket.startup.GuiModule;

public class ORValidate {

    // Geometry and lumped masses, mirroring sim/vehicle.py reference_vehicle().
    static final double DIAMETER        = 0.054;
    static final double RADIUS          = DIAMETER / 2.0;
    static final double NOSE_LENGTH     = 0.216;
    static final double TUBE_LENGTH     = 0.984;
    static final double TOTAL_LENGTH    = NOSE_LENGTH + TUBE_LENGTH;   // 1.200
    static final double FIN_ROOT        = 0.150;
    static final double FIN_TIP         = 0.070;
    static final double FIN_SPAN        = 0.055;
    static final double FIN_SWEEP       = 0.080;
    static final double FIN_STATION     = 1.050;   // root LE from nose tip
    static final int    FIN_COUNT       = 3;
    static final double STRUCTURE_MASS  = 0.550;   // kg, airframe without motor
    static final double STRUCTURE_CG    = 0.800;   // m from nose tip
    static final double MOTOR_STATION   = 1.050;   // m from nose tip, mount front
    static final double RAIL_LENGTH     = 1.83;

    public static void main(String[] args) throws Exception {
        String engPath  = args[0];
        String design   = args[1];
        String outPath  = args[2];
        Double cdForce  = args.length > 3 ? Double.valueOf(args[3]) : null;
        double windReq  = args.length > 4 ? Double.parseDouble(args[4]) : 0.0;

        // Guice wiring. GuiModule is misleadingly named: it supplies the
        // bindings the core needs and does not open a window.
        GuiModule gui = new GuiModule();
        PluginModule plug = new PluginModule();
        Injector injector = Guice.createInjector(gui, plug);
        Application.setInjector(injector);
        gui.startLoader();

        // ---- rocket -------------------------------------------------
        Rocket rocket = new Rocket();
        rocket.setName("54 mm reference airframe");

        AxialStage stage = new AxialStage();
        stage.setName("Sustainer");
        rocket.addChild(stage);

        NoseCone nose = new NoseCone(Transition.Shape.OGIVE, NOSE_LENGTH, RADIUS);
        nose.setName("Nose cone");
        nose.setShapeParameter(1.0);
        nose.setThickness(0.002);
        stage.addChild(nose);

        BodyTube tube = new BodyTube();
        tube.setName("Body tube");
        tube.setLength(TUBE_LENGTH);
        tube.setOuterRadius(RADIUS);
        tube.setThickness(0.001);
        stage.addChild(tube);

        TrapezoidFinSet fins = new TrapezoidFinSet();
        fins.setName("Fin set");
        fins.setFinCount(FIN_COUNT);
        fins.setRootChord(FIN_ROOT);
        fins.setTipChord(FIN_TIP);
        fins.setSweep(FIN_SWEEP);
        fins.setHeight(FIN_SPAN);
        fins.setThickness(0.003);
        fins.setAxialMethod(AxialMethod.TOP);
        fins.setAxialOffset(FIN_STATION - NOSE_LENGTH);  // relative to tube front
        tube.addChild(fins);

        InnerTube mount = new InnerTube();
        mount.setName("Motor mount");
        mount.setLength(0.200);
        mount.setOuterRadius(0.0150);
        mount.setThickness(0.0005);          // inner radius 0.0145 = 29 mm motor
        mount.setMotorMount(true);
        mount.setMotorOverhang(0.003);
        mount.setAxialMethod(AxialMethod.TOP);
        mount.setAxialOffset(MOTOR_STATION - NOSE_LENGTH);
        tube.addChild(mount);

        // ---- mass model ---------------------------------------------
        // OpenRocket estimates component mass from materials; that estimate
        // will not be 550 g. Override the whole stage so both tools fly the
        // same structural mass at the same CG. Motor mass is tracked
        // separately by OpenRocket and is deliberately NOT inside this
        // override, which matches how sim/vehicle.py separates structure
        // from motor.
        stage.setSubcomponentsOverriddenMass(true);
        stage.setSubcomponentsOverriddenCG(true);
        stage.setOverrideMass(STRUCTURE_MASS);
        stage.setMassOverridden(true);
        stage.setOverrideCGX(STRUCTURE_CG);
        stage.setCGOverridden(true);

        if (cdForce != null) {
            stage.setSubcomponentsOverriddenCD(true);
            stage.setOverrideCD(cdForce);
            stage.setCDOverridden(true);
        }

        // ---- motor ---------------------------------------------------
        RASPMotorLoader loader = new RASPMotorLoader();
        List<ThrustCurveMotor.Builder> builders;
        try (FileReader r = new FileReader(engPath)) {
            builders = loader.load(r, new File(engPath).getName());
        }
        ThrustCurveMotor motor = null;
        for (ThrustCurveMotor.Builder b : builders) {
            ThrustCurveMotor m = b.build();
            if (m.getDesignation().contains(design)) { motor = m; break; }
        }
        if (motor == null && !builders.isEmpty()) motor = builders.get(0).build();
        if (motor == null) throw new IllegalStateException("no motor in " + engPath);

        FlightConfigurationId fcid = new FlightConfigurationId();
        rocket.createFlightConfiguration(fcid);
        rocket.setSelectedConfiguration(fcid);

        MotorConfiguration mc = new MotorConfiguration(mount, fcid);
        mc.setMotor(motor);
        mc.setEjectionDelay(1000.0);     // no deployment before apogee
        mount.setMotorConfig(mc, fcid);

        FlightConfiguration config = rocket.getFlightConfiguration(fcid);
        config.setAllStages();

        // ---- simulation ----------------------------------------------
        OpenRocketDocument doc = OpenRocketDocumentFactory.createDocumentFromRocket(rocket);
        Simulation sim = new Simulation(doc, rocket);
        sim.setFlightConfigurationId(fcid);

        SimulationOptions opt = sim.getOptions();
        opt.setLaunchRodLength(RAIL_LENGTH);
        opt.setLaunchRodAngle(0.0);
        opt.setLaunchRodDirection(0.0);
        opt.setWindSpeedAverage(windReq);
        opt.setWindSpeedDeviation(0.0);
        opt.setWindTurbulenceIntensity(0.0);
        opt.setLaunchAltitude(0.0);
        opt.setISAAtmosphere(true);
        opt.setTimeStep(0.001);

        sim.simulate();
        FlightData d = sim.getSimulatedData();

        // ---- results --------------------------------------------------
        StringBuilder sb = new StringBuilder();
        sb.append("motor_designation=").append(motor.getDesignation()).append("\n");
        sb.append("motor_total_impulse=").append(motor.getTotalImpulseEstimate()).append("\n");
        sb.append("motor_burn_time=").append(motor.getBurnTimeEstimate()).append("\n");
        sb.append("motor_launch_mass=").append(motor.getLaunchMass()).append("\n");
        sb.append("motor_burnout_mass=").append(motor.getBurnoutMass()).append("\n");
        sb.append("apogee=").append(d.getMaxAltitude()).append("\n");
        sb.append("max_velocity=").append(d.getMaxVelocity()).append("\n");
        sb.append("max_mach=").append(d.getMaxMachNumber()).append("\n");
        sb.append("max_acceleration=").append(d.getMaxAcceleration()).append("\n");
        sb.append("time_to_apogee=").append(d.getTimeToApogee()).append("\n");
        sb.append("rail_exit_velocity=").append(d.getLaunchRodVelocity()).append("\n");

        // Read the launch conditions back out of the options object that was
        // actually simulated, rather than trusting that the setters took.
        sb.append("opt_wind_mean=").append(opt.getWindSpeedAverage()).append("\n");
        sb.append("opt_wind_dev=").append(opt.getWindSpeedDeviation()).append("\n");
        sb.append("opt_rod_length=").append(opt.getLaunchRodLength()).append("\n");
        sb.append("opt_rod_angle=").append(opt.getLaunchRodAngle()).append("\n");
        sb.append("opt_isa=").append(opt.isISAAtmosphere()).append("\n");
        sb.append("opt_timestep=").append(opt.getTimeStep()).append("\n");
        sb.append("opt_geodetic=").append(opt.getGeodeticComputation()).append("\n");

        FlightDataBranch br = d.getBranch(0);
        List<Double> alt = br.get(FlightDataType.TYPE_ALTITUDE);
        List<Double> vel = br.get(FlightDataType.TYPE_VELOCITY_TOTAL);
        List<Double> tt  = br.get(FlightDataType.TYPE_TIME);
        // State at burnout, so the comparison table covers the powered phase
        // as well as apogee.
        if (tt != null && alt != null && vel != null) {
            double bt = motor.getBurnTimeEstimate();
            int best = -1; double bestErr = Double.MAX_VALUE;
            for (int i = 0; i < tt.size(); i++) {
                Double t = tt.get(i);
                if (t == null || t.isNaN()) continue;
                double e = Math.abs(t - bt);
                if (e < bestErr) { bestErr = e; best = i; }
            }
            if (best >= 0) {
                if (best < vel.size()) sb.append("burnout_velocity=").append(vel.get(best)).append("\n");
                if (best < alt.size()) sb.append("burnout_altitude=").append(alt.get(best)).append("\n");
            }
        }
        List<Double> mach = br.get(FlightDataType.TYPE_MACH_NUMBER);
        List<Double> cd   = br.get(FlightDataType.TYPE_DRAG_COEFF);
        List<Double> mass = br.get(FlightDataType.TYPE_MASS);
        List<Double> time = br.get(FlightDataType.TYPE_TIME);

        if (mass != null && !mass.isEmpty()) {
            sb.append("launch_mass=").append(mass.get(0)).append("\n");
            double mn = Double.MAX_VALUE;
            for (Double v : mass) if (v != null && !v.isNaN()) mn = Math.min(mn, v);
            sb.append("burnout_mass=").append(mn).append("\n");
        }

        // OpenRocket's own drag coefficient, sampled where it is comparable.
        // The subsonic figure is the one to set against this project's 0.45
        // input: reading Cd at peak Mach would fold in the transonic rise and
        // compare unlike with unlike.
        if (mach != null && cd != null) {
            double sumLow = 0; int nLow = 0;
            double cdAtPeak = Double.NaN; double peak = -1;
            for (int i = 0; i < Math.min(mach.size(), cd.size()); i++) {
                Double m = mach.get(i), c = cd.get(i);
                if (m == null || c == null || m.isNaN() || c.isNaN()) continue;
                if (m > 0.05 && m < 0.40) { sumLow += c; nLow++; }
                if (m > peak) { peak = m; cdAtPeak = c; }
            }
            if (nLow > 0) sb.append("cd_subsonic_mean=").append(sumLow / nLow).append("\n");
            sb.append("cd_at_peak_mach=").append(cdAtPeak).append("\n");
            sb.append("cd_samples_subsonic=").append(nLow).append("\n");

            // Where OpenRocket's drag actually comes from. Knowing the split
            // says which part of the airframe a Cd assumption got wrong.
            FlightDataType[] parts = {
                FlightDataType.TYPE_FRICTION_DRAG_COEFF,
                FlightDataType.TYPE_PRESSURE_DRAG_COEFF,
                FlightDataType.TYPE_BASE_DRAG_COEFF,
                FlightDataType.TYPE_AXIAL_DRAG_COEFF };
            String[] names = { "friction", "pressure", "base", "axial" };
            for (int p = 0; p < parts.length; p++) {
                List<Double> series = br.get(parts[p]);
                if (series == null) continue;
                double s = 0; int n = 0;
                for (int i = 0; i < Math.min(mach.size(), series.size()); i++) {
                    Double m = mach.get(i), c = series.get(i);
                    if (m == null || c == null || m.isNaN() || c.isNaN()) continue;
                    if (m > 0.05 && m < 0.40) { s += c; n++; }
                }
                if (n > 0) sb.append("cd_").append(names[p])
                            .append("_subsonic=").append(s / n).append("\n");
            }

            StringBuilder curve = new StringBuilder();
            for (int i = 0; i < Math.min(mach.size(), cd.size()); i += 25) {
                Double m = mach.get(i), c = cd.get(i);
                Double t = (time != null && i < time.size()) ? time.get(i) : null;
                if (m == null || c == null || m.isNaN() || c.isNaN()) continue;
                curve.append(String.format("%.4f:%.4f:%.4f;",
                        t == null ? 0.0 : t, m, c));
            }
            sb.append("cd_curve=").append(curve).append("\n");
        }

        if (d.getWarningSet() != null && !d.getWarningSet().isEmpty()) {
            sb.append("warnings=").append(d.getWarningSet().toString()
                    .replace("\n", " ")).append("\n");
        }

        try (PrintWriter pw = new PrintWriter(outPath)) { pw.print(sb); }

        // Save the design so the same rocket can be opened and re-flown in the
        // OpenRocket GUI. A validation nobody can reproduce is an assertion.
        doc.addSimulation(sim);
        String orkPath = outPath.replaceAll("\\.properties$", "") + ".ork";
        StorageOptions so = new StorageOptions();
        so.setSaveSimulationData(false);
        try (java.io.FileOutputStream fos = new java.io.FileOutputStream(orkPath)) {
            new OpenRocketSaver().save(fos, doc, so,
                    new net.sf.openrocket.logging.WarningSet(),
                    new net.sf.openrocket.logging.ErrorSet());
        }
        System.out.print(sb);
        System.out.println("saved_ork=" + orkPath);
    }
}
