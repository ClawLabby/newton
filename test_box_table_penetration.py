"""Minimal repro: box penetrating kinematic table under downward force.

Creates a simple scene with a kinematic table and a dynamic box positioned above it.
Applies increasingly strong downward forces to the box each trial. Reports the
minimum force at which the box penetrates through the table.

This reproduces a bug observed in DexSuite Kuka Allegro training where the
manipulation object falls through the kinematic table at ~2.4 m/s impact velocity.

Usage:
    python test_box_table_penetration.py [--device cuda:0] [--num_substeps 2] [--visualize]
"""

import argparse
import math

import warp as wp

import newton

TABLE_CENTER_Z = 0.235  # Table center height (matches DexSuite)
TABLE_HALF_THICKNESS = 0.02  # Table is 0.04m thick
TABLE_SURFACE_Z = TABLE_CENTER_Z + TABLE_HALF_THICKNESS  # 0.255
TABLE_HALF_X = 0.4
TABLE_HALF_Y = 0.75

BOX_HALF_X = 0.025  # Box 0.05 x 0.1 x 0.1 (matches DexSuite cube)
BOX_HALF_Y = 0.05
BOX_HALF_Z = 0.05
BOX_MASS = 0.2  # kg

# Box starts resting on table surface, tilted slightly (one corner down)
BOX_START_Z = TABLE_SURFACE_Z + BOX_HALF_Z + 0.001  # Tiny gap above table
PENETRATION_Z = TABLE_CENTER_Z - TABLE_HALF_THICKNESS - 0.01  # Below table bottom


def build_scene(num_worlds: int = 1, device: str = "cuda:0"):
    """Build a minimal scene: ground plane, kinematic table, dynamic box."""
    builder = newton.ModelBuilder()

    for w in range(num_worlds):
        builder.begin_world()

        # Ground plane (static)
        builder.add_shape_plane(plane=(0.0, 0.0, 1.0, 0.0))

        # Kinematic table
        table_body = builder.add_body(
            xform=wp.transformf(
                p=(0.0, 0.0, TABLE_CENTER_Z),
                q=(0.0, 0.0, 0.0, 1.0),
            ),
            mass=0.0,
            is_kinematic=True,
            label="table",
        )
        builder.add_shape_box(
            body=table_body,
            hx=TABLE_HALF_X,
            hy=TABLE_HALF_Y,
            hz=TABLE_HALF_THICKNESS,
            label="table_shape",
        )

        # Dynamic box — tilted 30° around X axis (one corner down) to match
        # the kind of orientation a falling manipulated object might have
        angle = math.radians(30)
        qx = math.sin(angle / 2)
        qw = math.cos(angle / 2)

        box_body = builder.add_body(
            xform=wp.transformf(
                p=(0.0, 0.0, BOX_START_Z + 0.3),  # Start 30cm above table
                q=(qx, 0.0, 0.0, qw),  # Tilted 30° around X
            ),
            mass=BOX_MASS,
            label="box",
        )
        builder.add_shape_box(
            body=box_body,
            hx=BOX_HALF_X,
            hy=BOX_HALF_Y,
            hz=BOX_HALF_Z,
            label="box_shape",
        )

        builder.end_world()

    model = builder.finalize(device=device)
    return model


def run_trial(model, solver, force_z: float, dt: float, num_substeps: int,
              max_steps: int = 2000, verbose: bool = False, viewer=None):
    """Run a single trial with the given downward force on the box.
    
    Returns: (penetrated: bool, min_z: float, impact_velocity: float, step_at_penetration: int)
    """
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()

    sub_dt = dt / num_substeps

    box_body_idx = 1  # Box body index in world 0

    min_z = float('inf')
    impact_vel = 0.0
    step_at_pen = -1
    frames = []

    for step in range(max_steps):
        # Apply downward force to the box
        body_f = state_0.body_f
        body_f_torch = wp.to_torch(body_f)
        body_f_torch[box_body_idx, 2] = force_z  # Negative Z = downward
        
        # Run substeps
        for _ in range(num_substeps):
            model.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, sub_dt)
            state_0, state_1 = state_1, state_0

        # Render if viewer is active
        if viewer is not None:
            viewer.begin_frame(step * dt)
            viewer.log_state(state_0)
            viewer.end_frame()
            frame = viewer.get_frame()
            frames.append(wp.to_torch(frame).cpu().numpy().copy())

        # Check box Z position
        body_q = wp.to_torch(state_0.body_q)
        box_z = body_q[box_body_idx, 2].item()
        
        body_qd = wp.to_torch(state_0.body_qd)
        box_vz = body_qd[box_body_idx, 2].item()

        if box_z < min_z:
            min_z = box_z
            
        if verbose and step % 100 == 0:
            print(f"  step {step}: box Z={box_z:.4f}, vel_z={box_vz:.3f}")

        # Check for penetration
        if box_z < PENETRATION_Z:
            impact_vel = box_vz
            step_at_pen = step
            if verbose:
                print(f"  PENETRATION at step {step}: Z={box_z:.4f}, vel_z={box_vz:.3f}")
            # Capture a few more frames after penetration for video
            if viewer is not None:
                for extra in range(30):
                    body_f_torch[box_body_idx, 2] = force_z
                    for _ in range(num_substeps):
                        model.collide(state_0, contacts)
                        solver.step(state_0, state_1, control, contacts, sub_dt)
                        state_0, state_1 = state_1, state_0
                    viewer.begin_frame((step + extra + 1) * dt)
                    viewer.log_state(state_0)
                    viewer.end_frame()
                    frame = viewer.get_frame()
                    frames.append(wp.to_torch(frame).cpu().numpy().copy())
            return True, min_z, impact_vel, step_at_pen, frames

        # Check if box has settled on table (small velocity, near table surface)
        if abs(box_vz) < 0.01 and abs(box_z - TABLE_SURFACE_Z) < 0.1 and step > 50:
            if verbose:
                print(f"  Settled at step {step}: Z={box_z:.4f}")
            return False, min_z, box_vz, step, frames

    return False, min_z, 0.0, max_steps, frames


def main():
    parser = argparse.ArgumentParser(description="Test box-table penetration")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_substeps", type=int, default=2, 
                        help="Number of physics substeps (default: 2, matching DexSuite)")
    parser.add_argument("--dt", type=float, default=1.0/120.0,
                        help="Physics timestep in seconds (default: 1/120)")
    parser.add_argument("--integrator", type=str, default="implicitfast",
                        choices=["implicitfast", "implicit", "euler"],
                        help="MuJoCo integrator type")
    parser.add_argument("--iterations", type=int, default=1,
                        help="Solver iterations (MuJoCo default: 1)")
    parser.add_argument("--ls_iterations", type=int, default=4,
                        help="Line search iterations (MuJoCo default: 4)")
    parser.add_argument("--visualize", action="store_true",
                        help="Show Newton viewer during test")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("Box-Table Penetration Test")
    print("=" * 70)
    print(f"  Device: {args.device}")
    print(f"  Integrator: {args.integrator}")
    print(f"  Substeps: {args.num_substeps}")
    print(f"  dt: {args.dt:.6f}s (sub_dt: {args.dt/args.num_substeps:.6f}s)")
    print(f"  Solver iterations: {args.iterations}")
    print(f"  LS iterations: {args.ls_iterations}")
    print(f"  Table surface Z: {TABLE_SURFACE_Z}")
    print(f"  Penetration threshold Z: {PENETRATION_Z}")
    print(f"  Box mass: {BOX_MASS} kg")
    print("=" * 70)

    # Test with increasing downward forces
    # Gravity is -9.81, so net downward force = applied + gravity * mass
    # At 2.4 m/s impact (observed in DexSuite), KE = 0.5 * 0.2 * 2.4^2 = 0.576 J
    forces = [0.0, -5.0, -10.0, -20.0, -50.0, -100.0, -200.0, -500.0, -1000.0]

    results = []
    
    # Create viewer for the penetration trial
    viewer = None
    if args.visualize:
        from newton.viewer import ViewerGL
        viewer = ViewerGL(width=1280, height=720, headless=True)

    for force_z in forces:
        model = build_scene(num_worlds=1, device=args.device)
        solver = newton.solvers.SolverMuJoCo(
            model,
            integrator=args.integrator,
            iterations=args.iterations,
            ls_iterations=args.ls_iterations,
        )

        # Set up viewer for this model (only record the failing trial)
        trial_viewer = None
        if viewer is not None and force_z <= -200.0:
            viewer.set_model(model)
            trial_viewer = viewer

        net_force = force_z - 9.81 * BOX_MASS
        print(f"\nForce: {force_z:8.1f} N (net: {net_force:8.2f} N)")

        penetrated, min_z, vel_z, step, frames = run_trial(
            model, solver, force_z, args.dt, args.num_substeps,
            verbose=args.verbose, viewer=trial_viewer,
        )

        status = "PENETRATED" if penetrated else "OK"
        print(f"  Result: {status} | min_z={min_z:.4f} | vel_z={vel_z:.3f} | step={step}")

        results.append({
            "force": force_z,
            "net_force": net_force,
            "penetrated": penetrated,
            "min_z": min_z,
            "vel_z": vel_z,
            "step": step,
            "frames": frames,
        })

        if penetrated:
            # Found the threshold — do a binary search for exact threshold
            print(f"\n  >>> Penetration at {force_z} N! <<<")
            break

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for r in results:
        status = "FAIL" if r["penetrated"] else "PASS"
        print(f"  [{status}] Force={r['force']:8.1f}N  net={r['net_force']:8.2f}N  "
              f"min_z={r['min_z']:.4f}  vel_z={r['vel_z']:.3f}  step={r['step']}")

    penetrated_any = any(r["penetrated"] for r in results)
    if penetrated_any:
        first_fail = next(r for r in results if r["penetrated"])
        print(f"\n  FAILURE: Box penetrated table at {first_fail['force']}N applied force")
        print(f"  Impact velocity: {first_fail['vel_z']:.3f} m/s")
        print(f"  This reproduces the DexSuite box-through-table bug.")

        # Save video if we captured frames
        if first_fail["frames"]:
            import numpy as np
            try:
                import imageio
                video_path = "penetration_repro.mp4"
                writer = imageio.get_writer(video_path, fps=30)
                for frame in first_fail["frames"]:
                    writer.append_data(frame)
                writer.close()
                print(f"\n  Video saved: {video_path} ({len(first_fail['frames'])} frames)")
            except ImportError:
                print("\n  Install imageio to save video: pip install imageio[ffmpeg]")
    else:
        print(f"\n  All forces tested without penetration.")
        print(f"  Consider testing with higher forces or fewer substeps.")

    if viewer is not None:
        viewer.close()

    return 1 if penetrated_any else 0


if __name__ == "__main__":
    exit(main())
