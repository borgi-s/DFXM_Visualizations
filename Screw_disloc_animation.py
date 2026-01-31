"""
PyVista animation for a *screw-like* dislocation visualization (3D -> 2D -> zoom-out).

Sequence requested:
1) Start with atoms as big points, then shrink until they "disappear" (lines can remain).
2) Highlight the dislocation line.
3) Perform the shearing motion, then restore points to original size.
4) Move camera into a 2D view (orthographic) on the plane normal to the dislocation line.
5) "Zoom out" by stepping through larger lattices (more atoms in view), similar to the edge-dislocation animation.

Update (stress coloring):
- Coloring is now based on the *isotropic elasticity* stress field of an ideal screw dislocation.
- For a screw with line || z, the only non-zero components are σ_xz and σ_yz (up to symmetry).
  We compute them from u3 = (b/2π) atan(x2/x1), and σ = μ * grad(u_parallel) in the perpendicular plane.
- By default we use μ=1 (matching Atomsk's convention when μ is unknown: the code is effectively σ/μ).

Run:
  python Screw_dislocation_animation_sequence_zoomout_stress.py

Headless Linux/HPC:
- If you get a VTK/X server error, uncomment `pv.start_xvfb()`.
"""

from __future__ import annotations

import numpy as np

try:
    import pyvista as pv
except ImportError:
    raise SystemExit(
        "PyVista is not installed.\n\n"
        "Install:\n"
        "  pip install pyvista\n"
        "Optional (for MP4):\n"
        "  pip install imageio-ffmpeg\n"
    )


# ------------------------------------------------------------
# Screw dislocation stresses (isotropic elasticity)
# ------------------------------------------------------------
def screw_stress_tensor_iso(
    rx, ry, rz,
    b: float = 2.862e-4,
    mu: float = 1.0,
    alpha: float = 1e-20,
    line_axis: str = "z",
):
    """
    Compute the shear-stress tensor components for an ideal screw dislocation in isotropic elasticity.

    For line || z (classic):
      u_z = (b/2π) atan(y/x)
      σ_xz = μ ∂u_z/∂x = - μ (b/2π) y/(x^2+y^2)
      σ_yz = μ ∂u_z/∂y =   μ (b/2π) x/(x^2+y^2)

    This function generalizes by permuting axes:
      - line || x : perp plane (y,z), nonzero σ_xy and σ_xz
      - line || y : perp plane (x,z), nonzero σ_xy and σ_yz
      - line || z : perp plane (x,y), nonzero σ_xz and σ_yz

    Returns
    -------
    sigma : (N,3,3) ndarray
        Symmetric stress tensor (only the two shear components are filled).
    """
    rx = np.asarray(rx, float).ravel()
    ry = np.asarray(ry, float).ravel()
    rz = np.asarray(rz, float).ravel()
    N = rx.size

    ax = line_axis.lower()
    if ax == "x":
        # line || x, perp plane (y,z)
        p1, p2 = ry, rz
        i_par = 0
        j_p1, j_p2 = 1, 2  # y, z
    elif ax == "y":
        # line || y, perp plane (x,z)
        p1, p2 = rx, rz
        i_par = 1
        j_p1, j_p2 = 0, 2  # x, z
    elif ax == "z":
        # line || z, perp plane (x,y)
        p1, p2 = rx, ry
        i_par = 2
        j_p1, j_p2 = 0, 1  # x, y
    else:
        raise ValueError("line_axis must be one of {'x','y','z'}")

    denom = (p1 * p1 + p2 * p2) + alpha
    coef = mu * (b / (2.0 * np.pi))

    # grad(u_parallel) in the perpendicular plane
    du_dp1 = (-p2 / denom) * coef
    du_dp2 = ( p1 / denom) * coef

    sigma = np.zeros((N, 3, 3), dtype=float)

    # σ_{p1,par} = σ_{par,p1} = μ ∂u_par/∂p1
    sigma[:, j_p1, i_par] = du_dp1
    sigma[:, i_par, j_p1] = du_dp1

    # σ_{p2,par} = σ_{par,p2} = μ ∂u_par/∂p2
    sigma[:, j_p2, i_par] = du_dp2
    sigma[:, i_par, j_p2] = du_dp2

    return sigma


def stress_components_for_display(sigma: np.ndarray, line_axis: str = "z"):
    """
    Return the pair of shear components that exist for the chosen line axis.

    Returns (c1, c2, name1, name2), where name1/name2 are strings like 'sigma_xz'.
    """
    ax = line_axis.lower()
    if ax == "z":
        return sigma[:, 0, 2], sigma[:, 1, 2], "sigma_xz", "sigma_yz"
    if ax == "x":
        return sigma[:, 0, 1], sigma[:, 0, 2], "sigma_xy", "sigma_xz"
    if ax == "y":
        return sigma[:, 0, 1], sigma[:, 1, 2], "sigma_xy", "sigma_yz"
    raise ValueError("line_axis must be one of {'x','y','z'}")


def select_stress_scalar(c1, c2, name1: str, name2: str, mode: str = "c1"):
    """
    Choose which scalar to color by.

    mode:
      - "c1"  -> first shear component (e.g. sigma_xz for line||z)
      - "c2"  -> second shear component (e.g. sigma_yz for line||z)
      - "mag" -> sqrt(c1^2 + c2^2)
    """
    m = mode.lower()
    if m in ("c1", "1", "first", name1.lower()):
        return c1, name1
    if m in ("c2", "2", "second", name2.lower()):
        return c2, name2
    if m in ("mag", "magnitude", "abs"):
        return np.sqrt(c1 * c1 + c2 * c2), f"sqrt({name1}^2+{name2}^2)"
    raise ValueError(f"Unknown stress scalar mode: {mode!r}")


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def choose_Nz(Nx, Ny):
    n = int(round(np.sqrt(Nx * Ny)))
    anchors = np.array([6, 12, 24, 40], dtype=float)
    values  = np.array([5,  3,  2,  1 ], dtype=float)
    Nz = np.interp(n, anchors, values)
    return int(np.clip(np.rint(Nz), 1, 6))


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return (1.0 - t) * a + t * b


def _lines_from_pairs(pairs: list[tuple[int, int]]) -> np.ndarray:
    arr = np.empty((len(pairs), 3), dtype=np.int64)
    arr[:, 0] = 2
    arr[:, 1] = [p[0] for p in pairs]
    arr[:, 2] = [p[1] for p in pairs]
    return arr.ravel()


def scene_bounds(points: np.ndarray) -> tuple[float, float, float, float, float, float]:
    xmin, ymin, zmin = points.min(axis=0)
    xmax, ymax, zmax = points.max(axis=0)
    return xmin, xmax, ymin, ymax, zmin, zmax


def parallel_scale_from_bounds(bounds, line_axis: str) -> float:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    xr = xmax - xmin
    yr = ymax - ymin
    zr = zmax - zmin
    ax = line_axis.lower()
    if ax == "z":
        span = max(xr, yr)
    elif ax == "x":
        span = max(yr, zr)
    elif ax == "y":
        span = max(xr, zr)
    else:
        raise ValueError("line_axis must be one of {'x','y','z'}")
    return float(0.60 * 0.5 * span)


def point_size_from_counts(N_total: int) -> float:
    # Bigger points to reduce white space
    N0 = 8 * 8 * 5
    base = 250.0
    ps = base * np.sqrt(N0 / max(N_total, 1))
    return float(np.clip(ps, 10.0, 140.0))


def set_actor_opacity(actor, opacity: float):
    if actor is None:
        return
    actor.GetProperty().SetOpacity(float(opacity))


def set_actor_point_size(actor, ps: float):
    if actor is None:
        return
    actor.GetProperty().SetPointSize(float(ps))


def set_camera_3d(plotter: pv.Plotter, bounds, view_angle: float = 30.0):
    """Fixed 3D camera that reads as a cube and doesn't wobble."""
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    cz = 0.5 * (zmin + zmax)

    xr = xmax - xmin
    yr = ymax - ymin
    zr = zmax - zmin
    R = 0.5 * max(xr, yr, zr)
    R = max(R, 1e-6)

    direction = np.array([1.25, 0.95, 1.20], dtype=float)
    direction /= np.linalg.norm(direction)

    cam_dist = 6.4 * R
    pos = np.array([cx, cy, cz], dtype=float) + cam_dist * direction

    plotter.camera.position = tuple(pos.tolist())
    plotter.camera.focal_point = (float(cx), float(cy), float(cz))
    plotter.camera.up = (0.0, 1.0, 0.0)
    plotter.camera.view_angle = float(view_angle)
    plotter.reset_camera_clipping_range()


def camera_target_for_axis(bounds, line_axis: str, dist_factor: float = 6.0):
    """Target perspective camera looking along the dislocation line axis toward scene center."""
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    c = np.array([0.5*(xmin+xmax), 0.5*(ymin+ymax), 0.5*(zmin+zmax)], dtype=float)
    xr, yr, zr = (xmax-xmin), (ymax-ymin), (zmax-zmin)
    R = max(0.5*max(xr, yr, zr), 1e-6)

    ax = line_axis.lower()
    if ax == "z":
        direction = np.array([0.0, 0.0, 1.0])
        up = np.array([0.0, 1.0, 0.0])
    elif ax == "x":
        direction = np.array([1.0, 0.0, 0.0])
        up = np.array([0.0, 0.0, 1.0])  # avoid up parallel to view
    elif ax == "y":
        direction = np.array([0.0, 1.0, 0.0])
        up = np.array([0.0, 0.0, 1.0])
    else:
        raise ValueError("line_axis must be one of {'x','y','z'}")

    pos = c + (dist_factor * R) * direction
    return tuple(pos.tolist()), tuple(c.tolist()), tuple(up.tolist())


# ------------------------------------------------------------
# Scene construction
# ------------------------------------------------------------
def build_scene_screw(
    N_xy: int,
    a: float = 1.0,
    z_scale: float = 1.7,
    core_x: float = 0.0,
    core_y: float = 0.0,
    core_z: float = 0.0,
    line_axis: str = "z",
    shear_step: float | None = None,
    shear_w: float | None = None,
    y_w: float | None = None,
    decay_y: float | None = None,
    # stress params:
    b: float = 2.862e-4,
    mu: float = 1.0,
    alpha: float = 1e-20,
    stress_mode: str = "c1",
):
    """
    Perfect lattice and a *visual* "screw-like" sheared lattice.

    Visual defect:
      - split by x<0 vs x>0 (smoothed by tanh) but only in y>0 (smoothed gate + decay)
      - shift is applied along the dislocation line direction to make the 3D shear obvious

    Coloring:
      - uses isotropic-elastic stresses of an ideal screw dislocation (normalized to μ if mu=1).
    """
    Nx = Ny = int(N_xy)
    Nz = choose_Nz(Nx, Ny)

    if shear_step is None:
        shear_step = 0.5 * a
    if shear_w is None:
        shear_w = 0.15 * a
    if y_w is None:
        y_w = 0.60 * a
    if decay_y is None:
        decay_y = 4.0 * a

    x = (np.arange(Nx) - (Nx - 1) / 2) * a
    y = (np.arange(Ny) - (Ny - 1) / 2) * a
    X, Y = np.meshgrid(x, y, indexing="xy")

    z = (np.arange(Nz) - (Nz - 1) / 2) * a * z_scale

    base_per_layer = Nx * Ny
    N_total = Nz * base_per_layer

    pts_perf = np.column_stack([
        np.tile(X.ravel(), Nz),
        np.tile(Y.ravel(), Nz),
        np.repeat(z, base_per_layer),
    ])

    # --- top-half-only shear field (visual) ---
    split_x = np.tanh((X - core_x) / shear_w)               # -1 left, +1 right
    gate_y = 0.5 * (1.0 + np.tanh((Y - core_y) / y_w))      # ~0 below, ~1 above
    gate_y *= np.exp(-np.clip(Y - core_y, 0.0, None) / decay_y)

    d_2d = -shear_step * split_x * gate_y                    # signed magnitude

    pts_def = pts_perf.copy()
    ax = line_axis.lower()
    if ax == "x":
        pts_def[:, 0] += np.tile(d_2d.ravel(), Nz)
    elif ax == "y":
        pts_def[:, 1] += np.tile(d_2d.ravel(), Nz)
    elif ax == "z":
        pts_def[:, 2] += np.tile(d_2d.ravel(), Nz)
    else:
        raise ValueError("line_axis must be one of {'x','y','z'}")

    # --- stresses (theory) ---
    rx = pts_perf[:, 0] - core_x
    ry = pts_perf[:, 1] - core_y
    rz = pts_perf[:, 2] - core_z

    sigma = screw_stress_tensor_iso(rx, ry, rz, b=b, mu=mu, alpha=alpha, line_axis=line_axis)
    c1, c2, n1, n2 = stress_components_for_display(sigma, line_axis=line_axis)
    scalar, scalar_name = select_stress_scalar(c1, c2, n1, n2, mode=stress_mode)

    # color limits
    if stress_mode.lower() in ("mag", "magnitude", "abs"):
        vmin, vmax = np.percentile(scalar, [2, 98])
        clim = (float(vmin), float(vmax))
    else:
        smax = np.percentile(np.abs(scalar), 98)
        smax = float(max(smax, 1e-30))
        clim = (-smax, smax)

    # Lines: x,y neighbors within layers + between layers
    pairs: list[tuple[int, int]] = []
    for k in range(Nz):
        off = k * base_per_layer
        for j in range(Ny):
            row = off + j * Nx
            for i in range(Nx):
                idx = row + i
                if i + 1 < Nx:
                    pairs.append((idx, idx + 1))
                if j + 1 < Ny:
                    pairs.append((idx, idx + Nx))

    for k in range(Nz - 1):
        off0 = k * base_per_layer
        off1 = (k + 1) * base_per_layer
        for p in range(base_per_layer):
            pairs.append((off0 + p, off1 + p))

    lines = _lines_from_pairs(pairs)
    point_size = point_size_from_counts(N_total)

    return {
        "N_xy": N_xy,
        "Nx": Nx, "Ny": Ny, "Nz": Nz,
        "pts_perf": pts_perf,
        "pts_def": pts_def,
        "sigma_c1": c1,
        "sigma_c2": c2,
        "scalar": scalar,
        "scalar_name": scalar_name,
        "clim": clim,
        "lines": lines,
        "point_size": point_size,
        "bounds_def": scene_bounds(pts_def),
    }


# ------------------------------------------------------------
# Main animation
# ------------------------------------------------------------
def main():
    # pv.start_xvfb()  # uncomment for headless systems

    # ----------------------
    # Main knobs
    # ----------------------
    line_axis = "z"          # dislocation line direction
    N_start = 8              # starting lattice
    stress_mode = "c2"       # "c1", "c2", or "mag" (see select_stress_scalar)
    mu = 1.0                 # set to 1.0 to mimic Atomsk's σ/μ convention

    # "zoom-out" sizes (increasing FOV / more atoms in view)
    sizes = [N_start, 12, 16, 20, 28, 38, 48, 68, 96]

    out_mp4 = "screw_dislocation_sequence_stress_c2.mp4"
    out_gif = "screw_dislocation_sequence_stress.gif"

    # Timing (frames)
    fps = 60

    hold_big = 35
    shrink_frames = 90

    line_fade_in = 40
    line_hold = 25

    shear_frames = 200
    grow_frames = 80

    cam_move_frames = 90
    fade_to_color = 60

    zoom_fade = 90
    hold_each = 0
    final_hold = 180

    # Build first scene
    s0 = build_scene_screw(N_start, line_axis=line_axis, stress_mode=stress_mode, mu=mu)

    plotter = pv.Plotter(window_size=(1100, 900), off_screen=True)
    plotter.set_background("white")

    # Start in 3D perspective
    set_camera_3d(plotter, s0["bounds_def"], view_angle=30.0)

    # PolyData
    poly = pv.PolyData(s0["pts_perf"])
    poly["scalar"] = s0["scalar"]

    line_poly = pv.PolyData(s0["pts_perf"].copy())
    line_poly.lines = s0["lines"]

    # Actors: points (blue + colored) and lattice lines
    a_blue = plotter.add_mesh(
        poly,
        color="tab:blue",
        style="points",
        render_points_as_spheres=True,
        point_size=s0["point_size"],
        opacity=1.0,
    )
    a_col = plotter.add_mesh(
        poly,
        scalars="scalar",
        cmap="coolwarm",
        clim=s0["clim"],
        show_scalar_bar=False,
        style="points",
        render_points_as_spheres=True,
        point_size=s0["point_size"],
        opacity=0.0,
    )
    a_lines = plotter.add_mesh(
        line_poly,
        color="black",
        opacity=0.18,
        line_width=1.3,
    )

    # Dislocation line actor (highlight)
    xmin, xmax, ymin, ymax, zmin, zmax = s0["bounds_def"]
    ext = 0.15 * (zmax - zmin + 1e-6)
    if line_axis.lower() == "z":
        p0 = (0.0, 0.0, float(zmin - ext))
        p1 = (0.0, 0.0, float(zmax + ext))
    elif line_axis.lower() == "x":
        extx = 0.15 * (xmax - xmin + 1e-6)
        p0 = (float(xmin - extx), 0.0, 0.0)
        p1 = (float(xmax + extx), 0.0, 0.0)
    else:  # "y"
        exty = 0.15 * (ymax - ymin + 1e-6)
        p0 = (0.0, float(ymin - exty), 0.0)
        p1 = (0.0, float(ymax + exty), 0.0)

    disl_line = pv.Line(p0, p1)
    a_disl = plotter.add_mesh(
        disl_line,
        color="red",
        line_width=8.0,
        opacity=0.0,
    )

    # Writer
    writer = None
    try:
        plotter.open_movie(out_mp4, framerate=fps, quality=8)
        writer = "mp4"
        print(f"[writer] MP4 -> {out_mp4}")
    except Exception as e:
        print(f"[writer] MP4 failed ({type(e).__name__}: {e}). Falling back to GIF.")
        plotter.open_gif(out_gif, fps=fps)
        writer = "gif"
        print(f"[writer] GIF -> {out_gif}")

    def frame():
        plotter.render()
        plotter.write_frame()

    # ------------------------------------------------------------
    # (1) Big points hold
    # ------------------------------------------------------------
    for _ in range(hold_big):
        frame()

    # ------------------------------------------------------------
    # (1b) Shrink points until they "disappear" (keep lines)
    # ------------------------------------------------------------
    ps0 = s0["point_size"]
    ps_min = 0.1
    for k in range(shrink_frames):
        t = k / (shrink_frames - 1)
        ps = (1.0 - t) * ps0 + t * ps_min
        set_actor_point_size(a_blue, ps)
        set_actor_point_size(a_col, ps)
        set_actor_opacity(a_blue, 1.0 - t)
        set_actor_opacity(a_col, 0.0)
        set_actor_opacity(a_lines, 0.18 + 0.10 * t)
        frame()

    set_actor_opacity(a_blue, 0.0)

    # ------------------------------------------------------------
    # (2) Highlight dislocation line
    # ------------------------------------------------------------
    for k in range(line_fade_in):
        t = k / (line_fade_in - 1)
        set_actor_opacity(a_disl, t)
        frame()

    for _ in range(line_hold):
        frame()

    # ------------------------------------------------------------
    # (3) Shear motion (show via lines; points still tiny/invisible)
    # ------------------------------------------------------------
    s_shear = s0
    for k in range(shear_frames):
        t = k / (shear_frames - 1)
        pts = lerp(s_shear["pts_perf"], s_shear["pts_def"], t)
        poly.points = pts
        line_poly.points = pts
        if k % 12 == 0:
            plotter.reset_camera_clipping_range()
        frame()

    # ------------------------------------------------------------
    # (3b) Grow points back to original size
    # ------------------------------------------------------------
    for k in range(grow_frames):
        t = k / (grow_frames - 1)
        ps = (1.0 - t) * ps_min + t * ps0
        set_actor_point_size(a_blue, ps)
        set_actor_point_size(a_col, ps)
        set_actor_opacity(a_blue, t)   # bring back blue points
        set_actor_opacity(a_lines, (1.0 - t) * 0.28 + t * 0.18)
        set_actor_opacity(a_disl, 1.0)
        frame()

    set_actor_opacity(a_blue, 1.0)
    set_actor_opacity(a_lines, 0.18)

    # ------------------------------------------------------------
    # (4) Move camera to 2D view normal to dislocation line
    #     - animate perspective move toward axis view
    #     - then switch to parallel projection at the end
    # ------------------------------------------------------------
    bnds_now = scene_bounds(poly.points)
    pos_tgt, foc_tgt, up_tgt = camera_target_for_axis(bnds_now, line_axis=line_axis, dist_factor=7.0)

    pos0 = tuple(plotter.camera.position)
    foc0 = tuple(plotter.camera.focal_point)

    for k in range(cam_move_frames):
        t = k / (cam_move_frames - 1)

        pos = tuple((1.0 - t) * np.array(pos0) + t * np.array(pos_tgt))
        foc = tuple((1.0 - t) * np.array(foc0) + t * np.array(foc_tgt))

        plotter.camera.position = pos
        plotter.camera.focal_point = foc
        plotter.camera.up = up_tgt
        plotter.camera.view_angle = float((1.0 - t) * 30.0 + t * 18.0)

        set_actor_opacity(a_disl, 1.0 - t)
        if k % 10 == 0:
            plotter.reset_camera_clipping_range()
        frame()

    plotter.enable_parallel_projection()
    plotter.camera.position = pos_tgt
    plotter.camera.focal_point = foc_tgt
    plotter.camera.up = up_tgt
    plotter.camera.parallel_scale = parallel_scale_from_bounds(bnds_now, line_axis=line_axis)
    plotter.reset_camera_clipping_range()

    set_actor_opacity(a_disl, 0.0)
    a_disl.SetVisibility(False)

    # ------------------------------------------------------------
    # (4b) Crossfade blue -> colored (now in 2D view)
    # ------------------------------------------------------------
    for k in range(fade_to_color):
        t = k / (fade_to_color - 1)
        set_actor_opacity(a_blue, 1.0 - t)
        set_actor_opacity(a_col, t)
        set_actor_opacity(a_lines, 0.18 * (1.0 - 0.35 * t))
        frame()

    set_actor_opacity(a_blue, 0.0)
    set_actor_opacity(a_col, 1.0)

    # ------------------------------------------------------------
    # (5) "Zoom out" by stepping through sizes (more atoms in view)
    # ------------------------------------------------------------
    prev = {
        "a_col": a_col,
        "a_lines": a_lines,
        "scene": s0,
    }

    # ensure current geometry is sheared state
    poly.points = s0["pts_def"]
    line_poly.points = s0["pts_def"]

    for N_xy in sizes[1:]:
        s = build_scene_screw(N_xy, line_axis=line_axis, stress_mode=stress_mode, mu=mu)

        poly_n = pv.PolyData(s["pts_def"])
        poly_n["scalar"] = s["scalar"]

        line_poly_n = pv.PolyData(s["pts_def"].copy())
        line_poly_n.lines = s["lines"]

        a_col_n = plotter.add_mesh(
            poly_n,
            scalars="scalar",
            cmap="coolwarm",
            clim=s["clim"],
            show_scalar_bar=False,
            style="points",
            render_points_as_spheres=True,
            point_size=s["point_size"],
            opacity=0.0,
        )
        a_lines_n = plotter.add_mesh(
            line_poly_n,
            color="black",
            opacity=0.0,
            line_width=1.3,
        )

        bnds = s["bounds_def"]
        target_scale = parallel_scale_from_bounds(bnds, line_axis=line_axis)
        start_scale = float(plotter.camera.parallel_scale)

        for k in range(zoom_fade):
            t = k / (zoom_fade - 1)

            set_actor_opacity(prev["a_col"], 1.0)
            set_actor_opacity(a_col_n, 1.0)

            show_prev = (t < 0.5)
            prev["a_col"].SetVisibility(show_prev)
            prev["a_lines"].SetVisibility(show_prev)
            a_col_n.SetVisibility(not show_prev)
            a_lines_n.SetVisibility(not show_prev)

            set_actor_opacity(prev["a_lines"], 0.18 * (1.0 - t))
            set_actor_opacity(a_lines_n, 0.18 * t)

            plotter.camera.parallel_scale = float((1.0 - t) * start_scale + t * target_scale)
            frame()

        plotter.remove_actor(prev["a_col"])
        plotter.remove_actor(prev["a_lines"])

        prev = {
            "a_col": a_col_n,
            "a_lines": a_lines_n,
            "scene": s,
        }

        for _ in range(hold_each):
            frame()

    for _ in range(final_hold):
        frame()

    plotter.close()

    if writer == "mp4":
        print(f"Done. Wrote {out_mp4}")
    else:
        print(f"Done. Wrote {out_gif}")


if __name__ == "__main__":
    main()
