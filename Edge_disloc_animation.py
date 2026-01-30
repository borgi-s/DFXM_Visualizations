"""
PyVista animation for the edge-dislocation lattice.

What it does:
1) Shows the perfect lattice for N_xy=6 (monochrome).
2) Introduces the half-plane: smoothly morphs perfect -> defect geometry (monochrome),
   while fading in the inserted half-plane atoms and crossfading the bond network.
3) Crossfades to the strain-colormapped version.
4) Then steps through the remaining N_xy sizes, crossfading between scenes.

Outputs:
- MP4 if possible (needs imageio-ffmpeg / ffmpeg), otherwise GIF.

Run:
  python Edge_dislocation_pyvista_animation.py

Notes for headless Linux/HPC:
- If you get a VTK/X server error, try to uncomment `pv.start_xvfb()`.
"""

from __future__ import annotations

import os
import numpy as np

try:
    import pyvista as pv
except ImportError:
    # Avoid `raise ... from e` here so the script can be parsed even
    # by older Python builds that don't support exception chaining syntax.
    raise SystemExit(
        "PyVista is not installed (or you're running an old Python).\n\n"
        "Use Python 3 and install dependencies:\n"
        "  pip install pyvista\n"
        "Optional (for MP4 on many setups):\n"
        "  pip install imageio-ffmpeg\n"
    )


# -----------------------------
# Hirth & Lothe displacement-gradient for an edge dislocation
# (line along z, Burgers along +x)
# Returns grad_u = du_i/dx_j as a (N,3,3) array (z terms left as 0).
# -----------------------------
def grad_u_edge_hirth_lothe(rx, ry, b=2.862e-4, nu=0.334, alpha=1e-20):
    rx = np.asarray(rx, float)
    ry = np.asarray(ry, float)
    N = rx.size

    sqx = rx * rx
    sqy = ry * ry
    r2 = sqx + sqy
    denom = r2 * r2 + alpha
    bfactor = b / (4 * np.pi * (1 - nu))
    nufactor = 2 * nu * r2

    Fdd = np.zeros((N, 3, 3), dtype=float)
    Fdd[:, 0, 0] = -ry * (3 * sqx + sqy - nufactor) / denom
    Fdd[:, 0, 1] =  rx * (3 * sqx + sqy - nufactor) / denom
    Fdd[:, 1, 0] = -rx * (3 * sqy + sqx - nufactor) / denom
    Fdd[:, 1, 1] =  ry * (sqx - sqy + nufactor) / denom
    Fdd *= bfactor
    return Fdd


def choose_Nz(Nx, Ny):
    n = int(round(np.sqrt(Nx * Ny)))  # effective "side length"
    anchors = np.array([6, 12, 24, 40], dtype=float)
    values  = np.array([5,  3,  2,  1 ], dtype=float)
    Nz = np.interp(n, anchors, values)
    return int(np.clip(np.rint(Nz), 1, 5))


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return (1.0 - t) * a + t * b

def rotate_y(points: np.ndarray, deg: float, origin: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    """Rotate Nx3 points around the y-axis by `deg` degrees (right-handed), about `origin`."""
    pts = np.asarray(points, dtype=float)
    theta = np.deg2rad(float(deg))
    c, s = np.cos(theta), np.sin(theta)
    o = np.asarray(origin, dtype=float)
    p = pts - o
    x = p[:, 0] * c + p[:, 2] * s
    z = -p[:, 0] * s + p[:, 2] * c
    out = p.copy()
    out[:, 0] = x
    out[:, 2] = z
    return out + o


def _lines_from_pairs(pairs: list[tuple[int, int]]) -> np.ndarray:
    # VTK "lines" array: [2, i0, i1, 2, i0, i1, ...]
    arr = np.empty((len(pairs), 3), dtype=np.int64)
    arr[:, 0] = 2
    arr[:, 1] = [p[0] for p in pairs]
    arr[:, 2] = [p[1] for p in pairs]
    return arr.ravel()


def scene_bounds(points_list: list[np.ndarray]) -> tuple[float, float, float, float, float, float]:
    pts = np.vstack([p for p in points_list if p is not None and len(p) > 0])
    xmin, ymin, zmin = pts.min(axis=0)
    xmax, ymax, zmax = pts.max(axis=0)
    return xmin, xmax, ymin, ymax, zmin, zmax


def parallel_scale_from_bounds(bounds) -> float:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    xr = xmax - xmin
    yr = ymax - ymin
    # parallel_scale is roughly half-height of the view in world coords.
    return 0.60 * 0.5 * max(xr, yr)


def point_size_from_counts(N_total: int) -> float:
    # Matplotlib used a "marker area". PyVista uses point_size in pixels.
    # This heuristic keeps things readable across sizes.
    N0 = 6 * 6 * 5
    base = 140.0
    ps = base * np.sqrt(N0 / max(N_total, 1))
    return float(np.clip(ps, 4.0, 48.0))

def smoothstep(t: float) -> float:
    # smooth 0→1 with zero slope at ends (no abrupt starts/stops)
    return t * t * (3.0 - 2.0 * t)

def build_scene(N_xy: int, a: float = 1.0, z_scale: float = 2.0, x_insert: float = 0.0):
    Nx = Ny = int(N_xy)
    Nz = choose_Nz(Nx, Ny)

    z = (np.arange(Nz) - (Nz - 1) / 2) * a * z_scale

    # grid
    x = (np.arange(Nx) - (Nx - 1) / 2) * a
    y = (np.arange(Ny) - (Ny - 1) / 2) * a
    X, Y = np.meshgrid(x, y, indexing="xy")

    # strain constants
    b = 2.862e-4
    nu = 0.334

    # defect geometry parameters
    gap = a
    w = 0.1 * a

    i_left = int(np.max(np.where(x < x_insert)[0]))
    i_right = int(np.min(np.where(x > x_insert)[0]))

    y_w = 0.5 * a
    decay = 3.0 * a
    bottom_scale = 0.1

    s = 0.5 * (1.0 + np.tanh(Y / y_w))                 # 0..1 across y=0
    amp_y = bottom_scale + (1.0 - bottom_scale) * s    # bottom..top
    amp_y *= np.where(Y < 0, np.exp(Y / decay), 1.0)   # one-sided fade

    split = np.tanh((X - x_insert) / w)
    dX = 0.5 * gap * split * amp_y

    X_def = X + dX
    Y_def = Y.copy()

    # --- NEW: squeeze/compress top half (y>0) toward x_insert ---
    # strongest at y=0+, decays as y increases
    # --- NEW: squeeze/compress ONLY top half (y>0), decaying in both y and x ---
    squeeze_amp     = 0.20      # max compression at (x=0, y=0+)
    squeeze_decay_y = 20.0 * a   # decay length in y
    squeeze_decay_x = 2.0 * a   # decay length in x
    x0 = x_insert              # use x_insert

    squeeze = np.zeros_like(Y_def, dtype=float)
    mask = Y_def > 0.0

    dx = np.abs(X_def - x0)  # or use np.abs(X - x0) if you want decay based on the *perfect* x-grid
    squeeze[mask] = (
        squeeze_amp
        * np.exp(-Y_def[mask] / squeeze_decay_y)
        * np.exp(-dx[mask]    / squeeze_decay_x)
    )

    # compress x-distance toward x0
    X_def = x0 + (X_def - x0) * (1.0 - squeeze)

    grad_u0 = grad_u_edge_hirth_lothe(X_def.ravel(), Y_def.ravel(), b=b, nu=nu, alpha=1e-20)
    eps0 = 0.5 * (grad_u0 + np.transpose(grad_u0, (0, 2, 1)))
    eps_vol0_2d = eps0[:, 0, 0] + eps0[:, 1, 1]


    # inserted column only for y>0
    j_top = np.where(y > 0)[0]
    y_top = y[j_top]

    X_newcol_def = np.empty_like(y_top, dtype=float)
    for k, j in enumerate(j_top):
        X_newcol_def[k] = 0.5 * (X_def[j, i_left] + X_def[j, i_right])
    Y_newcol = y_top.copy()

    # Perfect-lattice "ghost" positions for inserted atoms (midpoint in perfect grid)
    X_newcol_perf = np.empty_like(y_top, dtype=float)
    for k, j in enumerate(j_top):
        X_newcol_perf[k] = 0.5 * (X[j, i_left] + X[j, i_right])

    # Replicate to 3D
    base_per_layer = Nx * Ny
    ins_per_layer = len(y_top)
    total_per_layer = base_per_layer + ins_per_layer

    # Base points (perfect and defect)
    base_perf = np.column_stack([np.tile(X.ravel(), Nz), np.tile(Y.ravel(), Nz), np.repeat(z, base_per_layer)])
    base_def  = np.column_stack([np.tile(X_def.ravel(), Nz), np.tile(Y_def.ravel(), Nz), np.repeat(z, base_per_layer)])

    # Inserted points (perfect "ghost" and defect)
    ins_perf = np.column_stack([np.tile(X_newcol_perf, Nz), np.tile(Y_newcol, Nz), np.repeat(z, ins_per_layer)])
    ins_def  = np.column_stack([np.tile(X_newcol_def,  Nz), np.tile(Y_newcol, Nz), np.repeat(z, ins_per_layer)])

    # Scalars (defect colored state)
    C_base = np.tile(eps_vol0_2d, Nz)

    grad_u_ins = grad_u_edge_hirth_lothe(X_newcol_def, Y_newcol, b=b, nu=nu, alpha=1e-20)
    eps_ins = 0.5 * (grad_u_ins + np.transpose(grad_u_ins, (0, 2, 1)))
    eps_vol_ins_2d = eps_ins[:, 0, 0] + eps_ins[:, 1, 1]
    C_ins = np.tile(eps_vol_ins_2d, Nz)

    all_vals = np.concatenate([C_base, C_ins]) if len(C_ins) else C_base
    vmin, vmax = np.percentile(all_vals, [2, 98])

    # Lines connectivity
    # Perfect: base-only nearest neighbor grid
    line_pairs_perf = []
    for k in range(Nz):
        layer_offset = k * base_per_layer
        for j in range(Ny):
            for i in range(Nx):
                idx = layer_offset + j * Nx + i
                if i + 1 < Nx:
                    line_pairs_perf.append((idx, idx + 1))
                if j + 1 < Ny:
                    line_pairs_perf.append((idx, idx + Nx))

    # Defect: base + inserted (per-layer indexing: base first, then inserted)
    j_top_set = set(j_top.tolist())
    line_pairs_def = []
    for k in range(Nz):
        layer_offset = k * total_per_layer

        def bidx(i, j) -> int:
            return layer_offset + j * Nx + i

        def iidx(m) -> int:
            return layer_offset + base_per_layer + m

        # base horizontal + vertical
        for j in range(Ny):
            for i in range(Nx):
                if i + 1 < Nx:
                    if not (i == i_left and j in j_top_set):
                        line_pairs_def.append((bidx(i, j), bidx(i + 1, j)))
                if j + 1 < Ny:
                    line_pairs_def.append((bidx(i, j), bidx(i, j + 1)))

        # inserted internal
        for m in range(ins_per_layer - 1):
            line_pairs_def.append((iidx(m), iidx(m + 1)))

        # inserted to left/right
        for m, j in enumerate(j_top):
            line_pairs_def.append((bidx(i_left,  j), iidx(m)))
            line_pairs_def.append((iidx(m), bidx(i_right, j)))

    # Points array for defect lines (must match the def indexing layout)
    # Build it layer-by-layer so indices match the connectivity functions above.
    combined_def = np.empty((Nz * total_per_layer, 3), dtype=float)
    combined_perf = np.empty((Nz * total_per_layer, 3), dtype=float)  # for morphing lines if needed
    for k in range(Nz):
        # base slice
        bs = k * base_per_layer
        be = bs + base_per_layer
        # ins slice
        is_ = k * ins_per_layer
        ie_ = is_ + ins_per_layer

        cs = k * total_per_layer
        ce = cs + total_per_layer

        combined_def[cs:cs + base_per_layer] = base_def[bs:be]
        combined_def[cs + base_per_layer:ce] = ins_def[is_:ie_]

        combined_perf[cs:cs + base_per_layer] = base_perf[bs:be]
        combined_perf[cs + base_per_layer:ce] = ins_perf[is_:ie_]

    # Heuristic point size
    N_total = Nz * (base_per_layer + max(ins_per_layer, 0))
    point_size = point_size_from_counts(N_total)

    # no bonds for big sizes
    show_lines = (N_xy <= 28)

    return {
        "N_xy": N_xy,
        "Nx": Nx, "Ny": Ny, "Nz": Nz,
        "base_perf": base_perf,
        "base_def": base_def,
        "ins_perf": ins_perf,
        "ins_def": ins_def,
        "C_base": C_base,
        "C_ins": C_ins,
        "clim": (float(vmin), float(vmax)),
        "point_size": point_size,
        "show_lines": show_lines,
        "lines_perf_points": base_perf.copy(),  # base-only lines
        "lines_perf": _lines_from_pairs(line_pairs_perf),
        "lines_def_points_perf": combined_perf,  # combined layout (for morphing)
        "lines_def_points_def": combined_def,
        "lines_def": _lines_from_pairs(line_pairs_def),
    }


def set_actor_opacity(actor, opacity: float):
    if actor is None:
        return
    actor.GetProperty().SetOpacity(float(opacity))


def set_actor_point_size(actor, ps: float):
    if actor is None:
        return
    actor.GetProperty().SetPointSize(float(ps))


def add_points_actors(plotter: pv.Plotter, base_poly: pv.PolyData, ins_poly: pv.PolyData,
                      clim: tuple[float, float], point_size: float):
    # Blue (monochrome) actors
    a_blue_base = plotter.add_mesh(
        base_poly,
        color="tab:blue",
        style="points",
        render_points_as_spheres=True,
        point_size=point_size,
        opacity=1.0,
    )
    a_blue_ins = plotter.add_mesh(
        ins_poly,
        color="tab:blue",
        style="points",
        render_points_as_spheres=True,
        point_size=point_size,
        opacity=0.0,  # start hidden
    )

    # Colored actors (same polydata, different mapper settings)
    a_col_base = plotter.add_mesh(
        base_poly,
        scalars="strain",
        cmap="coolwarm",
        clim=clim,
        show_scalar_bar=False,
        style="points",
        render_points_as_spheres=True,
        point_size=point_size,
        opacity=0.0,  # start hidden
    )
    a_col_ins = plotter.add_mesh(
        ins_poly,
        scalars="strain",
        cmap="coolwarm",
        clim=clim,
        show_scalar_bar=False,
        style="points",
        render_points_as_spheres=True,
        point_size=point_size,
        opacity=0.0,  # start hidden
    )

    return a_blue_base, a_blue_ins, a_col_base, a_col_ins

def add_lines_actor(plotter: pv.Plotter, line_poly: pv.PolyData, opacity: float = 0.25):
    return plotter.add_mesh(
        line_poly,
        color="black",
        opacity=opacity,
        line_width=1.0,
    )


def update_camera_for_scene(plotter: pv.Plotter, bounds, scale: float | None = None):
    """Set camera scale without resetting orientation (avoids 'zoom wobble')."""
    if scale is None:
        scale = parallel_scale_from_bounds(bounds)
    plotter.camera.parallel_scale = float(scale)
    plotter.reset_camera_clipping_range()
def main():
    # Uncomment for headless Linux if needed:
    # pv.start_xvfb()

    sizes = [6, 8, 10, 12, 16, 20, 28, 38, 48, 68, 96] #, 128]

    # Output choice (three MP4 segments)
    out_mp4_1 = "edge_dislocation_animation_part1_intro_hold1.mp4"
    out_mp4_2 = "edge_dislocation_animation_part2_hold1_to_hold3.mp4"
    out_mp4_3 = "edge_dislocation_animation_part3_fadesizes_final.mp4"
    out_gif_1 = "edge_dislocation_animation_part1_intro_hold1.gif"
    out_gif_2 = "edge_dislocation_animation_part2_hold1_to_hold3.gif"
    out_gif_3 = "edge_dislocation_animation_part3_fadesizes_final.gif"

    # Frame pacing
    fps = 60
    hold_1 = 60      # perfect lattice hold
    morph_12 = 150    # perfect -> defect (monochrome)
    hold_2 = 40
    fade_23 = 90     # monochrome -> colored
    hold_3 = 40
    fade_sizes = 90  # crossfade between N_xy sizes
    hold_sizes = 0
    final_hold = 300   # 120 frames = 4 seconds at 30 fps

    # Build first scene (N_xy=6)
    s0 = build_scene(sizes[0])

    plotter = pv.Plotter(window_size=(1000, 900), off_screen=True)
    plotter.set_background("white")

    # Camera: set orientation once (do NOT reset per frame)
    plotter.enable_parallel_projection()
    plotter.view_xy()

    # base/ins polydata (points)
    base_poly = pv.PolyData(s0["base_perf"])
    ins_poly = pv.PolyData(s0["ins_perf"])
    base_poly["strain"] = s0["C_base"]
    ins_poly["strain"] = s0["C_ins"] if len(s0["C_ins"]) else np.zeros(ins_poly.n_points)

    # lines polydata
    line_perf_poly = pv.PolyData(s0["lines_perf_points"])
    line_perf_poly.lines = s0["lines_perf"]

    line_def_poly = pv.PolyData(s0["lines_def_points_perf"])
    line_def_poly.lines = s0["lines_def"]

    # Actors
    a_blue_base, a_blue_ins, a_col_base, a_col_ins = add_points_actors(
        plotter, base_poly, ins_poly, clim=s0["clim"], point_size=s0["point_size"]
    )

    a_lines_perf = add_lines_actor(plotter, line_perf_poly, opacity=0.25) if s0["show_lines"] else None
    a_lines_def  = add_lines_actor(plotter, line_def_poly,  opacity=0.00) if s0["show_lines"] else None
    # Between-layer lines (perfect lattice only; used in the intro to cue 3D depth)
    a_lines_z = None
    line_z_poly = None
    if s0["Nz"] > 1:
        Nx0, Ny0, Nz0 = s0["Nx"], s0["Ny"], s0["Nz"]
        base_per_layer0 = Nx0 * Ny0
        line_pairs_z = []
        for kk in range(Nz0 - 1):
            off0 = kk * base_per_layer0
            off1 = (kk + 1) * base_per_layer0
            for ii in range(base_per_layer0):
                line_pairs_z.append((off0 + ii, off1 + ii))
        line_z_poly = pv.PolyData(s0["base_perf"])
        line_z_poly.lines = _lines_from_pairs(line_pairs_z)
        a_lines_z = add_lines_actor(plotter, line_z_poly, opacity=0.12)

    # Camera bounds for (a) intro rotation and (b) the main sequence
    # We distinguish between the *perfect* framing (used right after the intro)
    # and the *defect-safe* framing (used after the morph).
    bnds_perf0 = scene_bounds([s0["base_perf"], s0["ins_perf"]])
    scale_perf0 = parallel_scale_from_bounds(bnds_perf0)

    bnds0 = scene_bounds([s0["base_def"], s0["ins_def"]])
    scale_def0 = parallel_scale_from_bounds(bnds0)

    intro_max_angle = 60.0
    bnds_intro = scene_bounds([
        rotate_y(s0["base_perf"], -intro_max_angle),
        rotate_y(s0["base_perf"],  intro_max_angle),
    ])

    # Start on the intro view (fits the rotated lattice)
    update_camera_for_scene(plotter, bnds_intro)

    # Open movie writer (mp4 preferred)
    writer = None
    out_files = []

    def open_writer(out_mp4: str, out_gif: str):
        nonlocal writer
        try:
            plotter.open_movie(out_mp4, framerate=fps, quality=8)
            writer = "mp4"
            out_files.append(out_mp4)
            print(f"[writer] MP4 -> {out_mp4}")
        except Exception as e:
            print(f"[writer] MP4 failed ({type(e).__name__}: {e}). Falling back to GIF.")
            plotter.open_gif(out_gif, fps=fps)
            writer = "gif"
            out_files.append(out_gif)
            print(f"[writer] GIF -> {out_gif}")

    def close_writer():
        """Finalize the current movie writer without closing the plotter."""
        nonlocal writer
        movie_writer = getattr(plotter, "_movie_writer", None)
        if movie_writer is not None:
            try:
                movie_writer.close()
            finally:
                plotter._movie_writer = None
        writer = None

    # Segment 1: intro + hold_1
    open_writer(out_mp4_1, out_gif_1)

    # Helper to write a frame
    def frame():
        plotter.render()
        plotter.write_frame()


    # ----------------------------

    # (0) Intro: rotate perfect lattice + highlight center layer
    # ----------------------------
    # Keep only monochrome perfect lattice visible
    set_actor_opacity(a_blue_base, 1.0)
    set_actor_opacity(a_blue_ins,  0.0)
    set_actor_opacity(a_col_base,  0.0)
    set_actor_opacity(a_col_ins,   0.0)
    if a_lines_perf is not None:
        set_actor_opacity(a_lines_perf, 0.25)
    if a_lines_def is not None:
        set_actor_opacity(a_lines_def, 0.0)
    if a_lines_z is not None:
        set_actor_opacity(a_lines_z, 0.12)

    # Center layer (middle z-slice of the perfect lattice)
    z_levels = np.unique(s0["base_perf"][:, 2])
    z0 = z_levels[len(z_levels) // 2]
    center_mask = np.isclose(s0["base_perf"][:, 2], z0)
    center_pts0 = s0["base_perf"][center_mask].copy()

    center_poly = pv.PolyData(center_pts0.copy())
    a_center = plotter.add_mesh(
        center_poly,
        color="tab:red",
        style="points",
        render_points_as_spheres=True,
        point_size=min(48.0, 1.8 * s0["point_size"]),
        opacity=0.0,
    )

    # Rotation angles (degrees)
    intro_angle_start = -50.0
    intro_angle_mark  =  50.0
    intro_angle_end   =   0.0

    # Camera elevation (degrees): start elevated so depth is obvious, end flat in view_xy
    intro_elev_start = 12.0
    intro_elev_end   = 0.0

    # Timing (rotation is intentionally ~2x slower than before)
    intro_rot_a         = 400   # start -> mark  (was 160)
    intro_hold_mark     = 20    # hold at mark
    intro_fade_highlight= 30    # highlight fade frames (end of phase A)
    intro_fade_others   = 40    # fade away non-center layers (after mark)
    intro_rot_b         = 240   # mark -> end    (was 120)
    intro_fade_in_others= 5    # bring full lattice back once in focus

    # Conservative intro framing (we'll keep scale fixed during the intro)
    intro_pts_a = rotate_y(s0["base_perf"], intro_angle_start)
    intro_pts_b = rotate_y(s0["base_perf"], intro_angle_mark)
    intro_pts_ai = rotate_y(s0["ins_perf"], intro_angle_start) if len(s0["ins_perf"]) else None
    intro_pts_bi = rotate_y(s0["ins_perf"], intro_angle_mark) if len(s0["ins_perf"]) else None
    intro_bnds = scene_bounds([intro_pts_a, intro_pts_b, intro_pts_ai, intro_pts_bi, center_pts0])

    xmin, xmax, ymin, ymax, zmin, zmax = intro_bnds
    maxr = max(xmax - xmin, ymax - ymin, zmax - zmin)
    intro_scale = 0.85 * 0.5 * maxr
    intro_dist = 6.0 * maxr + 1.0  # distance doesn't change scale (parallel proj), but helps clipping

    def set_intro_camera(elev_deg: float, scale: float | None = None):
        ang = np.deg2rad(float(elev_deg))
        plotter.camera.focal_point = (0.0, 0.0, 0.0)
        plotter.camera.position = (0.0, intro_dist * np.sin(ang), intro_dist * np.cos(ang))
        plotter.camera.up = (0.0, 1.0, 0.0)
        if scale is None:
            scale = intro_scale
        plotter.camera.parallel_scale = float(scale)
        plotter.reset_camera_clipping_range()

    def scale_from_points_xy(pts: np.ndarray) -> float:
        """Orthographic zoom needed to fit the current points in the XY view."""
        xmin = float(np.min(pts[:, 0])); xmax = float(np.max(pts[:, 0]))
        ymin = float(np.min(pts[:, 1])); ymax = float(np.max(pts[:, 1]))
        xr = xmax - xmin
        yr = ymax - ymin
        return 0.60 * 0.5 * max(xr, yr)

    # Start with elevated camera
    set_intro_camera(intro_elev_start)

    # Phase A: rotate start -> mark, fade highlight in near the end
    for k in range(intro_rot_a):
        t = k / (intro_rot_a - 1)
        ang = intro_angle_start + (intro_angle_mark - intro_angle_start) * t

        base_poly.points = rotate_y(s0["base_perf"], ang)
        ins_poly.points  = rotate_y(s0["ins_perf"],  ang)
        center_poly.points = rotate_y(center_pts0, ang)

        # bonds follow the base lattice points
        line_perf_poly.points = base_poly.points
        if line_z_poly is not None:
            line_z_poly.points = base_poly.points

        if k >= intro_rot_a - intro_fade_highlight:
            tt = (k - (intro_rot_a - intro_fade_highlight)) / max(intro_fade_highlight - 1, 1)
            set_actor_opacity(a_center, smoothstep(tt))
        else:
            set_actor_opacity(a_center, 0.0)

        frame()

    # Ensure we are fully highlighted at the mark angle
    set_actor_opacity(a_center, 1.0)

    # Hold at the highlight angle
    for _ in range(intro_hold_mark):
        frame()

    # Fade away ALL other layers (keep only the highlighted center layer)
    # (We simply fade the full lattice actor out; the center layer stays visible via a_center.)
    for k in range(intro_fade_others):
        t = k / (intro_fade_others - 1)
        f = smoothstep(t)
        set_actor_opacity(a_blue_base, 1.0 - f)
        if a_lines_perf is not None:
            set_actor_opacity(a_lines_perf, 0.25 * (1.0 - f))
        if a_lines_z is not None:
            set_actor_opacity(a_lines_z, 0.12 * (1.0 - f))
        frame()

    # Phase B: rotate mark -> end, while flattening the camera elevation into the main view
    # Keep the center layer visible, then crossfade back to the full lattice near the end.
    # Also zoom in so that the final framing matches the main sequence.
    t_return0 = 0.55  # fraction of phase B where we start bringing back the full lattice (0..1)
    t_zoom0 = 0.55    # zoom starts here (kept late to avoid cropping while the lattice is still rotated)

    for k in range(intro_rot_b):
        t = k / (intro_rot_b - 1)
        ang = intro_angle_mark + (intro_angle_end - intro_angle_mark) * t

        base_poly.points = rotate_y(s0["base_perf"], ang)
        ins_poly.points  = rotate_y(s0["ins_perf"],  ang)
        center_poly.points = rotate_y(center_pts0, ang)

        # bonds follow the base lattice points
        line_perf_poly.points = base_poly.points
        if line_z_poly is not None:
            line_z_poly.points = base_poly.points

        # camera elevation: elevated -> flat
        elev = (1.0 - t) * intro_elev_start + t * intro_elev_end

        # Zoom: intro_scale -> *perfect* framing, but never crop the currently rotated lattice
        uz = (t - t_zoom0) / max(1e-12, (1.0 - t_zoom0))
        uz = float(np.clip(uz, 0.0, 1.0))
        fz = smoothstep(uz)
        scale_zoom = (1.0 - fz) * intro_scale + fz * scale_perf0
        scale_need = scale_from_points_xy(base_poly.points)
        set_intro_camera(elev, scale=max(scale_need, scale_zoom))

        # Crossfade: center layer -> full lattice (no "empty" gap)
        u = (t - t_return0) / max(1e-12, (1.0 - t_return0))
        u = float(np.clip(u, 0.0, 1.0))
        f_return = smoothstep(u)

        # Bring back full lattice + lines
        set_actor_opacity(a_blue_base, f_return)
        if a_lines_perf is not None:
            set_actor_opacity(a_lines_perf, 0.25 * f_return)
        if a_lines_z is not None:
            set_actor_opacity(a_lines_z, 0.12 * f_return)

        # Keep center strong until return begins, then fade it out as the full lattice appears
        if t < t_return0:
            set_actor_opacity(a_center, 1.0)
        else:
            set_actor_opacity(a_center, 1.0 - f_return)

        frame()

    # Restore unrotated perfect lattice
    base_poly.points = s0["base_perf"]
    ins_poly.points  = s0["ins_perf"]
    center_poly.points = center_pts0
    line_perf_poly.points = base_poly.points
    if line_z_poly is not None:
        line_z_poly.points = base_poly.points
    set_actor_opacity(a_center, 0.0)

    # Ensure we end the intro exactly in the main-sequence framing (flat view + correct zoom)
    # (Avoids a visible zoom "jump" right before scene (1) begins.)
    # End the intro exactly in the framing we use for the next scene (perfect lattice)
    set_intro_camera(0.0, scale_perf0)

    # Bring the full lattice back once we're in focus (start from current opacity; avoids a sudden vanish)
    start_base_op = float(a_blue_base.GetProperty().GetOpacity())
    start_lines_op = float(a_lines_perf.GetProperty().GetOpacity()) if a_lines_perf is not None else 0.0
    start_linesz_op = float(a_lines_z.GetProperty().GetOpacity()) if a_lines_z is not None else 0.0

    for k in range(intro_fade_in_others):
        t = k / (intro_fade_in_others - 1)
        f = smoothstep(t)
        set_actor_opacity(a_blue_base, start_base_op + (1.0 - start_base_op) * f)
        if a_lines_perf is not None:
            set_actor_opacity(a_lines_perf, start_lines_op + (0.25 - start_lines_op) * f)
        if a_lines_z is not None:
            set_actor_opacity(a_lines_z, start_linesz_op + (0.12 - start_linesz_op) * f)
        frame()

    # Hide the between-layer lines for the main (2D-looking) sequence
    if a_lines_z is not None:
        set_actor_opacity(a_lines_z, 0.0)
        a_lines_z.SetVisibility(False)

    # (1) Perfect lattice hold
    # ----------------------------
    for _ in range(hold_1):
        frame()

    # Close segment 1 and start segment 2
    close_writer()
    open_writer(out_mp4_2, out_gif_2)

    # ----------------------------
    # (2) Morph perfect -> defect (monochrome)
    # - atoms move
    # - inserted atoms fade in
    # - bonds crossfade perfect->defect
    # ----------------------------
    for k in range(morph_12):
        t = k / (morph_12 - 1)

        # Camera: zoom out so the morph never crops (perfect framing -> defect-safe framing)
        cam_scale = (1.0 - t) * scale_perf0 + t * scale_def0
        update_camera_for_scene(plotter, bnds0, scale=cam_scale)

        # points
        base_poly.points = lerp(s0["base_perf"], s0["base_def"], t)
        ins_poly.points  = lerp(s0["ins_perf"],  s0["ins_def"],  t)

        # lines: perfect uses base points only
        line_perf_poly.points = base_poly.points

        # defect lines use combined layout; rebuild combined from current base/ins
        # (layer ordering: base then ins, per layer)
        Nx, Ny, Nz = s0["Nx"], s0["Ny"], s0["Nz"]
        base_per_layer = Nx * Ny
        ins_per_layer = ins_poly.n_points // Nz
        total_per_layer = base_per_layer + ins_per_layer

        combined = np.empty((Nz * total_per_layer, 3), dtype=float)
        for kk in range(Nz):
            bs = kk * base_per_layer
            be = bs + base_per_layer
            is_ = kk * ins_per_layer
            ie_ = is_ + ins_per_layer
            cs = kk * total_per_layer
            combined[cs:cs + base_per_layer] = base_poly.points[bs:be]
            combined[cs + base_per_layer:cs + total_per_layer] = ins_poly.points[is_:ie_]
        line_def_poly.points = combined

        # opacities
        set_actor_opacity(a_blue_ins, t)
        if a_lines_perf is not None and a_lines_def is not None:
            set_actor_opacity(a_lines_perf, 0.25 * (1.0 - t))
            set_actor_opacity(a_lines_def,  0.25 * t)

        frame()

    # ensure defect lines visible after morph
    if a_lines_perf is not None and a_lines_def is not None:
        set_actor_opacity(a_lines_perf, 0.0)
        set_actor_opacity(a_lines_def, 0.25)

    for _ in range(hold_2):
        frame()

    # ----------------------------
    # (3) Crossfade monochrome -> colored
    # ----------------------------
    for k in range(fade_23):
        t = k / (fade_23 - 1)
        set_actor_opacity(a_blue_base, 1.0 - t)
        set_actor_opacity(a_blue_ins,  1.0 - t)
        set_actor_opacity(a_col_base,  t)
        set_actor_opacity(a_col_ins,   t)
        frame()

    # lock final opacities for colored
    set_actor_opacity(a_blue_base, 0.0)
    set_actor_opacity(a_blue_ins,  0.0)
    set_actor_opacity(a_col_base,  1.0)
    set_actor_opacity(a_col_ins,   1.0)

    for _ in range(hold_3):
        frame()

    # Close segment 2 and start segment 3
    close_writer()
    open_writer(out_mp4_3, out_gif_3)

    # ----------------------------
    # (4) Step through sizes: crossfade between N_xy scenes
    # ----------------------------
    prev_bundle = {
        "base_poly": base_poly,
        "ins_poly": ins_poly,
        "a_col_base": a_col_base,
        "a_col_ins": a_col_ins,
        "a_lines": a_lines_def if s0["show_lines"] else None,
        "scene": s0,
    }

    for N_xy in sizes[1:]:
        s = build_scene(N_xy)

        # New polydata + actors (start invisible)
        base_poly_n = pv.PolyData(s["base_def"])
        ins_poly_n  = pv.PolyData(s["ins_def"])
        base_poly_n["strain"] = s["C_base"]
        ins_poly_n["strain"]  = s["C_ins"] if len(s["C_ins"]) else np.zeros(ins_poly_n.n_points)

        a_col_base_n = plotter.add_mesh(
            base_poly_n, scalars="strain", cmap="coolwarm", clim=s["clim"], show_scalar_bar=False,
            style="points", render_points_as_spheres=True, point_size=s["point_size"], opacity=0.0
        )
        a_col_ins_n = plotter.add_mesh(
            ins_poly_n, scalars="strain", cmap="coolwarm", clim=s["clim"], show_scalar_bar=False,
            style="points", render_points_as_spheres=True, point_size=s["point_size"], opacity=0.0
        )

        # Lines for this size (optional)
        a_lines_n = None
        line_poly_n = None
        if s["show_lines"]:
            line_poly_n = pv.PolyData(s["lines_def_points_def"])
            line_poly_n.lines = s["lines_def"]
            a_lines_n = add_lines_actor(plotter, line_poly_n, opacity=0.0)

        # Camera target
        bnds = scene_bounds([s["base_def"], s["ins_def"]])
        target_scale = parallel_scale_from_bounds(bnds)

        # Crossfade
        start_scale = float(plotter.camera.parallel_scale)
        
        for k in range(fade_sizes):
            t = k / (fade_sizes - 1)

            # Always keep color intensity constant
            set_actor_opacity(prev_bundle["a_col_base"], 1.0)
            set_actor_opacity(prev_bundle["a_col_ins"],  1.0)
            set_actor_opacity(a_col_base_n,              1.0)
            set_actor_opacity(a_col_ins_n,               1.0)

            # Switch which one is visible (no transparency blending)
            show_prev = (t < 0.5)
            prev_bundle["a_col_base"].SetVisibility(show_prev)
            prev_bundle["a_col_ins"].SetVisibility(show_prev)
            a_col_base_n.SetVisibility(not show_prev)
            a_col_ins_n.SetVisibility(not show_prev)

            # (Optional) fade the lines, because they’re not your color signal
            if prev_bundle["a_lines"] is not None:
                set_actor_opacity(prev_bundle["a_lines"], 0.25 * (1.0 - t))
            if a_lines_n is not None:
                set_actor_opacity(a_lines_n, 0.25 * t)


            # camera scale morph (linear, no view reset)
            scale = (1.0 - t) * start_scale + t * target_scale
            update_camera_for_scene(plotter, bnds, scale=scale)

            frame()

        # Remove old actors to keep memory sane
        plotter.remove_actor(prev_bundle["a_col_base"])
        plotter.remove_actor(prev_bundle["a_col_ins"])
        if prev_bundle["a_lines"] is not None:
            plotter.remove_actor(prev_bundle["a_lines"])

        prev_bundle = {
            "base_poly": base_poly_n,
            "ins_poly": ins_poly_n,
            "a_col_base": a_col_base_n,
            "a_col_ins": a_col_ins_n,
            "a_lines": a_lines_n,
            "scene": s,
        }

        for _ in range(hold_sizes):
            frame()

    # ----------------------------
    # Final hold
    # ----------------------------

    for _ in range(final_hold):
        plotter.write_frame()

    plotter.close()

    if out_files:
        print("Done. Wrote:")
        for f in out_files:
            print(f"  - {f}")


if __name__ == "__main__":
    main()