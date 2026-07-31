"""
Visual binary orbital-element solver (Thiele-Innes method), refactored as a
pure library: no module-level globals, no disk writes, no import-time work.

The single entry point is `run_fit(...)`, which takes an astrometry table plus
configuration and returns a `FitResult` holding the fitted array, the accepted
family, the three Matplotlib figures, and a text log. Callers (a CLI, a
Streamlit app, a test) decide what to do with those.

Column layout of every fitted row / the fitted_values array:
    0=P  1=T  2=e  3=a"  4=i(deg)  5=Omega(deg)  6=omega(deg)
    7=M_total  8=cost  9=R2
"""

from dataclasses import dataclass, field
import io

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


# ======================================================================
# Data container
# ======================================================================
@dataclass
class FitResult:
    fitted_values: np.ndarray            # (N, 10) all successful grid fits
    accept: np.ndarray                   # (M, 10) cost- (+mass-) accepted subset
    best_fit: np.ndarray                 # global lowest-cost row
    best_accept_fit: np.ndarray          # lowest-cost row within `accept`
    orbit_fig: "plt.Figure"              # sky + true orbit panels of best_accept_fit
    period_fig: "plt.Figure"             # cost vs period
    sma_fig: "plt.Figure"                # cost vs semi-major axis
    eccent_fig: "plt.Figure"             # cost vs eccentricity
    log_text: str                        # human-readable run log
    ranges: dict                         # {label: (min, max, median)} over accept
    d_pc: float
    parallax_mas: float
    n_accept: int
    n_total: int


# ======================================================================
# Orbital mechanics  (all pure functions, angles in radians unless noted)
# ======================================================================
def solve_kepler(M, e, tol=1e-12, itmax=100):
    """Solve Kepler's equation M = E - e sin E (vectorized Newton)."""
    M = np.mod(M, 2 * np.pi)
    E = np.where(e < 0.8, M, np.pi * np.ones_like(M))
    for _ in range(itmax):
        denom = 1 - e * np.cos(E)
        dE = (E - e * np.sin(E) - M) / denom
        E -= dE
        if np.max(np.abs(dE)) < tol:
            break
    return E


def thiele_innes(a, i, Omega, omega):
    """Campbell elements -> Thiele-Innes constants A, B, F, G."""
    cO, sO = np.cos(Omega), np.sin(Omega)
    co, so = np.cos(omega), np.sin(omega)
    ci = np.cos(i)
    A = a * (co * cO - so * sO * ci)
    B = a * (co * sO + so * cO * ci)
    F = a * (-so * cO - co * sO * ci)
    G = a * (-so * sO + co * cO * ci)
    return A, B, F, G


def model_xy(params, t):
    """Predicted sky position (x=N, y=E) at epochs t for the 7 elements."""
    P, T, e, a, i, Omega, omega = params
    M = 2 * np.pi * (t - T) / P
    E = solve_kepler(M, e)
    X = np.cos(E) - e
    Y = np.sqrt(1 - e**2) * np.sin(E)
    A, B, F, G = thiele_innes(a, i, Omega, omega)
    x = A * X + F * Y   # North
    y = B * X + G * Y   # East
    return x, y


def residuals(params, t, x, y):
    xm, ym = model_xy(params, t)
    return np.concatenate([xm - x, ym - y])


# ======================================================================
# Data loading
# ======================================================================
def load_astrometry(source):
    """
    Read a CSV with columns THETA(deg), RHO("), YEAR, PARALLAX(mas, first row).

    `source` may be a path or any file-like object (e.g. a Streamlit upload).
    Returns (theta_rad, rho, t, x_obs, y_obs, parallax_mas).
    """
    # np.genfromtxt consumes file-like objects once, so read bytes and reuse.
    if hasattr(source, "read"):
        raw = source.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        buf1 = io.StringIO(raw)
        buf2 = io.StringIO(raw)
    else:
        buf1, buf2 = source, source

    data = np.genfromtxt(buf1, delimiter=",", skip_header=1, usecols=(0, 1, 2))
    theta = np.deg2rad(data[:, 0])
    rho = data[:, 1]
    t = data[:, 2]
    x_obs = rho * np.cos(theta)   # North
    y_obs = rho * np.sin(theta)   # East

    first_row = np.genfromtxt(buf2, delimiter=",", skip_header=1, max_rows=1)
    parallax_mas = float(first_row[3])
    return theta, rho, t, x_obs, y_obs, parallax_mas


# ======================================================================
# Fitting
# ======================================================================
def _bounds(P_lower, P_upper):
    lower = [P_lower, 0.0, 0.0, 1e-3, 0.0, 0.0, 0.0]
    upper = [P_upper, 3000.0, 0.999, 100.0, np.pi, 2 * np.pi, 2 * np.pi]
    return lower, upper


def fit_six_at_fixed_P(P_fixed, n_restarts, rng, t_obs, x_obs, y_obs,
                       lower, upper):
    """Fit the six non-period elements with P pinned by a razor-thin bound."""
    eps = max(P_fixed * 1e-9, 1e-9)
    lo = [P_fixed - eps] + lower[1:]
    hi = [P_fixed + eps] + upper[1:]

    local_best = None
    for _ in range(n_restarts):
        p0 = [
            P_fixed,
            rng.uniform(0.0, 3000.0),      # T
            rng.uniform(0.0, 0.95),        # e
            rng.uniform(0.1, 60.0),        # a (arcsec)
            rng.uniform(0.0, np.pi),       # i
            rng.uniform(0.0, 2 * np.pi),   # Omega
            rng.uniform(0.0, 2 * np.pi),   # omega
        ]
        try:
            sol = least_squares(residuals, p0, args=(t_obs, x_obs, y_obs),
                                bounds=(lo, hi), method="trf",
                                x_scale="jac", max_nfev=2000)
        except Exception:
            continue
        if local_best is None or sol.cost < local_best.cost:
            local_best = sol
    return local_best


def record(sol, t_obs, x_obs, y_obs, d_pc, ss_tot):
    P, T, e, a, i, Omega, omega = sol.x
    r = residuals(sol.x, t_obs, x_obs, y_obs)
    ss_res = np.sum(r**2)
    r_squared = 1 - ss_res / ss_tot
    a_AU = a * d_pc
    M_total = a_AU**3 / P**2
    return [P, T, e, a, np.degrees(i), np.degrees(Omega),
            np.degrees(omega), M_total, sol.cost, r_squared]


# ======================================================================
# Plotting  (each returns a Figure; no disk writes)
# ======================================================================
def plot_orbits(row, t_obs, x_obs, y_obs, d_pc, parallax_mas, target,
                n_P_grid, n_restarts_per_P, mass_constrain, unit='"'):
    """Two-panel sky-projected + true orbit figure for one fitted row."""
    P, T, e, a, i_deg, Omega_deg, omega_deg, M_total, cost, r_squared = row
    i, Omega, omega = np.radians(i_deg), np.radians(Omega_deg), np.radians(omega_deg)

    E_dense = np.linspace(0, 2 * np.pi, 2000)
    X_d = np.cos(E_dense) - e
    Y_d = np.sqrt(1 - e**2) * np.sin(E_dense)
    A_, B_, F_, G_ = thiele_innes(a, i, Omega, omega)
    x_fit = A_ * X_d + F_ * Y_d
    y_fit = B_ * X_d + G_ * Y_d

    x_peri, y_peri = A_ * (1 - e), B_ * (1 - e)
    x_apo, y_apo = A_ * (-1 - e), B_ * (-1 - e)
    peri_sky = np.hypot(x_peri, y_peri)
    apo_sky = np.hypot(x_apo, y_apo)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 7), layout="constrained")
    ax1.set_aspect("equal")
    ax2.set_aspect("equal")

    ax1.scatter(0, 0, color="red", s=180, marker="o", label="Primary", zorder=7)
    ax1.scatter(x_obs[-1], y_obs[-1], color="orange", s=120, marker="o",
                label="Secondary", zorder=6)
    ax1.scatter(x_obs, y_obs, c="tab:blue", marker="x", s=25,
                label="Observations", zorder=5)
    ax1.scatter(x_peri, y_peri, color="green", marker="x", s=120,
                label="Periastron", zorder=4)
    ax1.scatter(x_apo, y_apo, color="purple", marker="x", s=120,
                label="Apastron", zorder=3)

    # Line of nodes
    normal = np.array([-np.sin(Omega), np.cos(Omega)])
    node_vals = normal[0] * x_fit + normal[1] * y_fit
    crossings = np.where(np.sign(node_vals[:-1]) != np.sign(node_vals[1:]))[0]
    node_points = []
    for j in crossings:
        tt = node_vals[j] / (node_vals[j] - node_vals[j + 1])
        node_points.append((x_fit[j] + tt * (x_fit[j + 1] - x_fit[j]),
                            y_fit[j] + tt * (y_fit[j + 1] - y_fit[j])))
    if len(node_points) >= 2:
        node_x = np.array([node_points[0][0], node_points[1][0]])
        node_y = np.array([node_points[0][1], node_points[1][1]])
    else:
        node_x = np.array([-a, a]) * np.cos(Omega) * 1.2
        node_y = np.array([-a, a]) * np.sin(Omega) * 1.2
    ax1.plot(node_x, node_y, "--", color="gray", lw=0.8,
             label="Line of nodes", zorder=2)

    ax1.plot(x_fit, y_fit, "k-", lw=1.2, zorder=1)
    ax1.set_xlabel(f"$\\rightarrow$N ({unit})")
    ax1.set_ylabel(f"$\\rightarrow$E ({unit})")
    ax1.grid(True, alpha=0.3)
    fig.legend(fontsize=8, ncol=6, loc="outside lower center",
               frameon=False, borderaxespad=0.2)

    # True orbit panel
    a_AU = a * d_pc
    M_total = a_AU**3 / P**2
    x_true = a_AU * (np.cos(E_dense) - e)
    y_true = a_AU * np.sqrt(1 - e**2) * np.sin(E_dense)
    ax2.plot(x_true, y_true, "k-", lw=1.2, label="True Orbit", zorder=1)
    ax2.grid(True, alpha=0.3)
    ax2.scatter(0, 0, color="red", s=180, marker="o", label="Primary", zorder=5)
    ax2.scatter(a_AU * (1 - e), 0, color="green", marker="x", s=120,
                label="Periastron", zorder=3)
    ax2.scatter(a_AU * (-1 - e), 0, color="purple", marker="x", s=120,
                label="Apastron", zorder=3)

    # Observations mapped back into the true plane (guard edge-on singularity)
    det = A_ * G_ - B_ * F_
    if abs(det) < 1e-12 * max(a * a, 1e-12):
        X_obs = np.full_like(x_obs, np.nan)
        Y_obs = np.full_like(x_obs, np.nan)
    else:
        X_obs = (G_ * x_obs - F_ * y_obs) / det
        Y_obs = (-B_ * x_obs + A_ * y_obs) / det
    X_obs_AU = a_AU * X_obs
    Y_obs_AU = a_AU * Y_obs
    true_anom = np.arctan2(Y_obs_AU[-1], X_obs_AU[-1])

    ax2.scatter(X_obs_AU, Y_obs_AU, c="tab:blue", marker="x", s=25,
                label="Observations", zorder=4)
    ax2.scatter(X_obs_AU[-1], Y_obs_AU[-1], color="orange", s=120, marker="o",
                label="Secondary", zorder=5)
    ax2.set_xlabel("X (AU)")
    ax2.set_ylabel("Y (AU)")

    T_disp = t_obs[0] - np.mod(t_obs[0] - T, P)

    fig.suptitle(
        f'{target}\n'
        f'Obs Arc: {t_obs[0]:.0f} - {t_obs[-1]:.0f} | Orbit Fits = {n_P_grid}, Iterations = {n_restarts_per_P}\n'
        f'R² = {r_squared}, Cost = {cost}, Mass Constrain = {mass_constrain}\n\n'
        f'Parallax = {parallax_mas:.4f} mas, Distance = {d_pc:.2f} pc, $M_{{total}}$ = {M_total:.3f} M$_\\odot$\n'
        f'P = {P:.3f} yr, T = {T_disp:.3f} yr'
        , fontsize=10)

    ax1.set_title(
        f'Sky-Projected Orbit Fit\n'
        f'a = {a:.3f}{unit}, e = {e:.3f}, '
        f'i = {i_deg:.3f}$^\\circ$, $\\Omega$ = {Omega_deg:.3f}$^\\circ$, '
        f'$\\omega$ = {omega_deg:.3f}$^\\circ$\n'
        f'Apastron = {apo_sky:.3f}{unit}, Periastron = {peri_sky:.3f}{unit}',
        fontsize=9)
    ax2.set_title(
        f'True Orbit Fit\n'
        f'a = {a_AU:.3f} AU, e = {e:.3f}, '
        f'$\\nu$ = {np.degrees(true_anom):.3f}$^\\circ$\n'
        f'Apastron = {a_AU * (1 + e):.3f} AU, '
        f'Periastron = {a_AU * (1 - e):.3f} AU', fontsize=9)
    return fig


def _cost_title(target, t_obs, fit_mode, n_fitted, n_restarts_per_P,
                mass_constrain, n_accept, accept_factor, best_cost,
                P_lower, P_upper, m_total_frac_accept, m_total_guess):
    """Reproduce the exact multi-line cost-plot title from orbit3.py."""
    if mass_constrain == False:
        return (f'{target} | {t_obs[0]} - {t_obs[-1]}\n'
                f'mode = {fit_mode} | {n_fitted} Orbit Fits, {n_restarts_per_P} Iterations, Mass Constrain = {mass_constrain}\n'
                f'{n_accept}/{n_fitted} Accepted, Cost $\\leq$ {accept_factor * best_cost}\n'
                f'{P_lower} $\\leq$ P $\\leq$ {P_upper}')
    else:
        return (f'{target} | {t_obs[0]} - {t_obs[-1]}\n'
                f'mode = {fit_mode} | {n_fitted} Orbit Fits, {n_restarts_per_P} Iterations, Mass Constrain = {mass_constrain}\n'
                f'{n_accept}/{n_fitted} Accepted, Cost $\\leq$ {accept_factor * best_cost}\n'
                f'{P_lower} $\\leq$ P $\\leq$ {P_upper}, {(1-m_total_frac_accept) *  m_total_guess:.3f} M$_\\odot$ $\\leq$ $M_{{total}}$ $\\leq$ {(1+m_total_frac_accept) *  m_total_guess:.3f} M$_\\odot$')


def _cost_scatter(x_all, x_acc, costs, acc_costs, best_x, thresh,
                  xlabel, best_label, title, accept_factor, P_grid_log):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cost")
    ax.set_title(title)
    ax.scatter(x_all, costs, c="tab:blue", s=30, alpha=0.7, marker="x",
               label="Modeled Orbits")
    ax.scatter(x_acc, acc_costs, c="tab:red", s=30, alpha=0.7, marker="x",
               label="Accepted Orbits")
    ax.axhline(thresh, color="tab:red", ls="--", lw=1,
               label=f"Accept = {accept_factor}x Best Cost")
    ax.axvline(best_x, color="tab:green", ls=":", lw=1, label=best_label)
    ax.set_yscale("log")
    if P_grid_log:
        ax.set_xscale("log")
    ax.legend(fontsize=8)
    return fig


# ======================================================================
# Main entry point
# ======================================================================
def run_fit(source, *, target="Target", m1_guess=None, m2_guess=None,
            P_lower=100.0, P_upper=2000.0, n_P_grid=100, n_restarts_per_P=10,
            P_grid_log=False, accept_factor=1.5, m_total_frac_accept=0.1,
            seed=None, unit='"', progress=None):
    """
    Fit a visual binary orbit and return a FitResult.

    `source`   : path or file-like CSV (THETA, RHO, YEAR, PARALLAX).
    `progress` : optional callable(fraction_done) for UI progress bars.
    Angles/masses follow the column layout documented at module top.
    """
    theta, rho, t_obs, x_obs, y_obs, parallax_mas = load_astrometry(source)
    d_pc = 1.0 / (parallax_mas / 1000.0)
    ss_tot = np.sum((x_obs - x_obs.mean())**2) + np.sum((y_obs - y_obs.mean())**2)

    mass_constrain = m1_guess is not None and m2_guess is not None
    m_total_guess = (m1_guess + m2_guess) if mass_constrain else None

    lower, upper = _bounds(P_lower, P_upper)
    rng = np.random.default_rng(seed)

    if P_grid_log:
        P_values = np.geomspace(P_lower, P_upper, n_P_grid)
    else:
        P_values = np.linspace(P_lower, P_upper, n_P_grid)

    log = []
    log.append(f"Grid mode: {n_P_grid} periods in [{P_lower}, {P_upper}] yr, "
               f"{n_restarts_per_P} restarts each, "
               f"Mass Constrain = {mass_constrain}")
    log.append(f"Accept factor = {accept_factor}, "
               f"Mass fractional acceptance = {m_total_frac_accept}, seed = {seed}")

    fitted_values = []
    for gi, P_fixed in enumerate(P_values):
        sol = fit_six_at_fixed_P(P_fixed, n_restarts_per_P, rng,
                                 t_obs, x_obs, y_obs, lower, upper)
        if sol is not None:
            fitted_values.append(record(sol, t_obs, x_obs, y_obs, d_pc, ss_tot))
        if progress is not None:
            progress((gi + 1) / len(P_values))

    if not fitted_values:
        raise RuntimeError("No successful fits — check data and bounds.")

    fitted_values = np.array(fitted_values, dtype=float)
    best_idx = int(np.argmin(fitted_values[:, 8]))
    best_fit = fitted_values[best_idx]
    best_cost = best_fit[8]

    accept = fitted_values[fitted_values[:, 8] <= accept_factor * best_cost]
    if mass_constrain:
        accept = accept[accept[:, 7] <= (1 + m_total_frac_accept) * m_total_guess]
        accept = accept[accept[:, 7] >= (1 - m_total_frac_accept) * m_total_guess]

    if len(accept) == 0:
        log.append("No orbits pass the cost + mass constraints. "
                   "Widen accept_factor or m_total_frac_accept. "
                   "Falling back to lowest-cost orbit.")
        best_accept_fit = best_fit
    else:
        best_accept_fit = accept[int(np.argmin(accept[:, 8]))]

    # Range report
    labels = ['P (yr)', 'T (yr)', 'e', f'a ({unit})', 'i (deg)',
              'Omega (deg)', 'omega (deg)', 'M_tot (Msun)']
    ranges = {}
    if len(accept) > 0:
        log.append(f"ACCEPTABLE ORBITS: {len(accept)} of {len(fitted_values)}")
        for c, lab in enumerate(labels):
            col = accept[:, c]
            ranges[lab] = (float(col.min()), float(col.max()), float(np.median(col)))
            log.append(f"  {lab:14s} range [{col.min():10.3f}, "
                       f"{col.max():10.3f}]  median {np.median(col):10.3f}")

    # Figures  (titles/labels reproduced EXACTLY from orbit3.py)
    thresh = accept_factor * best_cost
    ttl = _cost_title(target, t_obs, "grid", len(fitted_values),
                      n_restarts_per_P, mass_constrain, len(accept),
                      accept_factor, best_cost, P_lower, P_upper,
                      m_total_frac_accept, m_total_guess)

    period_fig = _cost_scatter(
        fitted_values[:, 0], accept[:, 0],
        fitted_values[:, 8], accept[:, 8],
        best_accept_fit[0], thresh, "Orbital Period (yr)",
        f"Best Period = {best_accept_fit[0]} yr", ttl,
        accept_factor, P_grid_log)

    sma_fig = _cost_scatter(
        fitted_values[:, 3], accept[:, 3],
        fitted_values[:, 8], accept[:, 8],
        best_accept_fit[3], thresh, f"Semi-Major Axis ({unit})",
        f"Best Semi-Major Axis = {best_accept_fit[3]}{unit}", ttl,
        accept_factor, P_grid_log)

    eccent_fig = _cost_scatter(
        fitted_values[:, 2], accept[:, 2],
        fitted_values[:, 8], accept[:, 8],
        best_accept_fit[2], thresh, "Eccentricity",
        f"Best Eccentricity = {best_accept_fit[2]}", ttl,
        accept_factor, P_grid_log)

    orbit_fig = plot_orbits(best_accept_fit, t_obs, x_obs, y_obs, d_pc,
                            parallax_mas, target, n_P_grid, n_restarts_per_P,
                            mass_constrain, unit)

    return FitResult(
        fitted_values=fitted_values, accept=accept, best_fit=best_fit,
        best_accept_fit=best_accept_fit, orbit_fig=orbit_fig,
        period_fig=period_fig, sma_fig=sma_fig, eccent_fig=eccent_fig,
        log_text="\n".join(log), ranges=ranges, d_pc=d_pc,
        parallax_mas=parallax_mas, n_accept=len(accept),
        n_total=len(fitted_values))