"""
frontier_audit.core: Iso-Frontier sensitivity diagnostics and differential geometry tools.

Implements:
1. Bivariate spline response-surface modeling (PPL, Entropy, Secondary metrics).
2. Level curve and joint iso-frontier numerical tracing.
3. Local gradient and arc-length tangent vector estimation.
4. Jacobian rank-2 test for metric functional independence.
5. Arc-length sensitivity metric: kappa_m = dm / dl|_C.
6. Multi-seed bootstrap confidence intervals for kappa_m.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.interpolate as interpolate
import scipy.optimize as optimize


@dataclass
class IsoCurveResult:
    p_target: float
    s: np.ndarray
    g: np.ndarray
    arc_length: np.ndarray
    total_arc_length: float
    tangents: np.ndarray  # (n_pts, 2) unit tangent vectors [u_s, u_g]
    ppl_values: np.ndarray
    h_values: np.ndarray

    def to_dict(self) -> dict:
        return {
            "p_target": self.p_target,
            "s": self.s.tolist(),
            "g": self.g.tolist(),
            "arc_length": self.arc_length.tolist(),
            "total_arc_length": self.total_arc_length,
            "tangents": self.tangents.tolist(),
            "ppl_values": self.ppl_values.tolist(),
            "h_values": self.h_values.tolist(),
        }


@dataclass
class JacobianTestResult:
    metric_name: str
    s: float
    g: float
    grad_p: List[float]
    grad_h: List[float]
    grad_m: List[float]
    singular_values_J: List[float]
    singular_values_primary: List[float]
    primary_condition_number: float
    unit_tangent: List[float]
    kappa_m: float
    ortho_ratio: float
    is_null: bool  # True if ortho_ratio < threshold (functionally dependent)
    tangent_projection_ratio: float = 0.0
    rank_augmented: int = 2
    ci95_kappa: Optional[List[float]] = None
    significant_nonzero: Optional[bool] = None
    family_label: str = "Family 3 Exploratory"

    def to_dict(self) -> dict:
        return asdict(self)


class IsoFrontierAudit:
    """
    Response surface audit and differential sensitivity analysis for 2-knob samplers.

    Parameters:
        s_grid: 1D array of control knob 1 coordinates (e.g. SC-CFG values s in [1.0, 4.0]).
        g_grid: 1D array of control knob 2 coordinates (e.g. gamma values in [0.8, 2.5]).
        ppl_grid: 2D array of shape (len(s_grid), len(g_grid)) with Gen-PPL evaluations.
        h_grid: 2D array of shape (len(s_grid), len(g_grid)) with Entropy evaluations.
        m_grids: Optional dict of metric_name -> 2D array of shape (len(s_grid), len(g_grid)).
        kx: Spline degree along s (default: min(3, len(s_grid)-1)).
        ky: Spline degree along g (default: min(3, len(g_grid)-1)).
    """

    def __init__(
        self,
        s_grid: Union[List[float], np.ndarray],
        g_grid: Union[List[float], np.ndarray],
        ppl_grid: np.ndarray,
        h_grid: np.ndarray,
        m_grids: Optional[Dict[str, np.ndarray]] = None,
        kx: Optional[int] = None,
        ky: Optional[int] = None,
    ):
        self.s_grid = np.asarray(s_grid, dtype=np.float64)
        self.g_grid = np.asarray(g_grid, dtype=np.float64)
        self.ppl_grid = np.asarray(ppl_grid, dtype=np.float64)
        self.h_grid = np.asarray(h_grid, dtype=np.float64)

        if len(self.s_grid) < 3 or len(self.g_grid) < 3:
            raise ValueError(
                "IsoFrontierAudit requires at least 3 points along each axis with degree kx, ky >= 2 "
                "to support partial derivatives without FITPACK code 10 crash."
            )

        if not np.all(np.isfinite(self.s_grid)):
            raise ValueError("s_grid contains NaN or Inf values.")
        if not np.all(np.isfinite(self.g_grid)):
            raise ValueError("g_grid contains NaN or Inf values.")
        if not np.all(np.diff(self.s_grid) > 0):
            raise ValueError("s_grid must be strictly increasing.")
        if not np.all(np.diff(self.g_grid) > 0):
            raise ValueError("g_grid must be strictly increasing.")

        n_s = len(self.s_grid)
        n_g = len(self.g_grid)

        if self.ppl_grid.shape != (n_s, n_g):
            raise ValueError(f"ppl_grid shape {self.ppl_grid.shape} != expected {(n_s, n_g)}")
        if self.h_grid.shape != (n_s, n_g):
            raise ValueError(f"h_grid shape {self.h_grid.shape} != expected {(n_s, n_g)}")

        if not np.all(np.isfinite(self.ppl_grid)):
            raise ValueError("ppl_grid contains NaN or Inf values.")
        if not np.all(np.isfinite(self.h_grid)):
            raise ValueError("h_grid contains NaN or Inf values.")

        self.kx = kx if kx is not None else min(3, n_s - 1)
        self.ky = ky if ky is not None else min(3, n_g - 1)

        if self.kx < 2 or self.ky < 2:
            raise ValueError(
                "IsoFrontierAudit requires at least 3 points along each axis with degree kx, ky >= 2 "
                "to support partial derivatives without FITPACK code 10 crash."
            )

        self.ppl_surf = interpolate.RectBivariateSpline(
            self.s_grid, self.g_grid, self.ppl_grid, kx=self.kx, ky=self.ky
        )
        self.h_surf = interpolate.RectBivariateSpline(
            self.s_grid, self.g_grid, self.h_grid, kx=self.kx, ky=self.ky
        )

        self.m_surfs: Dict[str, interpolate.RectBivariateSpline] = {}
        if m_grids is not None:
            for name, grid in m_grids.items():
                arr = np.asarray(grid, dtype=np.float64)
                if arr.shape != (n_s, n_g):
                    raise ValueError(f"m_grid['{name}'] shape {arr.shape} != expected {(n_s, n_g)}")
                if not np.all(np.isfinite(arr)):
                    raise ValueError(f"m_grid['{name}'] contains NaN or Inf values.")
                self.m_surfs[name] = interpolate.RectBivariateSpline(
                    self.s_grid, self.g_grid, arr, kx=self.kx, ky=self.ky
                )

    def add_metric_grid(self, name: str, grid: np.ndarray) -> None:
        """Register a secondary evaluation metric surface."""
        arr = np.asarray(grid, dtype=np.float64)
        expected = (len(self.s_grid), len(self.g_grid))
        if arr.shape != expected:
            raise ValueError(f"grid shape {arr.shape} != expected {expected}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"grid for metric '{name}' contains NaN or Inf values.")
        self.m_surfs[name] = interpolate.RectBivariateSpline(
            self.s_grid, self.g_grid, arr, kx=self.kx, ky=self.ky
        )

    def eval_ppl(self, s: float, g: float) -> float:
        return float(self.ppl_surf(s, g)[0, 0])

    def eval_h(self, s: float, g: float) -> float:
        return float(self.h_surf(s, g)[0, 0])

    def eval_metric(self, s: float, g: float, metric_name: str) -> float:
        if metric_name not in self.m_surfs:
            raise KeyError(f"Metric '{metric_name}' not registered in audit.")
        return float(self.m_surfs[metric_name](s, g)[0, 0])

    def get_gradients(self, s: float, g: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradients [dP/ds, dP/dg] and [dH/ds, dH/dg] at (s, g).
        """
        dp_ds = float(self.ppl_surf(s, g, dx=1, dy=0)[0, 0])
        dp_dg = float(self.ppl_surf(s, g, dx=0, dy=1)[0, 0])

        dh_ds = float(self.h_surf(s, g, dx=1, dy=0)[0, 0])
        dh_dg = float(self.h_surf(s, g, dx=0, dy=1)[0, 0])

        return np.array([dp_ds, dp_dg], dtype=np.float64), np.array([dh_ds, dh_dg], dtype=np.float64)

    def get_secondary_gradient(self, s: float, g: float, metric_name: str) -> np.ndarray:
        """Compute gradient [dm/ds, dm/dg] for secondary metric at (s, g)."""
        if metric_name not in self.m_surfs:
            raise KeyError(f"Metric '{metric_name}' not registered.")
        dm_ds = float(self.m_surfs[metric_name](s, g, dx=1, dy=0)[0, 0])
        dm_dg = float(self.m_surfs[metric_name](s, g, dx=0, dy=1)[0, 0])
        return np.array([dm_ds, dm_dg], dtype=np.float64)

    def compute_unit_tangent(
        self, s: float, g: float, forward_s: bool = True
    ) -> np.ndarray:
        """
        Compute the unit tangent vector u_hat along the iso-PPL level curve at (s, g).

        By the implicit function theorem, along P(s, g) = const:
            dP/ds * ds + dP/dg * dg = 0  =>  tangent is orthogonal to grad_p.
        Oriented such that u_s >= 0 if forward_s is True, else u_s <= 0.
        """
        grad_p, _ = self.get_gradients(s, g)
        norm = float(np.linalg.norm(grad_p))
        if norm < 1e-15:
            return np.array([1.0 if forward_s else -1.0, 0.0], dtype=np.float64)

        # In 2D, vector orthogonal to [p_s, p_g] is [-p_g, p_s]
        tangent = np.array([-grad_p[1], grad_p[0]], dtype=np.float64) / norm
        if forward_s:
            if tangent[0] < -1e-12 or (abs(tangent[0]) <= 1e-12 and tangent[1] < 0):
                tangent = -tangent
        else:
            if tangent[0] > 1e-12 or (abs(tangent[0]) <= 1e-12 and tangent[1] > 0):
                tangent = -tangent
        return tangent

    def solve_g_at_s(
        self,
        p_target: float,
        s_val: float,
        g_bounds: Optional[Tuple[float, float]] = None,
        prev_g: Optional[float] = None,
    ) -> float:
        """
        Find gamma such that P(s_val, gamma) = p_target.

        If multiple roots exist (e.g. saddle points or multi-branch level curves),
        selects the root minimizing |gamma - prev_g| to ensure continuity.
        """
        if g_bounds is None:
            g_low, g_high = float(self.g_grid[0]), float(self.g_grid[-1])
        else:
            g_low, g_high = g_bounds

        if g_low >= g_high:
            raise ValueError(f"g_bounds must satisfy g_low < g_high, got [{g_low}, {g_high}]")

        def obj(g_cand: float) -> float:
            return float(self.ppl_surf(s_val, g_cand)[0, 0]) - p_target

        # Scan for candidate roots across the interval to capture all branches
        n_scan = 40
        g_samples = np.linspace(g_low, g_high, n_scan)
        f_vals = np.array([obj(g_c) for g_c in g_samples])

        candidate_roots: List[float] = []

        # Sub-interval sign changes and exact zeros
        for i in range(len(g_samples) - 1):
            ga, gb = float(g_samples[i]), float(g_samples[i + 1])
            fa, fb = float(f_vals[i]), float(f_vals[i + 1])
            if abs(fa) < 1e-10:
                candidate_roots.append(ga)
            elif fa * fb < 0:
                try:
                    r = float(optimize.brentq(obj, ga, gb, xtol=1e-10))
                    candidate_roots.append(r)
                except Exception:
                    pass
        if abs(f_vals[-1]) < 1e-10:
            candidate_roots.append(float(g_samples[-1]))

        # Fallback to full interval bracket if sub-intervals didn't capture sign change
        if not candidate_roots:
            val_low = float(f_vals[0])
            val_high = float(f_vals[-1])
            if val_low * val_high <= 0:
                try:
                    r = float(optimize.brentq(obj, g_low, g_high, xtol=1e-10))
                    candidate_roots.append(r)
                except Exception:
                    pass

        # Fallback to bounded scalar minimization
        if not candidate_roots:
            res = optimize.minimize_scalar(lambda g: abs(obj(g)), bounds=(g_low, g_high), method="bounded")
            if res.fun < 1e-3:
                candidate_roots.append(float(res.x))

        if not candidate_roots:
            raise RuntimeError(
                f"Could not find gamma for p_target={p_target} at s={s_val} within [{g_low}, {g_high}]."
            )

        # Deduplicate candidate roots within 1e-5
        unique_roots: List[float] = []
        for r in sorted(candidate_roots):
            if not any(abs(r - u) < 1e-5 for u in unique_roots):
                unique_roots.append(r)

        if prev_g is not None:
            # Continuity tracking: pick root minimizing distance to prev_g
            unique_roots.sort(key=lambda r: abs(r - prev_g))
            return unique_roots[0]
        else:
            return unique_roots[0]

    def trace_iso_curve(
        self,
        p_target: float,
        s_range: Optional[Tuple[float, float]] = None,
        n_pts: int = 100,
        g_bounds: Optional[Tuple[float, float]] = None,
    ) -> IsoCurveResult:
        """
        Trace the level curve P(s, g) = p_target over s in s_range.

        Solves for g(s) using 1D root-finding (Brentq) with continuity tracking.
        Detects domain exits or gaps fail-closed, raising RuntimeError instead of false stitching.
        """
        if s_range is None:
            s_min, s_max = float(self.s_grid[0]), float(self.s_grid[-1])
        else:
            s_min, s_max = s_range

        if g_bounds is None:
            g_low, g_high = float(self.g_grid[0]), float(self.g_grid[-1])
        else:
            g_low, g_high = g_bounds

        if n_pts == 1 or s_min == s_max:
            s_vals = np.array([s_min])
        else:
            s_vals = np.linspace(s_min, s_max, n_pts)

        forward_s = bool(s_vals[-1] >= s_vals[0]) if len(s_vals) > 1 else True

        s_valid: List[float] = []
        g_valid: List[float] = []
        tangents: List[np.ndarray] = []
        ppl_vals: List[float] = []
        h_vals: List[float] = []

        for s_i in s_vals:
            # Continuity prediction to guide root selection across saddle/branch crossings
            if len(g_valid) >= 2:
                ds = s_valid[-1] - s_valid[-2]
                if abs(ds) > 1e-12:
                    g_pred = g_valid[-1] + (g_valid[-1] - g_valid[-2]) * ((float(s_i) - s_valid[-1]) / ds)
                else:
                    g_pred = g_valid[-1]
            elif len(g_valid) == 1:
                g_pred = g_valid[-1]
            else:
                g_pred = None

            try:
                root = self.solve_g_at_s(p_target, float(s_i), (g_low, g_high), prev_g=g_pred)
            except RuntimeError as e:
                # Disconnected gap detection: fail closed on domain exit or missing root
                raise RuntimeError(
                    f"Level curve for p_target={p_target} is disconnected or exits domain at s={float(s_i):.4f}: {e}"
                ) from e

            # Check for sudden large branch discontinuity
            if len(g_valid) >= 1 and abs(root - g_valid[-1]) > 0.5 * (g_high - g_low):
                raise RuntimeError(
                    f"Discontinuous jump detected along level curve at s={float(s_i):.4f}: "
                    f"|g_i - g_{{i-1}}| = {abs(root - g_valid[-1]):.4f} > threshold."
                )

            s_valid.append(float(s_i))
            g_valid.append(root)
            u_hat = self.compute_unit_tangent(float(s_i), root, forward_s=forward_s)
            tangents.append(u_hat)
            ppl_vals.append(self.eval_ppl(float(s_i), root))
            h_vals.append(self.eval_h(float(s_i), root))

        if len(s_valid) == 0:
            raise RuntimeError(f"Could not trace level curve for p_target={p_target} within bounds.")

        s_arr = np.array(s_valid, dtype=np.float64)
        g_arr = np.array(g_valid, dtype=np.float64)
        tan_arr = np.array(tangents, dtype=np.float64)

        # At stationary/saddle points (||grad_p|| ~ 0), spline roundoff makes gradient-orthogonal
        # tangents ill-conditioned; recover the branch tangent from neighboring traced points.
        if len(s_valid) >= 2:
            for k in range(len(s_valid)):
                gp_k, _ = self.get_gradients(float(s_arr[k]), float(g_arr[k]))
                if float(np.linalg.norm(gp_k)) < 1e-9:
                    k0 = max(0, k - 1)
                    k1 = min(len(s_valid) - 1, k + 1)
                    chord = np.array([s_arr[k1] - s_arr[k0], g_arr[k1] - g_arr[k0]], dtype=np.float64)
                    c_norm = float(np.linalg.norm(chord))
                    if c_norm > 1e-15:
                        tan_arr[k] = chord / c_norm

        # Compute cumulative arc length
        if len(s_valid) == 1:
            arc_len = np.array([0.0], dtype=np.float64)
        else:
            diffs = np.sqrt(np.diff(s_arr) ** 2 + np.diff(g_arr) ** 2)
            arc_len = np.concatenate([[0.0], np.cumsum(diffs)])

        return IsoCurveResult(
            p_target=float(p_target),
            s=s_arr,
            g=g_arr,
            arc_length=arc_len,
            total_arc_length=float(arc_len[-1]),
            tangents=tan_arr,
            ppl_values=np.array(ppl_vals, dtype=np.float64),
            h_values=np.array(h_vals, dtype=np.float64),
        )

    def jacobian_rank_test(
        self,
        s: float,
        g: float,
        metric_name: str,
        null_threshold: float = 0.05,
        ci95_kappa: Optional[Union[List[float], Tuple[float, float]]] = None,
    ) -> JacobianTestResult:
        """
        Perform the Jacobian Sensitivity / Invariance Test for secondary metric m at (s, g).

        In R^2, theta = (s, gamma), so grad_p, grad_h, and grad_m are 2-vectors in R^2.
        Any 3x2 matrix has maximum rank min(3, 2) = 2; rank 3 is impossible.

        The primary Jacobian J_prim = [grad_p^T; grad_h^T] in R^{2x2} is empirically near-rank-1
        (condition number ~ 2,700-4,000, singular values ~17.87 vs 0.0066), meaning the primary
        frontier carries effectively one constraint in this sampler family.
        The tolerance-matched operating points form a thin 1D band around the level curve of Gen-PPL
        with unit tangent t perp grad_p.

        Along this matched band, m is locally invariant iff grad_m is collinear with grad_p,
        i.e. kappa_m = grad_m . t = 0.

        The tangent projection ratio rho_m = |kappa_m| / ||grad_m|| in [0, 1] measures the fraction
        of grad_m orthogonal to grad_p (tangential to the primary level curve).
        If rho_m < null_threshold, m is functionally dependent on the primary frontier (is_null = True, rank 1).
        If rho_m >= null_threshold, m varies non-trivially along the matched band (is_null = False, rank 2).
        """
        grad_p, grad_h = self.get_gradients(s, g)
        grad_m = self.get_secondary_gradient(s, g, metric_name)

        J = np.vstack([grad_p, grad_h, grad_m])
        s_J = np.linalg.svd(J, compute_uv=False)

        J_prim = np.vstack([grad_p, grad_h])
        s_prim = np.linalg.svd(J_prim, compute_uv=False)
        cond_prim = float(s_prim[0] / (s_prim[1] + 1e-12))

        norm_p = float(np.linalg.norm(grad_p))
        norm_h = float(np.linalg.norm(grad_h))
        norm_m = float(np.linalg.norm(grad_m))

        if norm_p > 1e-12:
            unit_tan = self.compute_unit_tangent(s, g)
        elif norm_h > 1e-12:
            unit_tan = np.array([-grad_h[1], grad_h[0]], dtype=np.float64) / norm_h
            if unit_tan[0] < -1e-12 or (abs(unit_tan[0]) <= 1e-12 and unit_tan[1] < 0):
                unit_tan = -unit_tan
        else:
            unit_tan = np.array([1.0, 0.0], dtype=np.float64)

        kappa = float(np.dot(grad_m, unit_tan))

        if norm_m > 1e-15:
            if norm_p <= 1e-12 and norm_h <= 1e-12:
                rho_m = 1.0
            else:
                rho_m = float(abs(kappa) / norm_m)
            tangent_proj_ratio = min(1.0, max(0.0, rho_m))
            ortho_ratio = tangent_proj_ratio
        else:
            tangent_proj_ratio = 0.0
            ortho_ratio = 0.0

        is_null = bool(ortho_ratio < null_threshold)
        rank_aug = 1 if is_null else 2

        sig_nonzero = None
        ci_list = None
        if ci95_kappa is not None:
            ci_list = [float(ci95_kappa[0]), float(ci95_kappa[1])]
            sig_nonzero = bool(ci_list[0] > 0 or ci_list[1] < 0)

        return JacobianTestResult(
            metric_name=metric_name,
            s=float(s),
            g=float(g),
            grad_p=grad_p.tolist(),
            grad_h=grad_h.tolist(),
            grad_m=grad_m.tolist(),
            singular_values_J=s_J.tolist(),
            singular_values_primary=s_prim.tolist(),
            primary_condition_number=cond_prim,
            unit_tangent=unit_tan.tolist(),
            kappa_m=kappa,
            ortho_ratio=ortho_ratio,
            is_null=is_null,
            tangent_projection_ratio=tangent_proj_ratio,
            rank_augmented=rank_aug,
            ci95_kappa=ci_list,
            significant_nonzero=sig_nonzero,
        )

    def compute_kappa(self, s: float, g: float, metric_name: str) -> float:
        """
        Compute arc-length sensitivity kappa_m = dm / dl|_C = grad_m^T * u_hat.
        """
        grad_m = self.get_secondary_gradient(s, g, metric_name)
        unit_tan = self.compute_unit_tangent(s, g)
        return float(np.dot(grad_m, unit_tan))

    def compute_kappa_along_curve(
        self, iso_curve: IsoCurveResult, metric_name: str
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        Compute kappa_m profile along an entire traced iso-frontier curve.
        """
        kappas = []
        for s_i, g_i, u_i in zip(iso_curve.s, iso_curve.g, iso_curve.tangents):
            grad_m = self.get_secondary_gradient(s_i, g_i, metric_name)
            kappas.append(float(np.dot(grad_m, u_i)))

        kappa_arr = np.array(kappas, dtype=np.float64)
        # Trapezoidal numerical integration of dm = kappa * dl
        delta_m_integral = float(np.trapezoid(kappa_arr, iso_curve.arc_length))

        return {
            "s": iso_curve.s,
            "g": iso_curve.g,
            "arc_length": iso_curve.arc_length,
            "kappa": kappa_arr,
            "mean_kappa": float(np.mean(kappa_arr)),
            "delta_m_integrated": delta_m_integral,
        }

    @staticmethod
    def bootstrap_kappa_ci(
        seed_grids: List[Dict[str, np.ndarray]],
        s_grid: np.ndarray,
        g_grid: np.ndarray,
        p_target: float,
        s_eval: float,
        metric_name: str,
        n_boot: int = 500,
        alpha: float = 0.05,
        random_state: int = 42,
    ) -> Dict[str, float]:
        """
        Estimate seed bootstrap confidence interval for kappa_m at (s_eval, g(s_eval)).

        Args:
            seed_grids: List of seed dictionaries, each containing:
                        {'ppl': (n_s, n_g), 'h': (n_s, n_g), metric_name: (n_s, n_g)}.
            s_grid: 1D SC-CFG grid.
            g_grid: 1D gamma grid.
            p_target: Target PPL operating point.
            s_eval: SC-CFG value at which to evaluate kappa.
            metric_name: Name of secondary metric.
            n_boot: Number of bootstrap iterations.
            alpha: Significance level for 100*(1-alpha)% CI.
            random_state: RNG seed.
        """
        rng = np.random.RandomState(random_state)
        n_seeds = len(seed_grids)
        if n_seeds < 2:
            raise ValueError("At least 2 seed grids required for bootstrap.")

        # Point estimate on full average
        mean_ppl = np.mean([sg["ppl"] for sg in seed_grids], axis=0)
        mean_h = np.mean([sg["h"] for sg in seed_grids], axis=0)
        mean_m = np.mean([sg[metric_name] for sg in seed_grids], axis=0)

        main_audit = IsoFrontierAudit(s_grid, g_grid, mean_ppl, mean_h, {metric_name: mean_m})
        point_g = main_audit.solve_g_at_s(p_target, s_eval)
        point_kappa = main_audit.compute_kappa(s_eval, point_g, metric_name)

        boot_kappas = []
        for _ in range(n_boot):
            idx = rng.choice(n_seeds, size=n_seeds, replace=True)
            b_ppl = np.mean([seed_grids[i]["ppl"] for i in idx], axis=0)
            b_h = np.mean([seed_grids[i]["h"] for i in idx], axis=0)
            b_m = np.mean([seed_grids[i][metric_name] for i in idx], axis=0)

            try:
                b_audit = IsoFrontierAudit(s_grid, g_grid, b_ppl, b_h, {metric_name: b_m})
                b_g = b_audit.solve_g_at_s(p_target, s_eval)
                k_val = b_audit.compute_kappa(s_eval, b_g, metric_name)
                boot_kappas.append(k_val)
            except Exception:
                continue

        if len(boot_kappas) < 10:
            raise RuntimeError("Bootstrap failed to converge on sufficient replicates.")

        boot_arr = np.array(boot_kappas, dtype=np.float64)
        low_pct = 100.0 * (alpha / 2.0)
        high_pct = 100.0 * (1.0 - alpha / 2.0)

        ci_low = float(np.percentile(boot_arr, low_pct))
        ci_high = float(np.percentile(boot_arr, high_pct))
        se_boot = float(np.std(boot_arr, ddof=1))

        return {
            "metric_name": metric_name,
            "p_target": float(p_target),
            "s_eval": float(s_eval),
            "g_eval": point_g,
            "kappa_point": float(point_kappa),
            "kappa_boot_mean": float(np.mean(boot_arr)),
            "se": se_boot,
            "ci_lower": ci_low,
            "ci_upper": ci_high,
            "ci_level": float(1.0 - alpha),
            "n_boot_success": len(boot_kappas),
        }
