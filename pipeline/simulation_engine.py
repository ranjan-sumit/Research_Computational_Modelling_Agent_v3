"""
Stage 10: Multi-Algorithm Simulation Engine
Algorithms:
  1. Monte Carlo          — uncertainty propagation through CI ranges
  2. Markov Chain         — disease state progression over time
  3. Survival Analysis    — time-to-event, Weibull-based
  4. Bayesian Network     — conditional probability estimation
  5. Bootstrap Resampling — robust CIs for small/incomplete data
  6. Tornado Sensitivity  — one-at-a-time parameter importance
  7. Hybrid              — Monte Carlo wrapping Markov

Auto-selector runs ALL applicable algorithms based on available data.
"""
import numpy as np
from scipy import stats

np.random.seed(42)
N_SIM   = 10_000
N_YEARS = 10


# ── Utilities ─────────────────────────────────────────────────────────────────

def _by_cat(params: list, *cats) -> list:
    return [p for p in params if p.get("category") in cats]


def _sample(p: dict, n: int = N_SIM) -> np.ndarray:
    val = float(p.get("value", 0) or 0)
    lo  = p.get("ci_lower")
    hi  = p.get("ci_upper")
    if lo is not None and hi is not None:
        std = max((float(hi) - float(lo)) / 3.92, 1e-6)
        s   = np.random.normal(val, std, n)
    else:
        std = abs(val) * 0.20 if val else 0.01
        s   = np.random.normal(val, std, n)
    return np.clip(s, 0, 100 if p.get("unit") == "percent" else None)


def _stats(arr: np.ndarray) -> dict:
    return {
        "mean":        round(float(np.mean(arr)), 4),
        "median":      round(float(np.median(arr)), 4),
        "ci_lower_95": round(float(np.percentile(arr, 2.5)), 4),
        "ci_upper_95": round(float(np.percentile(arr, 97.5)), 4),
        "std":         round(float(np.std(arr)), 4),
    }


def _modifier(scenario: str) -> float:
    s = scenario.lower()
    if "optimistic" in s:   return 0.80
    if "conservative" in s: return 1.20
    if "pessimistic" in s:  return 1.35
    return 1.0


# ── 1. Monte Carlo ────────────────────────────────────────────────────────────

def run_monte_carlo(params: list, scenarios: list) -> dict:
    outcome_p  = _by_cat(params, "prevalence", "incidence", "proportion", "rate", "survival")
    risk_p     = _by_cat(params, "odds_ratio", "relative_risk", "hazard_ratio")
    efficacy_p = _by_cat(params, "efficacy")

    if not outcome_p:
        outcome_p = [p for p in params if p.get("value") is not None][:5]

    sensitivity = {}
    scenario_results = {}

    for scenario in scenarios:
        mod  = _modifier(scenario)
        skey = scenario.lower().replace(" ", "_")
        sr   = {}

        distributions = []
        for p in outcome_p[:8]:
            samples = _sample(p, N_SIM) * mod
            name    = p.get("name", p.get("category", "outcome"))
            sr[name] = {**_stats(samples), "source": p.get("source_paper", "")}
            distributions.append((name, samples))

        if risk_p and distributions:
            combined_risk = np.ones(N_SIM)
            for rp in risk_p[:4]:
                combined_risk *= np.clip(_sample(rp, N_SIM), 0.1, 20)
            _, base = distributions[0]
            adjusted = base * combined_risk / np.maximum(np.mean(combined_risk), 1e-9)
            sr["risk_adjusted_outcome"] = _stats(adjusted)

        if efficacy_p and distributions:
            _, base = distributions[0]
            for ep in efficacy_p[:2]:
                eff = np.clip(_sample(ep, N_SIM) / 100.0, 0, 1)
                sr[f"with_treatment_{ep.get('name','tx')}"] = _stats(base * (1 - eff) * mod)

        scenario_results[skey] = sr

        # Sensitivity: correlate each input with combined output
        if distributions:
            combined = np.mean(np.stack([s for _, s in distributions], axis=0), axis=0)
            for p in outcome_p[:8]:
                s = _sample(p, N_SIM)
                corr = (float(np.corrcoef(s, combined)[0, 1])
                        if np.std(combined) > 0 and np.std(s) > 0 else 0.0)
                sensitivity[p.get("name", "param")] = abs(corr)

    sensitivity_ranked = sorted(sensitivity.items(), key=lambda x: x[1], reverse=True)

    return {
        "model_used":         "Monte Carlo",
        "n_simulations":      N_SIM,
        "scenarios_run":      scenarios,
        "scenario_results":   scenario_results,
        "sensitivity_ranking": [{"parameter": k, "importance": round(v, 4)}
                                 for k, v in sensitivity_ranked],
        "headline_numbers":   {f"{sk}_{pn}_mean": sr[pn]["mean"]
                               for sk, sr in scenario_results.items()
                               for pn in list(sr.keys())[:2]},
        "summary":            {"model": "Monte Carlo", "n_sim": N_SIM},
    }


# ── 2. Markov Chain ───────────────────────────────────────────────────────────

def run_markov(params: list, scenarios: list) -> dict:
    trans_p = _by_cat(params, "transition_probability")
    states  = ["Healthy", "At_Risk", "Diseased", "Recovered", "Death"]
    n_s     = len(states)

    # Default transition matrix
    defaults = np.array([
        [0.85, 0.10, 0.03, 0.01, 0.01],
        [0.05, 0.60, 0.25, 0.05, 0.05],
        [0.02, 0.08, 0.55, 0.20, 0.15],
        [0.10, 0.10, 0.10, 0.65, 0.05],
        [0.00, 0.00, 0.00, 0.00, 1.00],
    ])

    scenario_results = {}
    for scenario in scenarios:
        mod  = _modifier(scenario)
        skey = scenario.lower().replace(" ", "_")

        T = defaults.copy()
        if trans_p:
            for p in trans_p:
                val  = float(p.get("value", 0)) / 100.0 * mod
                val  = np.clip(val, 0, 0.95)
                name = p.get("name", "").lower()
                for i, s1 in enumerate(states):
                    for j, s2 in enumerate(states):
                        if s1.lower() in name and s2.lower() in name and i != j:
                            T[i][j] = val

        # Normalise rows
        for i in range(n_s):
            rs = T[i].sum()
            T[i] = T[i] / rs if rs > 0 else (np.eye(n_s)[i])

        cohort = np.zeros(n_s)
        cohort[0] = 10_000
        trajectories = {s: [float(cohort[j])] for j, s in enumerate(states)}

        for _ in range(N_YEARS):
            cohort = cohort @ T
            for j, s in enumerate(states):
                trajectories[s].append(round(float(cohort[j]), 1))

        scenario_results[skey] = {
            "states":            states,
            "trajectories":      trajectories,
            "final_year":        {s: trajectories[s][-1] for s in states},
            "transition_matrix": T.tolist(),
            "years_simulated":   N_YEARS,
        }

    disease_ends = {}
    for sc in scenarios:
        sk = sc.lower().replace(" ", "_")
        if sk in scenario_results:
            disease_ends[f"{sk}_disease_end"] = round(
                scenario_results[sk]["final_year"].get("Diseased", 0), 1
            )

    return {
        "model_used":       "Markov Chain",
        "cohort_size":      10_000,
        "n_years":          N_YEARS,
        "states":           states,
        "scenarios_run":    scenarios,
        "scenario_results": scenario_results,
        "headline_numbers": disease_ends,
        "sensitivity_ranking": [],
        "summary":          {"model": "Markov Chain", "years": N_YEARS},
    }


# ── 3. Survival Analysis (Weibull approximation) ──────────────────────────────

def run_survival_analysis(params: list, scenarios: list) -> dict:
    survival_p = _by_cat(params, "survival", "hazard_ratio", "rate")
    if not survival_p:
        survival_p = [p for p in params if p.get("value") is not None][:3]

    time_points = list(range(0, N_YEARS + 1))
    scenario_results = {}

    for scenario in scenarios:
        mod  = _modifier(scenario)
        skey = scenario.lower().replace(" ", "_")

        # Weibull survival: S(t) = exp(-(t/λ)^k)
        # Estimate lambda from baseline survival, k=1.5 (typical healthcare)
        base_surv = 1.0
        for p in survival_p[:2]:
            v = float(p.get("value", 50)) / 100.0
            base_surv = min(base_surv, max(v, 0.01))

        k      = 1.5
        lam    = 5.0 / ((-np.log(base_surv * mod)) ** (1 / k) + 1e-9)
        lam    = max(lam, 0.5)

        curve  = [round(float(np.exp(-((t / lam) ** k))), 4) for t in time_points]
        median = round(float(lam * (np.log(2) ** (1 / k))), 2)

        scenario_results[skey] = {
            "time_points":    time_points,
            "survival_curve": curve,
            "median_survival_years": median,
            "five_year_survival":    curve[5] if len(curve) > 5 else None,
            "ten_year_survival":     curve[10] if len(curve) > 10 else None,
        }

    return {
        "model_used":       "Survival Analysis (Weibull)",
        "scenarios_run":    scenarios,
        "scenario_results": scenario_results,
        "headline_numbers": {
            f"{sc.lower().replace(' ','_')}_5yr_survival":
                scenario_results.get(sc.lower().replace(" ", "_"), {}).get("five_year_survival", 0)
            for sc in scenarios
        },
        "sensitivity_ranking": [],
        "summary":          {"model": "Survival Analysis", "shape_parameter_k": 1.5},
    }


# ── 4. Bayesian Network (conditional probability) ─────────────────────────────

def run_bayesian_network(params: list, scenarios: list) -> dict:
    risk_p   = _by_cat(params, "odds_ratio", "relative_risk")
    prev_p   = _by_cat(params, "prevalence", "incidence")

    baseline = float(prev_p[0].get("value", 15)) / 100.0 if prev_p else 0.15
    scenario_results = {}

    for scenario in scenarios:
        mod  = _modifier(scenario)
        skey = scenario.lower().replace(" ", "_")

        # Sequential Bayesian update: P(D|R1,R2,...) via log-odds
        log_odds_baseline = np.log(baseline / (1 - baseline + 1e-9))
        risk_factors = []
        combined_lo  = log_odds_baseline

        for rp in risk_p[:6]:
            or_val = float(rp.get("value", 1.0)) * mod
            lo_inc = np.log(max(or_val, 0.01))
            combined_lo += lo_inc
            prob_with = 1 / (1 + np.exp(-combined_lo))
            risk_factors.append({
                "factor":       rp.get("name", rp.get("condition", "risk_factor")),
                "odds_ratio":   round(or_val, 3),
                "prob_with_this_factor": round(float(prob_with), 4),
                "source_paper": rp.get("source_paper", ""),
            })

        posterior = float(1 / (1 + np.exp(-combined_lo)))

        scenario_results[skey] = {
            "baseline_probability": round(baseline, 4),
            "posterior_probability": round(posterior, 4),
            "absolute_increase":    round(posterior - baseline, 4),
            "relative_increase":    round((posterior - baseline) / max(baseline, 1e-9), 4),
            "risk_factors":         risk_factors,
        }

    return {
        "model_used":       "Bayesian Network",
        "scenarios_run":    scenarios,
        "scenario_results": scenario_results,
        "headline_numbers": {
            f"{sc.lower().replace(' ','_')}_posterior_prob":
                round(scenario_results.get(
                    sc.lower().replace(" ", "_"), {}
                ).get("posterior_probability", 0) * 100, 2)
            for sc in scenarios
        },
        "sensitivity_ranking": [],
        "summary":          {"model": "Bayesian Network", "baseline": baseline},
    }


# ── 5. Bootstrap Resampling ───────────────────────────────────────────────────

def run_bootstrap(params: list, scenarios: list, n_boot: int = 5000) -> dict:
    numeric = [p for p in params if p.get("value") is not None]
    if not numeric:
        return {
            "model_used": "Bootstrap Resampling",
            "scenarios_run": scenarios,
            "headline_numbers": {},
            "sensitivity_ranking": [],
            "scenario_results": {},
            "summary": {"error": "No numeric params"},
        }

    values = np.array([float(p.get("value", 0)) for p in numeric])
    scenario_results = {}

    for scenario in scenarios:
        mod  = _modifier(scenario)
        skey = scenario.lower().replace(" ", "_")

        boot_means = np.array([
            np.mean(np.random.choice(values * mod, size=len(values), replace=True))
            for _ in range(n_boot)
        ])

        scenario_results[skey] = {
            **_stats(boot_means),
            "n_bootstrap": n_boot,
            "stability":   "stable" if float(np.std(boot_means)) < 0.1 * abs(float(np.mean(boot_means))) else "unstable",
        }

    return {
        "model_used":       "Bootstrap Resampling",
        "scenarios_run":    scenarios,
        "scenario_results": scenario_results,
        "headline_numbers": {
            f"{sc.lower().replace(' ','_')}_boot_mean":
                scenario_results.get(sc.lower().replace(" ", "_"), {}).get("mean", 0)
            for sc in scenarios
        },
        "sensitivity_ranking": [],
        "summary":          {"model": "Bootstrap", "n_boot": n_boot},
    }


# ── 6. Tornado Sensitivity (one-at-a-time) ────────────────────────────────────

def run_tornado(params: list, scenarios: list) -> dict:
    numeric = [p for p in params if p.get("value") is not None]
    if not numeric:
        return {
            "model_used": "Tornado Sensitivity",
            "scenarios_run": scenarios,
            "headline_numbers": {},
            "sensitivity_ranking": [],
            "scenario_results": {},
            "summary": {},
        }

    base_vals = np.array([float(p.get("value", 0)) for p in numeric])
    base_mean = float(np.mean(base_vals)) if len(base_vals) > 0 else 0

    swings = []
    for i, p in enumerate(numeric):
        lo = float(p.get("ci_lower", float(p.get("value", 0)) * 0.8))
        hi = float(p.get("ci_upper", float(p.get("value", 0)) * 1.2))

        vals_low  = base_vals.copy(); vals_low[i]  = lo
        vals_high = base_vals.copy(); vals_high[i] = hi

        mean_low  = float(np.mean(vals_low))
        mean_high = float(np.mean(vals_high))
        swing     = abs(mean_high - mean_low)

        swings.append({
            "parameter":  p.get("name", f"param_{i}"),
            "low_value":  round(lo, 4),
            "high_value": round(hi, 4),
            "mean_at_low":  round(mean_low, 4),
            "mean_at_high": round(mean_high, 4),
            "swing":        round(swing, 4),
            "importance":   round(swing / max(abs(base_mean), 1e-9), 4),
            "source_paper": p.get("source_paper", ""),
        })

    swings.sort(key=lambda x: x["swing"], reverse=True)

    return {
        "model_used":     "Tornado Sensitivity",
        "scenarios_run":  scenarios,
        "baseline_mean":  round(base_mean, 4),
        "parameter_swings": swings,
        "scenario_results": {"base": {"mean": base_mean}},
        "headline_numbers": {"top_parameter": swings[0]["parameter"] if swings else ""},
        "sensitivity_ranking": [
            {"parameter": s["parameter"], "importance": s["importance"]}
            for s in swings[:8]
        ],
        "summary":        {"model": "Tornado", "top_driver": swings[0]["parameter"] if swings else "N/A"},
    }


# ── 7. Hybrid ─────────────────────────────────────────────────────────────────

def run_hybrid(params: list, scenarios: list) -> dict:
    mc = run_monte_carlo(params, scenarios)
    mk = run_markov(params, scenarios)
    return {
        "model_used":       "Hybrid Monte Carlo + Markov",
        "monte_carlo":      mc,
        "markov":           mk,
        "scenarios_run":    scenarios,
        "headline_numbers": {**mc["headline_numbers"], **mk["headline_numbers"]},
        "sensitivity_ranking": mc["sensitivity_ranking"],
        "summary":          {"mc": mc["summary"], "markov": mk["summary"]},
        "scenario_results": mc["scenario_results"],
    }


# ── Auto-Selector ─────────────────────────────────────────────────────────────

def auto_select_models(params: list, sufficiency: dict) -> list:
    """
    Decide which algorithms to run based on available data.
    Returns a list of model keys to run.
    Always includes Tornado (always useful).
    """
    trans_p    = _by_cat(params, "transition_probability")
    survival_p = _by_cat(params, "survival")
    risk_p     = _by_cat(params, "odds_ratio", "relative_risk", "hazard_ratio")
    prev_p     = _by_cat(params, "prevalence", "incidence")
    any_params = [p for p in params if p.get("value") is not None]

    models = ["tornado"]  # always

    if len(any_params) >= 3:
        models.append("monte_carlo")

    if trans_p:
        models.append("markov_chain")

    if survival_p or any(
        p.get("time_horizon") and p.get("time_horizon") not in ["null", None, ""]
        for p in params
    ):
        models.append("survival_analysis")

    if len(risk_p) >= 2 and prev_p:
        models.append("bayesian_network")

    score = sufficiency.get("coverage_score", 5)
    if score < 5 or len(any_params) < 5:
        models.append("bootstrap")

    return list(dict.fromkeys(models))  # deduplicate, preserve order


def run_all_applicable(params: list, council_decision: dict, sufficiency: dict) -> dict:
    """
    Run all algorithms the data supports.
    Returns results from each algorithm + a combined summary.
    """
    final    = council_decision.get("final_decision", {})
    scenarios = final.get("scenarios") or ["Base Case", "Optimistic", "Conservative"]

    models_to_run = auto_select_models(params, sufficiency)

    # Council may have overridden — ensure council pick is included
    council_model = final.get("selected_model", "monte_carlo")
    model_map = {
        "monte_carlo":             "monte_carlo",
        "markov_chain":            "markov_chain",
        "survival_model":          "survival_analysis",
        "bayesian_network":        "bayesian_network",
        "hybrid_monte_carlo_markov": "hybrid",
    }
    mapped = model_map.get(council_model)
    if mapped and mapped not in models_to_run:
        models_to_run.insert(0, mapped)

    runners = {
        "monte_carlo":       lambda: run_monte_carlo(params, scenarios),
        "markov_chain":      lambda: run_markov(params, scenarios),
        "survival_analysis": lambda: run_survival_analysis(params, scenarios),
        "bayesian_network":  lambda: run_bayesian_network(params, scenarios),
        "bootstrap":         lambda: run_bootstrap(params, scenarios),
        "tornado":           lambda: run_tornado(params, scenarios),
        "hybrid":            lambda: run_hybrid(params, scenarios),
    }

    all_results   = {}
    all_headlines = {}
    primary_result = None

    for model_key in models_to_run:
        runner = runners.get(model_key)
        if not runner:
            continue
        try:
            result = runner()
            all_results[model_key]   = result
            all_headlines.update(result.get("headline_numbers", {}))
            if primary_result is None:
                primary_result = result
        except Exception as e:
            all_results[model_key] = {"model_used": model_key, "error": str(e)}

    # Combined sensitivity across all models
    all_sens = {}
    for r in all_results.values():
        for s in r.get("sensitivity_ranking", []):
            p = s.get("parameter", "")
            all_sens[p] = max(all_sens.get(p, 0), s.get("importance", 0))
    combined_sens = sorted(all_sens.items(), key=lambda x: x[1], reverse=True)

    return {
        "model_used":         f"Multi-Algorithm ({', '.join(all_results.keys())})",
        "models_run":         list(all_results.keys()),
        "scenarios_run":      scenarios,
        "algorithm_results":  all_results,
        "headline_numbers":   all_headlines,
        "sensitivity_ranking": [
            {"parameter": k, "importance": round(v, 4)} for k, v in combined_sens[:8]
        ],
        "scenario_results":   primary_result.get("scenario_results", {}) if primary_result else {},
        "summary": {
            "models_run":    list(all_results.keys()),
            "total_results": len(all_results),
        },
    }


# ── Public entry point ────────────────────────────────────────────────────────

def run_simulation(params: list, council_decision: dict, sufficiency: dict = None) -> dict:
    """Main dispatcher — runs all applicable algorithms."""
    return run_all_applicable(params, council_decision, sufficiency or {})
