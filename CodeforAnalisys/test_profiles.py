"""Every analysis profile of a maintained case matches what actually runs.

Physics (beta, anisotropy, kappa, box, mi/me, B0) comes from the case file and
the shared header; numerics (grid, particles per cell, nmax, output cadences)
from the COSMA job script that runs the case, or the header defaults when
there is none. A profile that drifts from its run silently mislabels every
unit conversion of the analysis.
"""
import os
import re
from pathlib import Path

import pytest

os.environ.pop("PSC_ANALYSIS_DATA_DIR", None)
import psc_units  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFINE = re.compile(r"#define (PSC_\w+)\s+(\S+)")
HEADER = dict(DEFINE.findall((ROOT / "src" / "psc_anisotropy_case.hxx").read_text()))
JOBS = {}
for script in (ROOT / "cosma_jobs" / "simulacion").glob("*.sh"):
    text = script.read_text()
    target = re.search(r"^PSC_TARGET=(\S+)", text, re.M)
    if target:
        JOBS[target.group(1)] = dict(re.findall(r'^(PSC_\w+)="\$\{\w+:-([^}]+)\}"', text, re.M))
CASES = [name for name in psc_units._PROFILES if (ROOT / "src" / f"psc_{name}.cxx").exists()]


def _num(value):
    return None if value is None else float(str(value).strip("()").rstrip("f"))


@pytest.mark.parametrize("name", CASES)
def test_profile_matches_case_and_job(name):
    profile = psc_units._PROFILES[name]
    case = dict(DEFINE.findall((ROOT / "src" / f"psc_{name}.cxx").read_text()))
    env = JOBS.get(f"psc_{name}", {})
    expected = {
        "beta_i_par": case["PSC_BETA_I_PAR"], "Ti_perp_over_Ti_par": case["PSC_TI_PERP_OVER_TI_PAR"],
        "beta_e_par": case["PSC_BETA_E_PAR"], "Te_perp_over_Te_par": case["PSC_TE_PERP_OVER_TE_PAR"],
        "kappa": case.get("PSC_KAPPA") if case.get("PSC_USE_KAPPA") == "1" else None,
        "domain_di": case.get("PSC_DOMAIN_DI", HEADER["PSC_DOMAIN_DI"]),
        "mass_ratio": case.get("PSC_MASS_RATIO", HEADER["PSC_MASS_RATIO"]),
        "vA_over_c": HEADER["PSC_VA_OVER_C"],
        "ngrid": env.get("PSC_NGRID", HEADER["PSC_NGRID_DEFAULT"]),
        "nicell": env.get("PSC_NICELL", HEADER["PSC_NICELL_DEFAULT"]),
        "nmax": env.get("PSC_NMAX", HEADER["PSC_NMAX_DEFAULT"]),
        "fields_every": env.get("PSC_FIELDS_EVERY", HEADER["PSC_FIELDS_EVERY_DEFAULT"]),
        "particles_every": env.get("PSC_PARTICLES_EVERY", HEADER["PSC_PARTICLES_EVERY_DEFAULT"]),
    }
    defaults = {"fields_every": 500, "particles_every": 10000}
    for key, want in expected.items():
        have = profile.get(key, defaults.get(key))
        if want is None:
            assert have is None, f"{name}.{key}: profile {have}, case runs without it"
        else:
            assert _num(have) == pytest.approx(_num(want), rel=1e-12), f"{name}.{key}: profile {have}, run {want}"
    assert profile["particle_basename"] == f"prt_{name}"
