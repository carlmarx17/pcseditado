# Analysis pipeline for PSC outputs.
#
# Paths supplied on the command line are interpreted relative to this
# directory. Results are grouped by RUN_NAME so different simulations never
# overwrite each other.

PYTHON ?= $(abspath ../.venv/bin/python)
DATA_DIR ?= ../build/src
# Particle series names, from prt_<case>.<step>.h5 or .bp. Portable: BSD find
# (macOS) has no -printf, and the previous GNU-only form made detection and the
# CASE/data cross-check silently do nothing on a laptop.
PRT_SERIES = find "$(DATA_DIR)" -maxdepth 1 \( -name 'prt_*.h5' -o -name 'prt_*.bp' \) \
	2>/dev/null | sed -E 's|.*/||; s/^(prt_)?([^.]+)\.[0-9]+(_p[0-9]+)?\.(h5|bp)$$/\2/' | sort -u
DETECTED_CASE = $(shell detected=$$($(PRT_SERIES) | head -1); \
	if test -n "$$detected"; then echo "$$detected"; \
	else basename "$(abspath $(DATA_DIR))" | grep -E '^[FMW]_[SMW]_bM$$' || true; fi)
CASE ?= $(DETECTED_CASE)
RUN_NAME ?= $(if $(CASE),$(CASE),$(notdir $(abspath $(DATA_DIR))))
RESULTS_ROOT ?= ../analysis_results
OUTPUT_DIR ?= $(RESULTS_ROOT)/$(RUN_NAME)

B0_REF ?= 0.08
PSC_PROFILE ?= $(CASE)
export PSC_ANALYSIS_DATA_DIR = $(abspath $(DATA_DIR))
# Optional JSON with runtime dt_code/nicell and geometry overrides.
export PSC_ANALYSIS_CONFIG
PARTICLE_BASENAME ?= $(if $(CASE),prt_$(CASE),)

FIELD_PATTERN = $(DATA_DIR)/pfd.*_p*.h5
MOMENT_PATTERN = $(DATA_DIR)/pfd_moments.*_p*.h5
PARTICLE_PATTERN = $(if $(PARTICLE_BASENAME),$(DATA_DIR)/$(PARTICLE_BASENAME).*.h5,$(DATA_DIR)/prt*.h5)
FIRST_PARTICLE = $(firstword $(sort $(wildcard $(PARTICLE_PATTERN))))

ANISOTROPY_OUT = $(OUTPUT_DIR)/01_anisotropy
FIELDS_OUT = $(OUTPUT_DIR)/02_fields
PARTICLES_OUT = $(OUTPUT_DIR)/03_particles
SPECTRAL_OUT = $(OUTPUT_DIR)/04_spectra
DIAMAGNETIC_OUT = $(OUTPUT_DIR)/05_diamagnetic
HEATFLUX_OUT = $(OUTPUT_DIR)/06_heat_flux
STRUCTURES_OUT = $(OUTPUT_DIR)/07_structures
VALIDATION_OUT = $(OUTPUT_DIR)/08_validation
PHYSICS_OUT = $(OUTPUT_DIR)/09_physical_diagnostics
RECONNECTION_OUT = $(OUTPUT_DIR)/10_reconnection
COMPARE_CASES ?=
COMPARE_OUT ?= $(RESULTS_ROOT)/comparison_physical
# Convergence / realizations: CONVERGENCE_RUNS='base=../analysis_results/A dx2=../analysis_results/B'
CONVERGENCE_RUNS ?=
CONVERGENCE_OUT ?= $(RESULTS_ROOT)/convergence
# Free label of this realization (seed, dt/ppc variant); stored in the manifest.
RUN_TAG ?=
# PSC job logs with the gauss/continuity checks. COSMA writes them next to the
# run directory as <job-name>_<jobid>.out, the job id being the suffix of the
# run directory name (see cosma_jobs/simulacion/*.sh).
RUN_JOBID = $(lastword $(subst _, ,$(notdir $(abspath $(DATA_DIR)))))
LOGS ?= $(wildcard $(DATA_DIR)/*.out $(DATA_DIR)/*.log $(dir $(abspath $(DATA_DIR)))*_$(RUN_JOBID).out)
THEORY_CSV ?=
# Dispersion-diagram controls.
#
# DISPERSION_MODE is the main knob: it selects the instability preset
# (mirror, firehose-oblique, firehose-parallel, emic, whistler, generic) and
# with it the k range, angular band, frequency axis and time handling that the
# mode actually needs, plus the resolution report that says whether the box and
# the output cadence can represent it at all. Anything set explicitly below
# still wins over the preset.
#
# The aperiodic modes (mirror, oblique firehose) have omega_r = 0 and therefore
# no dispersion branch: for those the omega-k diagram is a consistency check,
# not the measurement -- read gamma(k_par,k_perp) from the growth map instead.
#
# DISPERSION_T_START/T_END bound the FFT window. Keep it inside a single
# physical phase: a window spanning both linear growth and saturation is not
# stationary and the transform then measures the envelope, not omega(k).
# Snapshot cadence: SNAPSHOT_EVERY controls the saved PNG maps (thesis
# figures), GIF_EVERY the animation frames (rendered in memory, not saved).
SNAPSHOT_EVERY ?= 100000
GIF_EVERY ?= 10000
FIELDS_GIF ?= 1
DISPERSION_MODE ?= generic
DISPERSION_COMPONENT ?= total
DISPERSION_THETA_MAX ?=
DISPERSION_THETA_MIN ?=
DISPERSION_KMAX_DI ?=
DISPERSION_DEGROWTH ?=
DISPERSION_OMEGA_SCALE ?=
DISPERSION_SPATIAL_WINDOW ?=
DISPERSION_TEMPORAL_WINDOW ?=
DISPERSION_T_START ?=
DISPERSION_T_END ?=
GROWTH_COMPONENT ?= parallel
GROWTH_T_START ?= $(DISPERSION_T_START)
GROWTH_T_END ?= $(DISPERSION_T_END)
# Empty -> the script's profile-aware default: 1.5 d_i^-1 for ion-scale cases,
# k d_e <= 2 (k d_i <= 2 sqrt(mi/me)) for the electron-driven whistler cases.
GROWTH_KPAR_MAX ?=
GROWTH_KPERP_MAX ?=
GROWTH_DISPLAY_KPAR_MAX ?=
GROWTH_DISPLAY_KPERP_MAX ?=
GROWTH_SHADING ?= auto
GROWTH_CONTOURS ?= 7
GROWTH_ANGLE_STEP ?= 15
GROWTH_MIN_RVALUE ?= 0.7

export MPLCONFIGDIR ?= /tmp/psc-matplotlib

# Figure theme, shared by every plotting script (see plot_style.py).
#   paper -> white background, 300 dpi, a PDF next to every PNG. Thesis and
#            journal figures: A&A/ApJ/JGR want white or transparent.
#   dark  -> the previous GitHub-dark look, for screen and slides.
export PSC_FIG_THEME ?= paper

.PHONY: all analysis thesis common manifest spectrum F_S_bM F_M_bM F_W_bM M_S_bM M_M_bM M_W_bM \
	W_S_bM W_M_bM W_W_bM brazil mirror structures diamagnetic fields spectral dispersion growth-map \
	residuals energy-exchange estimators convergence \
	anisotropy-dispersion \
	polarization validate particles heatflux physics compare-physics clean help \
	theory theory-self-test prt-region prt-bfield vdf-spatial \
	theory-liouville theory-liouville-x theory-liouville-sweep \
	liouville-self-test kappa-eff-self-test reconnection \
	check-fields check-moments check-particles check-case check-resolution \
	check-initial-conditions show-inputs

all: analysis

common: brazil fields particles spectral diamagnetic heatflux validate physics \
	energy-if-present residuals structures energy-exchange estimators

analysis: manifest common

# Everything a thesis chapter needs for one run, in dependency order: the
# initial-condition check and manifest first, then the validation evidence
# (energy, field residuals, estimators), then the physics.
thesis: manifest energy-if-present residuals physics brazil spectral structures \
	energy-exchange estimators heatflux particles validate diamagnetic fields
	@echo "Thesis products for $(CASE) in $(abspath $(OUTPUT_DIR))"

F_S_bM F_M_bM F_W_bM M_S_bM M_M_bM M_W_bM W_S_bM W_M_bM W_W_bM:
	$(MAKE) analysis DATA_DIR="$(DATA_DIR)" CASE="$@" RESULTS_ROOT="$(RESULTS_ROOT)"

manifest: check-case check-initial-conditions
	PSC_PROFILE=$(PSC_PROFILE) $(PYTHON) write_analysis_manifest.py \
		--data-dir "$(DATA_DIR)" --output-dir "$(OUTPUT_DIR)" --case "$(CASE)" \
		$(if $(RUN_TAG),--run-tag "$(RUN_TAG)",)

show-inputs:
	@echo "Run name:       $(RUN_NAME)"
	@echo "Case:           $(CASE)"
	@echo "Data directory: $(abspath $(DATA_DIR))"
	@echo "Profile:        $(PSC_PROFILE)"
	@echo "Results:        $(abspath $(OUTPUT_DIR))"
	@echo "Fields:         $$(find "$(DATA_DIR)" -maxdepth 1 -name 'pfd.*_p*.h5' | wc -l)"
	@echo "Moments:        $$(find "$(DATA_DIR)" -maxdepth 1 -name 'pfd_moments.*_p*.h5' | wc -l)"
	@echo "Particles:      $$(find "$(DATA_DIR)" -maxdepth 1 -name 'prt*.h5' | wc -l)"
	@echo "Particle base:  $(PARTICLE_BASENAME)"

check-case:
	@test -n "$(CASE)" || { \
		echo "ERROR: CASE is empty and could not be detected from prt_<CASE>.<step>.h5"; \
		echo "Use: make analysis DATA_DIR=../run CASE=F_M_bM"; exit 1; \
	}
	@count=$$($(PRT_SERIES) | wc -l); \
	if test "$$count" -gt 1; then \
		echo "ERROR: multiple particle series found in $(DATA_DIR)."; \
		echo "Put each PSC case in its own directory."; exit 1; \
	fi
	@if test -n "$(DETECTED_CASE)" && test "$(DETECTED_CASE)" != "$(CASE)"; then \
		echo "ERROR: CASE=$(CASE), but particle files identify $(DETECTED_CASE)."; \
		exit 1; \
	fi

check-fields:
	@test -n "$(firstword $(wildcard $(FIELD_PATTERN)))" || { \
		echo "ERROR: no field files matched $(FIELD_PATTERN)"; exit 1; \
	}

check-moments:
	@test -n "$(firstword $(wildcard $(MOMENT_PATTERN)))" || { \
		echo "ERROR: no moment files matched $(MOMENT_PATTERN)"; exit 1; \
	}

check-particles:
	@test -n "$(FIRST_PARTICLE)" || { \
		echo "ERROR: no particle files matched $(PARTICLE_PATTERN)"; exit 1; \
	}

# Compares the grid resolution baked into the PSC_PROFILE (psc_units.py)
# against the shape actually stored in one real snapshot. A run whose
# resolution doesn't match its declared profile silently corrupts every
# downstream reshape/plot instead of failing loudly, so this runs before
# any target that touches field or moment files.
check-resolution: check-case
	@PSC_PROFILE=$(PSC_PROFILE) $(PYTHON) check_resolution.py \
		--data-dir "$(DATA_DIR)" --case "$(CASE)"

# Compares the T_par/T_perp/n/B0 measured at the earliest snapshot against
# what PSC_PROFILE declares for CASE. Catches pointing an analysis at the
# wrong run directory, or at a CASE whose profile has drifted from the .cxx
# that produced the data; runs once, before the manifest, not per stage.
check-initial-conditions: check-case
	@PSC_PROFILE=$(PSC_PROFILE) $(PYTHON) check_initial_conditions.py \
		--particles "$(PARTICLE_PATTERN)" --fields "$(FIELD_PATTERN)" \
		--moments "$(MOMENT_PATTERN)" --case "$(CASE)"

brazil: check-case check-fields check-moments check-resolution
	@echo "Generating anisotropy evolution and Brazil trajectory"
	PSC_PROFILE=$(PSC_PROFILE) $(PYTHON) anisotropy_analysis.py \
		--data-dir "$(DATA_DIR)" --B0 $(B0_REF) \
		--run-name "$(RUN_NAME)" --outdir "$(ANISOTROPY_OUT)"

# Saturated structures (holes/peaks, |B| skewness, n-|B| correlation, total-
# pressure balance). Replaces the old mirror_physics.py maps (now in legacy/);
# `make mirror` is kept as an alias.
structures: check-case check-fields check-resolution
	@echo "Measuring saturated magnetic structures and pressure balance"
	PSC_PROFILE=$(PSC_PROFILE) $(PYTHON) structures_analysis.py \
		--data-dir "$(DATA_DIR)" --fields "$(FIELD_PATTERN)" --moments "$(MOMENT_PATTERN)" \
		--outdir "$(STRUCTURES_OUT)" $(STRUCTURES_FLAGS)

mirror: structures

diamagnetic: check-case check-fields check-moments check-resolution
	@echo "Generating diamagnetic-current plots"
	PSC_PROFILE=$(PSC_PROFILE) $(PYTHON) diamagnetic_current.py \
		--moments "$(MOMENT_PATTERN)" --fields "$(FIELD_PATTERN)" \
		--outdir "$(DIAMAGNETIC_OUT)" \
		--every $(SNAPSHOT_EVERY) --gif-every $(GIF_EVERY)

fields: check-case check-fields check-resolution
	@echo "Generating magnetic-field fluctuation plots"
	PSC_PROFILE=$(PSC_PROFILE) $(PYTHON) fluctuationofmagneticfiel.py \
		--B0 $(B0_REF) --plane yz --smooth 1.5 \
		--pattern "$(FIELD_PATTERN)" --outdir "$(FIELDS_OUT)" \
		--every $(SNAPSHOT_EVERY) --gif_every $(GIF_EVERY) \
		$(if $(FIELDS_GIF),--create_gifs --fixed_scale,)

# spectrum = gamma(k), E(k,t), helicity, compressibility only. `spectral` runs
# it and then the four other spectral products; a cluster job can instead run
# spectrum / dispersion / growth-map (x2) / polarization as parallel steps.
spectrum: check-case check-fields check-resolution
	@echo "Generating spectral plots"
	PSC_PROFILE=$(PSC_PROFILE) $(PYTHON) spectral_analysis.py \
		--fields "$(FIELD_PATTERN)" --outdir "$(SPECTRAL_OUT)" \
		$(if $(GROWTH_T_START),--t-start "$(GROWTH_T_START)",) \
		$(if $(GROWTH_T_END),--t-end "$(GROWTH_T_END)",)

spectral: spectrum
	$(MAKE) dispersion DATA_DIR="$(DATA_DIR)" CASE="$(CASE)" \
		RESULTS_ROOT="$(RESULTS_ROOT)"
	$(MAKE) growth-map DATA_DIR="$(DATA_DIR)" CASE="$(CASE)" \
		RESULTS_ROOT="$(RESULTS_ROOT)" GROWTH_COMPONENT=parallel
	$(MAKE) growth-map DATA_DIR="$(DATA_DIR)" CASE="$(CASE)" \
		RESULTS_ROOT="$(RESULTS_ROOT)" GROWTH_COMPONENT=perp
	$(MAKE) polarization DATA_DIR="$(DATA_DIR)" CASE="$(CASE)" \
		RESULTS_ROOT="$(RESULTS_ROOT)" THEORY_CSV="$(THEORY_CSV)"

dispersion: check-case check-fields check-resolution
	@echo "Measuring spatial modes, growth and frequency; generating dispersion diagrams"
	PSC_PROFILE=$(PSC_PROFILE) $(PYTHON) dispersion_analysis.py \
		--fields "$(FIELD_PATTERN)" --outdir "$(SPECTRAL_OUT)" \
		--mode "$(DISPERSION_MODE)" \
		--component "$(DISPERSION_COMPONENT)" \
		$(if $(DISPERSION_THETA_MAX),--theta-max-deg "$(DISPERSION_THETA_MAX)",) \
		$(if $(DISPERSION_THETA_MIN),--theta-min-deg "$(DISPERSION_THETA_MIN)",) \
		$(if $(DISPERSION_KMAX_DI),--kmax-di "$(DISPERSION_KMAX_DI)",) \
		$(if $(DISPERSION_DEGROWTH),--degrowth "$(DISPERSION_DEGROWTH)",) \
		$(if $(DISPERSION_OMEGA_SCALE),--omega-scale "$(DISPERSION_OMEGA_SCALE)",) \
		$(if $(DISPERSION_SPATIAL_WINDOW),--spatial-window "$(DISPERSION_SPATIAL_WINDOW)",) \
		$(if $(DISPERSION_TEMPORAL_WINDOW),--temporal-window "$(DISPERSION_TEMPORAL_WINDOW)",) \
		$(if $(DISPERSION_T_START),--t-start "$(DISPERSION_T_START)",) \
		$(if $(DISPERSION_T_END),--t-end "$(DISPERSION_T_END)",)

growth-map: check-case check-fields check-resolution
	@echo "Generating gamma(k_parallel,k_perp) growth-rate map"
	PSC_PROFILE=$(PSC_PROFILE) $(PYTHON) growth_rate_map.py \
		--fields "$(FIELD_PATTERN)" --outdir "$(SPECTRAL_OUT)" \
		--component "$(GROWTH_COMPONENT)" \
		$(if $(GROWTH_T_START),--t-start "$(GROWTH_T_START)",) \
		$(if $(GROWTH_T_END),--t-end "$(GROWTH_T_END)",) \
		$(if $(GROWTH_KPAR_MAX),--kpar-max "$(GROWTH_KPAR_MAX)",) \
		$(if $(GROWTH_KPERP_MAX),--kperp-max "$(GROWTH_KPERP_MAX)",) \
		$(if $(GROWTH_DISPLAY_KPAR_MAX),--display-kpar-max "$(GROWTH_DISPLAY_KPAR_MAX)",) \
		$(if $(GROWTH_DISPLAY_KPERP_MAX),--display-kperp-max "$(GROWTH_DISPLAY_KPERP_MAX)",) \
		--shading "$(GROWTH_SHADING)" --contours "$(GROWTH_CONTOURS)" \
		--angle-step "$(GROWTH_ANGLE_STEP)" --min-rvalue "$(GROWTH_MIN_RVALUE)"

anisotropy-dispersion: check-case check-fields check-moments check-resolution
	@echo "Generating temperature-anisotropy dispersion plots"
	PSC_PROFILE=$(PSC_PROFILE) $(PYTHON) temperature_anisotropy_dispersion.py \
		--moments "$(MOMENT_PATTERN)" --fields "$(FIELD_PATTERN)" \
		--outdir "$(SPECTRAL_OUT)"

# omega-k_parallel maps (P+, P-, sigma_m) via the circular polarization
# psi_pm = (dBx +- i dBy)/B0, plus per-mode growth curves with a theory
# overlay (THEORY_CSV: kdi,omega_r_over_Omegai,gamma_over_Omegai), and the
# mirror delta_Bz(k_perp,k_parallel) oblique spectrum for mirror cases.
polarization: check-case check-fields check-resolution
	@echo "Generating polarized omega-k dispersion maps and mode-growth curves"
	PSC_PROFILE=$(PSC_PROFILE) $(PYTHON) polarization_dispersion.py \
		--fields "$(FIELD_PATTERN)" --outdir "$(SPECTRAL_OUT)" \
		$(if $(THEORY_CSV),--theory-csv "$(THEORY_CSV)",) \
		$(if $(DISPERSION_T_START),--t-start "$(DISPERSION_T_START)",) \
		$(if $(DISPERSION_T_END),--t-end "$(DISPERSION_T_END)",) $(POLARIZATION_FLAGS)

validate: check-case check-particles
	@echo "Validating moments against $(FIRST_PARTICLE)"
	PSC_PROFILE=$(PSC_PROFILE) $(PYTHON) validate_moments.py "$(FIRST_PARTICLE)" \
		--run-name "$(RUN_NAME)" --outdir "$(VALIDATION_OUT)"

particles: check-case check-particles
	@echo "Generating particle VDF, anisotropy, and Brazil trajectory"
	PSC_PROFILE=$(PSC_PROFILE) $(PYTHON) plot_prt.py \
		"$(PARTICLE_PATTERN)" --run-name "$(RUN_NAME)" \
		--outdir "$(PARTICLES_OUT)"

# Third central moment of the particle VDF in the local frame of the prt
# window, truncated at s_max (needed for kappa = 3) with sampling errors.
heatflux: check-case check-particles check-fields check-resolution
	@echo "Measuring the particle heat flux (third moment) in the prt window"
	PSC_PROFILE=$(PSC_PROFILE) $(PYTHON) heat_flux_analysis.py \
		--data-dir "$(DATA_DIR)" --particles "$(PARTICLE_PATTERN)" \
		--fields "$(FIELD_PATTERN)" --outdir "$(HEATFLUX_OUT)" $(HEATFLUX_FLAGS)

physics: check-case check-resolution
	@echo "Generating integrated physical diagnostics"
	PSC_PROFILE=$(PSC_PROFILE) $(PYTHON) physical_diagnostics.py \
		--data-dir "$(DATA_DIR)" --particles "$(PARTICLE_PATTERN)" \
		--fields "$(FIELD_PATTERN)" --moments "$(MOMENT_PATTERN)" \
		--outdir "$(PHYSICS_OUT)" \
		$(if $(GROWTH_T_START),--growth-t-start "$(GROWTH_T_START)",) \
		$(if $(GROWTH_T_END),--growth-t-end "$(GROWTH_T_END)",)

theory: check-case
	@echo "Solving the parallel linear dispersion relation for $(CASE)"
	PSC_PROFILE=$(PSC_PROFILE) $(PYTHON) linear_theory.py --case "$(CASE)" \
		--out "$(SPECTRAL_OUT)/linear_theory.csv" $(THEORY_FLAGS)
	@echo "Now: make polarization DATA_DIR=... CASE=$(CASE) THEORY_CSV=$(SPECTRAL_OUT)/linear_theory.csv"

theory-self-test:
	$(PYTHON) linear_theory.py --self-test

.PHONY: energy energy-if-present
energy-if-present: check-case
	@if test -n "$(wildcard $(DATA_DIR)/diag*.asc)"; then \
		$(MAKE) energy; \
	else echo "[MISSING] No diag*.asc: global energy conservation cannot be evaluated"; fi

energy: check-case
	PSC_PROFILE=$(PSC_PROFILE) $(PYTHON) energy_conservation.py \
		$(DATA_DIR)/diag*.asc --outdir "$(PHYSICS_OUT)"

# div B of the Yee field snapshots plus PSC's own Gauss / continuity checks
# parsed from the job log (LOGS='path/to/job.out' if it is elsewhere).
residuals: check-case check-fields
	@echo "Field-solver residuals: div B, Gauss, continuity"
	PSC_PROFILE=$(PSC_PROFILE) $(PYTHON) field_residuals.py \
		--data-dir "$(DATA_DIR)" --fields "$(FIELD_PATTERN)" --outdir "$(PHYSICS_OUT)" \
		$(if $(strip $(LOGS)),--logs $(LOGS),)

# J_s.E per species (parallel/perpendicular), integrated in time and checked
# against the kinetic-energy change from DiagEnergies.
energy-exchange: check-case check-fields check-moments check-resolution
	@echo "Field-particle energy exchange J.E per species"
	PSC_PROFILE=$(PSC_PROFILE) $(PYTHON) energy_exchange.py \
		--data-dir "$(DATA_DIR)" --fields "$(FIELD_PATTERN)" --moments "$(MOMENT_PATTERN)" \
		--outdir "$(PHYSICS_OUT)"

# The four anisotropy estimators of the pipeline on one time axis.
estimators: check-case check-fields check-moments check-particles check-resolution
	@echo "Comparing particle and moment anisotropy estimators"
	PSC_PROFILE=$(PSC_PROFILE) $(PYTHON) estimator_consistency.py \
		--data-dir "$(DATA_DIR)" --outdir "$(PHYSICS_OUT)"

# Adiabatic Liouville mapping + trapped-domain closures (paper fig. 2).
# Defaults (kappa0, A0) come from the active PSC_PROFILE; override with
# LIOUVILLE_FLAGS='--kappa0 3 --A0 2'.
theory-liouville: check-case
	@echo "Computing Liouville closure profiles for $(CASE)"
	PSC_PROFILE=$(PSC_PROFILE) $(PYTHON) liouville_kappa.py --figure \
		--outdir "$(SPECTRAL_OUT)" $(LIOUVILLE_FLAGS)

# The same closures re-labelled along the model structure
# B(x)/B0 = 1 - a sech^2(x/L). The moments depend on x only through b(x), so
# this is a re-labelling of theory-liouville and not a second calculation; it
# is the form that shows how much of the range of b a structure of a given
# depth actually samples. Cost is one quadrature (~1 s) per distinct b, and
# b(x) is even, so roughly LIOUVILLE_NX/2 per closure.
LIOUVILLE_NX ?= 41
LIOUVILLE_A ?=
theory-liouville-x: check-case
	@echo "Computing Liouville closure profiles along B(x) for $(CASE)"
	PSC_PROFILE=$(PSC_PROFILE) $(PYTHON) liouville_kappa.py --figure-x \
		--outdir "$(SPECTRAL_OUT)" --n-x $(LIOUVILLE_NX) \
		$(if $(LIOUVILLE_A),--a $(LIOUVILLE_A),) $(LIOUVILLE_FLAGS)

# Parameter study over (kappa, A0, a). Every combination costs a full
# theory-liouville-x, so keep the lists short and LIOUVILLE_NX small while
# exploring. kappa = 0 in the list means bi-Maxwellian.
LIOUVILLE_KAPPA_LIST ?= 3 5 0
LIOUVILLE_A0_LIST ?=
LIOUVILLE_A_LIST ?= 0.1 0.25 0.4
theory-liouville-sweep: check-case
	@echo "Sweeping Liouville closures over (kappa, A0, a)"
	PSC_PROFILE=$(PSC_PROFILE) $(PYTHON) liouville_kappa.py --sweep \
		--outdir "$(SPECTRAL_OUT)/liouville_sweep" --n-x $(LIOUVILLE_NX) \
		--kappa-list $(LIOUVILLE_KAPPA_LIST) \
		$(if $(LIOUVILLE_A0_LIST),--A0-list $(LIOUVILLE_A0_LIST),) \
		--a-list $(LIOUVILLE_A_LIST) $(LIOUVILLE_FLAGS)

liouville-self-test:
	$(PYTHON) liouville_kappa.py --self-test

kappa-eff-self-test:
	$(PYTHON) kappa_eff.py --self-test

# Reconnection runs (double Harris sheet) live outside the anisotropy case
# matrix: their profiles come from reconnection_analysis.py, not psc_units,
# and the boxes are rectangular. This target therefore skips check-case.
# RECONNECTION_PROFILE: reconnection | reconnection_comparable.
RECONNECTION_PROFILE ?= reconnection_comparable
RECONNECTION_FLAGS ?=
reconnection: check-fields
	@echo "Generating reconnection diagnostics and the B-kappa correlation"
	$(PYTHON) reconnection_analysis.py --data-dir "$(DATA_DIR)" \
		--profile "$(RECONNECTION_PROFILE)" --outdir "$(RECONNECTION_OUT)" \
		$(RECONNECTION_FLAGS)

prt-region: check-case check-fields
	@echo "Locating the prt window in the field-fluctuation maps"
	PSC_PROFILE=$(PSC_PROFILE) $(PYTHON) prt_region_field_cut.py \
		--data-dir "$(DATA_DIR)" --outdir "$(PARTICLES_OUT)" \
		$(if $(STEPS),--steps $(STEPS),) --average-over-window

# Time-averaged <|B|>, B_rms and dB_rms inside the prt window, with the PIC
# noise floor removed. PRT_BFIELD_FLAGS takes e.g.
# '--avg-window 100 158 --noise-window 0 0.3 --kmax-de 2.0 --floor static'.
prt-bfield: check-case check-fields
	@echo "Time-averaged B statistics inside the prt window (noise-cleaned)"
	PSC_PROFILE=$(PSC_PROFILE) $(PYTHON) prt_region_bfield_stats.py \
		--data-dir "$(DATA_DIR)" --outdir "$(PHYSICS_OUT)" $(PRT_BFIELD_FLAGS)

vdf-spatial: check-case check-fields check-particles
	@echo "Computing the spatially resolved VDF inside the prt window"
	PSC_PROFILE=$(PSC_PROFILE) $(PYTHON) vdf_spatial.py \
		--data-dir "$(DATA_DIR)" --outdir "$(PARTICLES_OUT)" \
		$(if $(STEPS),--steps $(STEPS),) $(VDF_FLAGS)

compare-physics:
	@test -n "$(COMPARE_CASES)" || { \
		echo "ERROR: set COMPARE_CASES='maxwellian=../analysis_results/mirror_maxwellian/09_physical_diagnostics kappa=../analysis_results/mirror_kappa/09_physical_diagnostics'"; \
		exit 1; \
	}
	$(PYTHON) compare_physical_cases.py $(COMPARE_CASES) --outdir "$(COMPARE_OUT)"

convergence:
	@test -n "$(CONVERGENCE_RUNS)" || { \
		echo "ERROR: set CONVERGENCE_RUNS='base=../analysis_results/run_a dx2=../analysis_results/run_b ...'"; \
		exit 1; \
	}
	$(PYTHON) convergence_study.py $(CONVERGENCE_RUNS) --outdir "$(CONVERGENCE_OUT)"

clean:
	@test -n "$(RUN_NAME)" && test "$(OUTPUT_DIR)" != "/" && rm -rf "$(OUTPUT_DIR)"

help:
	@echo "Usage:"
	@echo "  make analysis DATA_DIR=../path/to/run CASE=mirror_bimaxwellian_moderate"
	@echo "  make brazil DATA_DIR=../path/to/run CASE=mirror_bimaxwellian_moderate"
	@echo "  make particles DATA_DIR=../path/to/run CASE=mirror_bikappa3"
	@echo "  make F_M_bM DATA_DIR=../path/to/F_M_bM      # legacy profile"
	@echo "  make compare-physics COMPARE_CASES='maxwellian=../analysis_results/mirror_maxwellian/09_physical_diagnostics kappa=../analysis_results/mirror_kappa/09_physical_diagnostics'"
	@echo ""
	@echo "Current cases: mirror_bimaxwellian_{strong,moderate,weak}  mirror_bikappa3 mirror_bikappa5"
	@echo "               mirror_bikappa3_moderate mirror_bikappa5_moderate"
	@echo "               firehose_bimaxwellian_{strong,moderate,weak}  firehose_bikappa3 firehose_bikappa5"
	@echo "               firehose_bimaxwellian_{strong,moderate}_bigbox40  firehose_bikappa{3,5}_bigbox40"
	@echo "               whistler_bimaxwellian_{strong,moderate,weak}  whistler_bikappa3_moderate"
	@echo "Legacy cases:  F_S_bM F_M_bM F_W_bM M_S_bM M_M_bM M_W_bM W_S_bM W_M_bM W_W_bM"
	@echo "Main targets: thesis analysis brazil particles fields spectral dispersion growth-map polarization"
	@echo "              diamagnetic heatflux validate physics structures residuals energy-exchange estimators"
	@echo "              compare-physics convergence anisotropy-dispersion"
	@echo "  make thesis DATA_DIR=... CASE=... [RUN_TAG=seed2] [GROWTH_T_START=.. GROWTH_T_END=..] [LOGS=job.out]"
	@echo "  make convergence CONVERGENCE_RUNS='base=../analysis_results/A dx2=../analysis_results/B'"
	@echo "  make polarization DATA_DIR=... CASE=... THEORY_CSV=theory.csv  # omega-k dispersion (psi_pm), mode growth, mirror oblique spectrum"
	@echo "  make theory CASE=...                          # solve linear dispersion -> linear_theory.csv (fills gamma_theory)"
	@echo "  make theory-self-test                         # validate the solver against known limits"
	@echo "  make prt-region DATA_DIR=... CASE=...          # where the prt window sits in the field fluctuations"
	@echo "  make prt-bfield DATA_DIR=... CASE=...          # time-averaged <|B|>, B_rms, dB_rms in the prt window"
	@echo "  make vdf-spatial DATA_DIR=... CASE=...         # VDF resolved in space; hole vs peak populations + 2D VDF with the trapping boundary"
	@echo "  make theory-liouville CASE=...                # closure profiles vs b (paper fig. 2)"
	@echo "  make theory-liouville-x CASE=...              # the same closures vs x along B(x) = B0[1 - a sech^2(x/L)]"
	@echo "  make theory-liouville-sweep CASE=...          # (kappa, A0, a) study; LIOUVILLE_KAPPA_LIST/A0_LIST/A_LIST, LIOUVILLE_NX"
	@echo "  make reconnection DATA_DIR=... [RECONNECTION_PROFILE=reconnection_comparable]"
	@echo "                                                # double-Harris diagnostics + <|B|> vs kappa_eff correlation"
	@echo "Utility target: show-inputs"
	@echo "Results default to ../analysis_results/<RUN_NAME>/"
