# Analysis pipeline for PSC outputs.
#
# Paths supplied on the command line are interpreted relative to this
# directory. Results are grouped by RUN_NAME so different simulations never
# overwrite each other.

PYTHON ?= $(abspath ../.venv/bin/python)
DATA_DIR ?= ../build/src
DETECTED_CASE = $(shell detected=$$(find "$(DATA_DIR)" -maxdepth 1 -type f \
	-name 'prt_*.h5' -printf '%f\n' 2>/dev/null | sed -E \
	's/^(prt_)?([^.]+)\.[0-9]+(_p[0-9]+)?\.h5$$/\2/' | sort -u | head -1); \
	if test -n "$$detected"; then echo "$$detected"; \
	else basename "$(abspath $(DATA_DIR))" | grep -E '^[FMW]_[SMW]_bM$$' || true; fi)
CASE ?= $(DETECTED_CASE)
RUN_NAME ?= $(if $(CASE),$(CASE),$(notdir $(abspath $(DATA_DIR))))
RESULTS_ROOT ?= ../analysis_results
OUTPUT_DIR ?= $(RESULTS_ROOT)/$(RUN_NAME)

B0_REF ?= 0.08
PSC_PROFILE ?= $(CASE)
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
MIRROR_OUT = $(OUTPUT_DIR)/07_mirror_structures
VALIDATION_OUT = $(OUTPUT_DIR)/08_validation
PHYSICS_OUT = $(OUTPUT_DIR)/09_physical_diagnostics
COMPARE_CASES ?=
COMPARE_OUT ?= $(RESULTS_ROOT)/comparison_physical
THEORY_CSV ?=

export MPLCONFIGDIR ?= /tmp/psc-matplotlib

.PHONY: all analysis common manifest F_S_bM F_M_bM F_W_bM M_S_bM M_M_bM M_W_bM \
	W_S_bM W_M_bM W_W_bM brazil mirror diamagnetic fields spectral dispersion \
	polarization validate particles heatflux physics compare-physics clean help \
	check-fields check-moments check-particles check-case check-resolution show-inputs

all: analysis

common: brazil fields particles spectral diamagnetic heatflux validate physics

analysis: manifest common
	@if printf '%s' "$(CASE)" | grep -q '^M_'; then \
		$(MAKE) mirror DATA_DIR="$(DATA_DIR)" CASE="$(CASE)" \
			RESULTS_ROOT="$(RESULTS_ROOT)"; \
	fi

F_S_bM F_M_bM F_W_bM M_S_bM M_M_bM M_W_bM W_S_bM W_M_bM W_W_bM:
	$(MAKE) analysis DATA_DIR="$(DATA_DIR)" CASE="$@" RESULTS_ROOT="$(RESULTS_ROOT)"

manifest: check-case
	PSC_PROFILE=$(PSC_PROFILE) $(PYTHON) write_analysis_manifest.py \
		--data-dir "$(DATA_DIR)" --output-dir "$(OUTPUT_DIR)" --case "$(CASE)"

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
	@count=$$(find "$(DATA_DIR)" -maxdepth 1 -type f -name 'prt_*.h5' \
		-printf '%f\n' 2>/dev/null | sed -E \
		's/^(prt_)?([^.]+)\.[0-9]+(_p[0-9]+)?\.h5$$/\2/' | sort -u | wc -l); \
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

brazil: check-case check-fields check-moments check-resolution
	@echo "Generating anisotropy evolution and Brazil trajectory"
	PSC_PROFILE=$(PSC_PROFILE) $(PYTHON) anisotropy_analysis.py \
		--data-dir "$(DATA_DIR)" --B0 $(B0_REF) \
		--run-name "$(RUN_NAME)" --outdir "$(ANISOTROPY_OUT)"

mirror: check-case check-fields check-resolution
	@echo "Generating mirror-structure plots"
	PSC_PROFILE=$(PSC_PROFILE) $(PYTHON) mirror_physics.py \
		--B0 $(B0_REF) --fields "$(FIELD_PATTERN)" --outdir "$(MIRROR_OUT)"

diamagnetic: check-case check-fields check-moments check-resolution
	@echo "Generating diamagnetic-current plots"
	PSC_PROFILE=$(PSC_PROFILE) $(PYTHON) diamagnetic_current.py \
		--moments "$(MOMENT_PATTERN)" --fields "$(FIELD_PATTERN)" \
		--outdir "$(DIAMAGNETIC_OUT)"

fields: check-case check-fields check-resolution
	@echo "Generating magnetic-field fluctuation plots"
	PSC_PROFILE=$(PSC_PROFILE) $(PYTHON) fluctuationofmagneticfiel.py \
		--B0 $(B0_REF) --plane yz --smooth 1.5 \
		--pattern "$(FIELD_PATTERN)" --outdir "$(FIELDS_OUT)"

spectral: check-case check-fields check-resolution
	@echo "Generating spectral plots"
	PSC_PROFILE=$(PSC_PROFILE) $(PYTHON) spectral_analysis.py \
		--fields "$(FIELD_PATTERN)" --outdir "$(SPECTRAL_OUT)"
	$(MAKE) dispersion DATA_DIR="$(DATA_DIR)" CASE="$(CASE)" \
		RESULTS_ROOT="$(RESULTS_ROOT)"
	$(MAKE) polarization DATA_DIR="$(DATA_DIR)" CASE="$(CASE)" \
		RESULTS_ROOT="$(RESULTS_ROOT)" THEORY_CSV="$(THEORY_CSV)"

dispersion: check-case check-fields check-resolution
	@echo "Generating frequency--phase-velocity mode-density plot"
	PSC_PROFILE=$(PSC_PROFILE) $(PYTHON) dispersion_analysis.py \
		--fields "$(FIELD_PATTERN)" --outdir "$(SPECTRAL_OUT)"

# omega-k_parallel maps (P+, P-, sigma_m) via the circular polarization
# psi_pm = (dBx +- i dBy)/B0, plus per-mode growth curves with a theory
# overlay (THEORY_CSV: kdi,omega_r_over_Omegai,gamma_over_Omegai), and the
# mirror delta_Bz(k_perp,k_parallel) oblique spectrum for mirror cases.
polarization: check-case check-fields check-resolution
	@echo "Generating polarized omega-k dispersion maps and mode-growth curves"
	PSC_PROFILE=$(PSC_PROFILE) $(PYTHON) polarization_dispersion.py \
		--fields "$(FIELD_PATTERN)" --outdir "$(SPECTRAL_OUT)" \
		$(if $(THEORY_CSV),--theory-csv "$(THEORY_CSV)",)

validate: check-case check-particles
	@echo "Validating moments against $(FIRST_PARTICLE)"
	PSC_PROFILE=$(PSC_PROFILE) $(PYTHON) validate_moments.py "$(FIRST_PARTICLE)" \
		--run-name "$(RUN_NAME)" --outdir "$(VALIDATION_OUT)"

particles: check-case check-particles
	@echo "Generating particle VDF, anisotropy, and Brazil trajectory"
	PSC_PROFILE=$(PSC_PROFILE) $(PYTHON) plot_prt.py \
		"$(PARTICLE_PATTERN)" --run-name "$(RUN_NAME)" \
		--outdir "$(PARTICLES_OUT)"

heatflux: check-case check-fields check-moments check-resolution
	@echo "Generating heat-flux plots"
	PSC_PROFILE=$(PSC_PROFILE) $(PYTHON) heat_flux_analysis.py \
		--moments "$(MOMENT_PATTERN)" --fields "$(FIELD_PATTERN)" \
		--outdir "$(HEATFLUX_OUT)"

physics: check-case check-resolution
	@echo "Generating integrated physical diagnostics"
	PSC_PROFILE=$(PSC_PROFILE) $(PYTHON) physical_diagnostics.py \
		--data-dir "$(DATA_DIR)" --particles "$(PARTICLE_PATTERN)" \
		--fields "$(FIELD_PATTERN)" --moments "$(MOMENT_PATTERN)" \
		--outdir "$(PHYSICS_OUT)"

compare-physics:
	@test -n "$(COMPARE_CASES)" || { \
		echo "ERROR: set COMPARE_CASES='maxwellian=../analysis_results/mirror_maxwellian/09_physical_diagnostics kappa=../analysis_results/mirror_kappa/09_physical_diagnostics'"; \
		exit 1; \
	}
	$(PYTHON) compare_physical_cases.py $(COMPARE_CASES) --outdir "$(COMPARE_OUT)"

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
	@echo "Current cases: mirror_bimaxwellian_strong mirror_bimaxwellian_moderate mirror_bimaxwellian_weak"
	@echo "               firehose_bimaxwellian_strong firehose_bimaxwellian_moderate firehose_bimaxwellian_weak"
	@echo "               whistler_bimaxwellian_strong whistler_bimaxwellian_moderate whistler_bimaxwellian_weak"
	@echo "               mirror_bikappa3 mirror_bikappa5 firehose_bikappa3 firehose_bikappa5"
	@echo "Legacy cases:  F_S_bM F_M_bM F_W_bM M_S_bM M_M_bM M_W_bM W_S_bM W_M_bM W_W_bM"
	@echo "Main targets: analysis brazil particles fields spectral polarization diamagnetic heatflux validate physics compare-physics"
	@echo "  make polarization DATA_DIR=... CASE=... THEORY_CSV=theory.csv  # omega-k dispersion (psi_pm), mode growth, mirror oblique spectrum"
	@echo "Utility target: show-inputs"
	@echo "Results default to ../analysis_results/<RUN_NAME>/"
