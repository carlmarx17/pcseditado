# Makefile — Analysis Pipeline for Kappa-Anisotropy PIC Simulation
# Targets mirror the Python scripts in this directory.

# Variables
PYTHON   = python3
B0_REF   = 0.05     # Matches vA_over_c = 0.05 in psc_mirror_maxwellian.cxx
DATA_DIR ?= ../build/src
PARTICLE_BASENAME ?= prt_M_S_bM
PARTICLE_PATTERN = $(DATA_DIR)/$(PARTICLE_BASENAME).*.h5
FIRST_PARTICLE = $(firstword $(sort $(wildcard $(PARTICLE_PATTERN))))

.PHONY: all brazil mirror diamagnetic fields spectral validate particles heatflux report clean help check-particles

# spectral_analysis is excluded from 'all' — still in development
all: brazil mirror diamagnetic fields validate particles heatflux

brazil:
	@echo "==========================================================="
	@echo "Generating Brazil Plot (Anisotropy vs Beta)"
	@echo "==========================================================="
	$(PYTHON) anisotropy_analysis.py --B0 $(B0_REF) --moments "$(DATA_DIR)/pfd_moments.*.h5" --fields "$(DATA_DIR)/pfd.*.h5"

mirror:
	@echo "==========================================================="
	@echo "Generating Mirror Instability Plots"
	@echo "==========================================================="
	$(PYTHON) mirror_physics.py --B0 $(B0_REF) --fields "$(DATA_DIR)/pfd.*.h5"

diamagnetic:
	@echo "==========================================================="
	@echo "Generating Diamagnetic Current plots"
	@echo "==========================================================="
	$(PYTHON) diamagnetic_current.py --moments "$(DATA_DIR)/pfd_moments.*.h5" --fields "$(DATA_DIR)/pfd.*.h5"

fields:
	@echo "==========================================================="
	@echo "Visualising Magnetic Field Fluctuations"
	@echo "==========================================================="
	$(PYTHON) fluctuationofmagneticfiel.py --B0 $(B0_REF) --plane yz --smooth 1.5 --pattern "$(DATA_DIR)/pfd.*.h5"

spectral:
	@echo "==========================================================="
	@echo "Generating Spectral Analysis Plots (PSD)"
	@echo "==========================================================="
	$(PYTHON) spectral_analysis.py --fields "$(DATA_DIR)/pfd.*.h5"

check-particles:
	@test -n "$(FIRST_PARTICLE)" || { \
		echo "ERROR: No particle files found: $(PARTICLE_PATTERN)"; \
		exit 1; \
	}

validate: check-particles
	@echo "==========================================================="
	@echo "Validating Moments"
	@echo "==========================================================="
	$(PYTHON) validate_moments.py "$(FIRST_PARTICLE)"

particles: check-particles
	@echo "==========================================================="
	@echo "Generating Particle VDF and Phase-Space Plots"
	@echo "==========================================================="
	$(PYTHON) plot_prt.py "$(PARTICLE_PATTERN)"
	$(PYTHON) plot_vdf_3d.py "$(FIRST_PARTICLE)"
	$(PYTHON) plot_moments_scatter_3d.py "$(FIRST_PARTICLE)"
heatflux:
	@echo "==========================================================="
	@echo "Generating Heat Flux and Anisotropy Analysis"
	@echo "==========================================================="
	$(PYTHON) heat_flux_analysis.py --moments "$(DATA_DIR)/pfd_moments.*.h5" --fields "$(DATA_DIR)/pfd.*.h5"
report:
	@echo "==========================================================="
	@echo "Generating Validation PDF Report"
	@echo "==========================================================="
	$(PYTHON) generar_pdf_reporte.py

clean:
	@echo "==========================================================="
	@echo "Cleaning generated plots"
	@echo "==========================================================="
	rm -rf anisotropy_plots/ mirror_plots/ field_images/ spectral_plots/ prt_plots/ fancy_vdf_plots/ phase_space_plots/ diamagnetic_plots/ heatflux_plots/

help:
	@echo "Available targets:"
	@echo "  make brazil       - Brazil plot: anisotropy vs beta_par  (anisotropy_analysis.py)"
	@echo "  make mirror       - Mirror instability plots             (mirror_physics.py)"
	@echo "  make diamagnetic  - Diamagnetic current analysis + GIF   (diamagnetic_current.py)"
	@echo "  make fields       - Magnetic field fluctuations          (fluctuationofmagneticfiel.py)"
	@echo "  make spectral     - Spectral analysis (PSD)              (spectral_analysis.py)"
	@echo "  make validate     - Moment validation                    (validate_moments.py)"
	@echo "  make particles    - VDFs and phase-space plots           (plot_prt.py, plot_vdf_3d.py, ...)"
	@echo "  make heatflux     - Heat flux + anisotropy analysis      (heat_flux_analysis.py)"
	@echo "  make report       - Generate PDF validation report       (generar_pdf_reporte.py)"
	@echo "  make all          - Run all targets (except spectral and report)"
	@echo "  make clean        - Remove all generated plot directories"
	@echo "  DATA_DIR=path     - Override input directory (default: ../build/src)"
	@echo "  PARTICLE_BASENAME=name - Particle prefix (default: prt_M_S_bM)"
	@echo ""
	@echo "  NOTE: spectral_analysis.py is excluded from 'all' (still in development)"
