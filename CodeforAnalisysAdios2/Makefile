# Makefile — Analysis Pipeline for Kappa-Anisotropy PIC Simulation
# Targets mirror the Python scripts in this directory.

# Variables
PYTHON   = python3
B0_REF   = 0.1      # Matches vA_over_c = 0.1 in psc_temp_aniso.cxx
DATA_DIR = ../build/src
PARTICLE_PATTERN = $(DATA_DIR)/prt.*.bp
PARTICLE_SAMPLE  = $(lastword $(sort $(wildcard $(PARTICLE_PATTERN))))

.PHONY: all brazil mirror diamagnetic fields spectral validate particles report clean help

# spectral_analysis is excluded from 'all' — still in development
all: brazil mirror diamagnetic fields validate particles

brazil:
	@echo "==========================================================="
	@echo "Generating Brazil Plot (Anisotropy vs Beta)"
	@echo "==========================================================="
	$(PYTHON) anisotropy_analysis.py --B0 $(B0_REF) --moments "$(DATA_DIR)/pfd_moments.*.bp" --fields "$(DATA_DIR)/pfd.*.bp"

mirror:
	@echo "==========================================================="
	@echo "Generating Mirror Instability Plots"
	@echo "==========================================================="
	$(PYTHON) mirror_physics.py --B0 $(B0_REF) --fields "$(DATA_DIR)/pfd.*.bp"

diamagnetic:
	@echo "==========================================================="
	@echo "Generating Diamagnetic Current plots"
	@echo "==========================================================="
	$(PYTHON) diamagnetic_current.py --moments "$(DATA_DIR)/pfd_moments.*.bp" --fields "$(DATA_DIR)/pfd.*.bp"

fields:
	@echo "==========================================================="
	@echo "Visualising Magnetic Field Fluctuations"
	@echo "==========================================================="
	$(PYTHON) fluctuationofmagneticfiel.py --B0 $(B0_REF) --plane yz --smooth 1.5 --pattern "$(DATA_DIR)/pfd.*.bp"

spectral:
	@echo "==========================================================="
	@echo "Generating Spectral Analysis Plots (PSD)"
	@echo "==========================================================="
	$(PYTHON) spectral_analysis.py --fields "$(DATA_DIR)/pfd.*.bp"

validate:
	@echo "==========================================================="
	@echo "Validating Moments"
	@echo "==========================================================="
	@if [ -z "$(PARTICLE_SAMPLE)" ]; then echo "No particle BP files found matching $(PARTICLE_PATTERN)"; exit 1; fi
	$(PYTHON) validate_moments.py "$(PARTICLE_SAMPLE)"
particles:
	@echo "==========================================================="
	@echo "Generating Particle VDF and Phase-Space Plots"
	@echo "==========================================================="
	@if [ -z "$(PARTICLE_SAMPLE)" ]; then echo "No particle BP files found matching $(PARTICLE_PATTERN)"; exit 1; fi
	$(PYTHON) plot_prt.py "$(PARTICLE_PATTERN)"
	$(PYTHON) plot_vdf_3d.py "$(PARTICLE_SAMPLE)"
	$(PYTHON) plot_moments_scatter_3d.py "$(PARTICLE_SAMPLE)"
report:
	@echo "==========================================================="
	@echo "Generating Validation PDF Report"
	@echo "==========================================================="
	$(PYTHON) generar_pdf_reporte.py

clean:
	@echo "==========================================================="
	@echo "Cleaning generated plots"
	@echo "==========================================================="
	rm -rf anisotropy_plots/ mirror_plots/ field_images/ spectral_plots/ prt_plots/ fancy_vdf_plots/ phase_space_plots/ diamagnetic_plots/

help:
	@echo "Available targets:"
	@echo "  make brazil       - Brazil plot: anisotropy vs beta_par  (anisotropy_analysis.py)"
	@echo "  make mirror       - Mirror instability plots             (mirror_physics.py)"
	@echo "  make diamagnetic  - Diamagnetic current analysis + GIF   (diamagnetic_current.py)"
	@echo "  make fields       - Magnetic field fluctuations          (fluctuationofmagneticfiel.py)"
	@echo "  make spectral     - Spectral analysis (PSD)              (spectral_analysis.py)"
	@echo "  make validate     - Moment validation                    (validate_moments.py)"
	@echo "  make particles    - VDFs and phase-space plots           (plot_prt.py, plot_vdf_3d.py, ...)"
	@echo "  make report       - Generate PDF validation report       (generar_pdf_reporte.py)"
	@echo "  make all          - Run all targets (except spectral and report)"
	@echo "  make clean        - Remove all generated plot directories"
	@echo ""
	@echo "  NOTE: spectral_analysis.py is excluded from 'all' (still in development)"
