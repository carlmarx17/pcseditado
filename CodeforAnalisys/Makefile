# Makefile to run Analysis Code for Anisotropy and Mirror Instability

# Variables
PYTHON = python3
B0_REF = 0.1 # This aligns with vA_over_c = 0.1 in psc_temp_aniso.cxx
DATA_DIR = ../build/src

.PHONY: all brazil mirror fields spectral validate clean particles

all: brazil mirror fields spectral validate particles

brazil:
	@echo "==========================================================="
	@echo "Generating Brazil Plot for Anisotropy"
	@echo "==========================================================="
	$(PYTHON) anisotropy_analysis.py --B0 $(B0_REF) --moments "$(DATA_DIR)/pfd_moments.*.h5" --fields "$(DATA_DIR)/pfd.*.h5"

mirror:
	@echo "==========================================================="
	@echo "Generating Mirror Instability plots"
	@echo "==========================================================="
	# Note: mirror_physics.py needs update to accept pattern, or we run from there
	$(PYTHON) -c "from mirror_physics import MirrorPhysicsPlotter; p = MirrorPhysicsPlotter(B0=$(B0_REF)); p.process_file(1000, '$(DATA_DIR)/pfd.001000_p000000.h5')"

fields:
	@echo "==========================================================="
	@echo "Visualizing Magnetic Field Fluctuations"
	@echo "==========================================================="
	$(PYTHON) fluctuationofmagneticfiel.py --B0 $(B0_REF) --plane yz --smooth 1.5 --pattern "$(DATA_DIR)/pfd.*.h5"

spectral:
	@echo "==========================================================="
	@echo "Computing Magnetic Field Power Spectrum"
	@echo "==========================================================="
	$(PYTHON) spectral_analysis.py --B0 $(B0_REF) --plane yz --fields "$(DATA_DIR)/pfd.*.h5"

validate:
	@echo "==========================================================="
	@echo "Validating Moments"
	@echo "==========================================================="
	$(PYTHON) validate_moments.py $(DATA_DIR)/prt.000001000.h5

particles:
	@echo "==========================================================="
	@echo "Generating Particle VDF and Phase Space Plots"
	@echo "==========================================================="
	$(PYTHON) plot_prt.py $(DATA_DIR)/prt.000001000.h5
	$(PYTHON) plot_vdf_3d.py $(DATA_DIR)/prt.000001000.h5
	$(PYTHON) plot_moments_scatter_3d.py $(DATA_DIR)/prt.000001000.h5

clean:
	@echo "==========================================================="
	@echo "Cleaning generated plots"
	@echo "==========================================================="
	rm -rf anisotropy_plots/ mirror_plots/ field_images/ spectral_plots/ prt_plots/ fancy_vdf_plots/ phase_space_plots/

help:
	@echo "Available Commands:"
	@echo "  make brazil   - Runs anisotropy_analysis.py to generate Brazil plots"
	@echo "  make mirror   - Runs mirror_physics.py to visualize magnetic holes"
	@echo "  make fields   - Runs fluctuationofmagneticfiel.py to visualize fluctuations"
	@echo "  make spectral - Runs spectral_analysis.py to generate PSD plots"
	@echo "  make all      - Runs all analysis scripts"
	@echo "  make clean    - Removes the generated plot directories"
