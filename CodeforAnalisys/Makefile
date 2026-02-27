# Makefile to run Analysis Code for Anisotropy and Mirror Instability

# Variables
PYTHON = python3
B0_REF = 0.1 # This aligns with vA_over_c = 0.1 in psc_temp_aniso.cxx

.PHONY: all brazil mirror fields clean

all: brazil mirror fields

brazil:
	@echo "==========================================================="
	@echo "Generating Brazil Plot for Anisotropy (beta_par vs T_perp/T_par)"
	@echo "==========================================================="
	$(PYTHON) anisotropy_analysis.py --B0 $(B0_REF)

mirror:
	@echo "==========================================================="
	@echo "Generating Mirror Instability plots (Magnetic Holes and Current)"
	@echo "==========================================================="
	$(PYTHON) mirror_physics.py

fields:
	@echo "==========================================================="
	@echo "Visualizing Magnetic Field Fluctuations"
	@echo "==========================================================="
	$(PYTHON) fluctuationofmagneticfiel.py --B0 $(B0_REF) --plane yz --smooth 1.5

clean:
	@echo "==========================================================="
	@echo "Cleaning generated plots"
	@echo "==========================================================="
	rm -rf anisotropy_plots/ mirror_plots/ field_images/

help:
	@echo "Available Commands:"
	@echo "  make brazil   - Runs anisotropy_analysis.py to generate Brazil plots"
	@echo "  make mirror   - Runs mirror_physics.py to visualize magnetic holes"
	@echo "  make fields   - Runs fluctuationofmagneticfiel.py to visualize fluctuations"
	@echo "  make all      - Runs all the three analysis scripts"
	@echo "  make clean    - Removes the generated plot directories"
