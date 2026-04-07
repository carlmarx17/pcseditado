#!/usr/bin/env python3
"""
generate_pdf_report.py
======================
Generate a validation report (PDF) for the PSC PIC simulation.

Includes:
  1. Initial conditions and statistical validation methodology.
  2. Goodness-of-fit results (K-S and A-D tests).
  3. Kappa vs Maxwellian comparison plots.
  4. 3D VDF visualisation and phase-space scatter plots.
  5. Magnetic-fluctuation evidence of mirror-mode structures.
"""

import os

try:
    from fpdf import FPDF
except ImportError:
    print("ERROR: fpdf2 is not installed. Run:  pip install fpdf2")
    exit(1)

# ── Image paths ──────────────────────────────────────────────────────────────
BASE = "/home/carlmarxt/Documents/pcseditado"

IMG_GOF = f"{BASE}/build/src/prt_plots/goodness_of_fit.png"
IMG_KAPPA = f"{BASE}/build/src/prt_plots/kappa_vs_maxwellian.png"
IMG_VDF_3D = f"{BASE}/build/src/fancy_vdf_plots/iones_vdf_3d.png"
IMG_FLUCT = f"{BASE}/CodeforAnalisys/field_images/Bmag_fluct_step11600_yz.png"
IMG_SCATTER_IONS = f"{BASE}/build/src/phase_space_plots/iones_3d_scatter.png"
IMG_SCATTER_ELEC = f"{BASE}/build/src/phase_space_plots/electrones_3d_scatter.png"
IMG_VALIDATION = f"{BASE}/build/src/validation_plots/validation_summary.png"

OUTPUT_PDF = f"{BASE}/CodeforAnalisys/validation_report.pdf"


class ReportPDF(FPDF):
    """Custom PDF class with consistent header/footer."""

    def header(self):
        self.set_font("helvetica", "B", 15)
        self.cell(
            0, 10,
            "Validation Report: Initial Conditions and Dynamics",
            border=False, align="C",
        )
        self.ln(10)
        self.set_font("helvetica", "B", 12)
        self.cell(
            0, 10,
            "PIC-PSC Simulation: Kappa Distribution and Mirror Instability",
            border=False, align="C",
        )
        self.ln(15)

    def footer(self):
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def _add_image(pdf: FPDF, path: str, width: int = 170):
    """Insert an image if the file exists; otherwise print a warning."""
    if os.path.exists(path):
        pdf.image(path, w=width)
        pdf.ln(5)
    else:
        pdf.set_font("helvetica", "I", 9)
        pdf.cell(0, 6, f"[Image not found: {os.path.basename(path)}]", ln=1)
        pdf.ln(3)


def create_report():
    """Build the full validation report PDF."""
    pdf = ReportPDF()

    # ── PAGE 1: Introduction and methodology ─────────────────────────────
    pdf.add_page()
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, "1. Initial Physical Conditions and Validation Methodology", ln=1)

    pdf.set_font("helvetica", "", 12)
    pdf.set_x(10)
    pdf.multi_cell(0, 6, (
        "The simulation was initialised to excite the mirror instability with:\n\n"
        " - Ion parallel beta: 10.0\n"
        " - Anisotropy T_perp / T_par: 3.5\n"
        " - Kappa index: 3.0\n"
        " - Mass ratio m_i / m_e: 64"
    ))
    pdf.ln(5)

    pdf.set_font("helvetica", "B", 11)
    pdf.cell(0, 8, "Statistical Validation Methods:", ln=1)
    pdf.set_font("helvetica", "", 11)
    pdf.set_x(10)
    pdf.multi_cell(0, 6, (
        "To certify the rigour of the energy-distribution injection, "
        "the code performs detailed non-parametric statistical tests on "
        "the macro-particles generated at the initial condition (t=0):\n\n"
        "A) Kolmogorov-Smirnov (K-S) test: Measures the maximum deviation (D) "
        "between the empirical cumulative distribution function (from the code) "
        "and the theoretical one. It strongly verifies the accuracy of the "
        "probability density at low and medium energies.\n\n"
        "B) Anderson-Darling (A-D) test: Uses a weighted integration that assigns "
        "much greater weight to the distribution tails. Since the Kappa distribution "
        "is intrinsically characterised by its power-law over-population at high "
        "energies, this test determines whether the tail has been under- or "
        "over-sampled compared to a purely Maxwellian distribution."
    ))
    pdf.ln(5)

    _add_image(pdf, IMG_GOF)

    pdf.set_font("helvetica", "I", 10)
    pdf.multi_cell(0, 5, (
        "K-S / A-D conclusion: The Maxwellian hypothesis is rejected with "
        "virtual certainty (p-value ~ 0) both in the bulk and the tails. "
        "The Kappa hypothesis (kappa = 3.0) is fully compatible and is NOT "
        "rejected, validating the initial-condition loading."
    ))

    # ── PAGE 2: Tail analysis and validation summary ─────────────────────
    pdf.add_page()
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, "2. High-Energy Tail Comparative Analysis", ln=1)
    pdf.set_font("helvetica", "", 11)
    pdf.multi_cell(0, 6, (
        "The panels below show, in multiple logarithmic scales, "
        "the systematic deviation caused by the Kappa injection compared "
        "to a linear fit (which would correspond to a pure Gaussian/Maxwellian)."
    ))

    _add_image(pdf, IMG_KAPPA, width=180)

    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, "3. Additional Validation Summary", ln=1)
    _add_image(pdf, IMG_VALIDATION, width=160)

    # ── PAGE 3: VDF and phase space ──────────────────────────────────────
    pdf.add_page()
    pdf.set_x(10)
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, "4. Volumetric VDF and Phase-Space Scatter Plots", ln=1)
    pdf.set_font("helvetica", "", 11)
    pdf.set_x(10)
    pdf.multi_cell(0, 6, (
        "For deeper visual validation, the 3D volumetric distribution function "
        "and 3D momentum-space scatter plot are presented below."
    ))

    pdf.set_x(10)
    pdf.set_font("helvetica", "B", 11)
    pdf.multi_cell(0, 6, "3D Volumetric VDF (Ions):")
    _add_image(pdf, IMG_VDF_3D, width=160)

    pdf.set_x(10)
    pdf.set_font("helvetica", "B", 11)
    pdf.multi_cell(0, 6, "3D Phase-Space Scatter Plot (Ions):")
    _add_image(pdf, IMG_SCATTER_IONS, width=160)

    # ── PAGE 4: Electron scatter and magnetic pockets ────────────────────
    pdf.add_page()
    pdf.set_x(10)
    pdf.set_font("helvetica", "B", 11)
    pdf.multi_cell(0, 6, "3D Phase-Space Scatter Plot (Electrons):")
    _add_image(pdf, IMG_SCATTER_ELEC, width=160)

    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, "5. Magnetic Pockets / Mirror-Mode Structures", ln=1)
    pdf.set_font("helvetica", "", 11)
    pdf.multi_cell(0, 6, (
        "Confirming the simulation dynamics past step 11600: "
        "the oscillations and depressions visible in the B_mag fluctuation "
        "heat-map consolidate into magnetic pockets (macro-structures) "
        "characteristic of the mirror-unstable regime."
    ))
    pdf.ln(5)

    _add_image(pdf, IMG_FLUCT, width=180)

    pdf.output(OUTPUT_PDF)
    print(f"PDF report generated: {OUTPUT_PDF}")


if __name__ == "__main__":
    create_report()
