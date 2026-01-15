# -*- coding: utf-8 -*-
"""
Minimal R–T Curve (No Text, No Labels)
Author: Huahua
Description:
    Draws a clean R–T curve without any text, title, or legend.
"""

import numpy as np
import matplotlib.pyplot as plt

# ==============================
# 1. Data
# ==============================
T = np.array([30, 35, 40, 45, 50, 55, 60, 65])  # Temperature (°C)
Rt = np.array([57.88, 59.39, 61.58, 63.24, 64.71, 66.06, 66.96, 66.99])  # Measured
R_theory = np.array([55.80, 56.89, 58.00, 59.13, 60.24, 61.35, 62.46, 63.57])  # Theoretical

# ==============================
# 2. Plot (no text, no labels)
# ==============================
plt.figure(figsize=(7, 4.5))
plt.plot(T, Rt, 'r-', linewidth=1.8)
plt.plot(T, R_theory, 'b--', linewidth=1.5)
plt.grid(True, linestyle='--', alpha=0.6)

# Remove all text decorations
plt.title("")
plt.xlabel("")
plt.ylabel("")
plt.legend().remove() if plt.gca().get_legend() else None

# ==============================
# 3. Save and Show
# ==============================
plt.tight_layout()
plt.savefig("RT_curve_minimal.png", dpi=300, bbox_inches='tight')
plt.show()

print("✅ Figure saved as: RT_curve_minimal.png")
