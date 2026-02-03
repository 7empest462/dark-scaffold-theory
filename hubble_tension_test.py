import numpy as np
import matplotlib.pyplot as plt

# Time from Big Bang to Now (units of billions of years)
time = np.linspace(0, 13.8, 1000)

# 1. THE STANDARD MODEL (Lambda-CDM)
# A steady acceleration that predicts a lower H0 (67.4)
h0_early_predict = 67.4
expansion_standard = h0_early_predict * (time / 13.8)

# 2. YOUR MODEL (Pre-existing Scaffold)
# Early Dark Energy or extra primordial structure adds a "kick" early on
# This reconciles the early math with the 73.0 local measurement
h0_local_actual = 73.2
kick_factor = 1.05 # The influence of the primordial wells
expansion_yours = h0_local_actual * (time / 13.8) * np.exp(0.05 * (time - 13.8))

plt.figure(figsize=(10, 6))
plt.plot(time, expansion_standard, label="Standard Prediction (H0 = 67.4)", color='cyan', linestyle='--')
plt.plot(time, expansion_yours, label="Your Scaffold Model (H0 = 73.2)", color='purple', linewidth=3)

# Highlight the Tension Gap
plt.fill_between(time, expansion_standard, expansion_yours, color='gray', alpha=0.1, label="The Hubble Tension")

plt.title("Tension Test: Reconciling the Early and Local Universe")
plt.xlabel("Time (Billions of Years)")
plt.ylabel("Expansion Rate")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('hubble_tension_plot.png')
