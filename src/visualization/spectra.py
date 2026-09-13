# Plot real-vs-fake radial spectra
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..evaluation.spectra import radial_spectrum


def plot_spectra(real_images, fake_images, out_path="outputs/spectra/spectra.png"):
    pr = radial_spectrum(real_images)
    pf = radial_spectrum(fake_images)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(pr / pr.sum(), label="real")
    ax.semilogy(pf / pf.sum(), label="generated")
    ax.set_xlabel("radial frequency bin")
    ax.set_ylabel("power (normalized)")
    ax.set_title("Radial spectra: real vs generated")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
