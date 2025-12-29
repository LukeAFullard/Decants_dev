import matplotlib.pyplot as plt
import pandas as pd
from decants.objects import DecantResult

def plot_adjustment(result: DecantResult):
    """
    Unified plotting logic for DecantResult.

    Layout:
    - Top Panel: Overlay of Original, Adjusted, and Trend (if available or inferred).
    - Middle Panel: The Isolated Covariate Effect.
    - Bottom Panel: Residuals of Adjusted Series (or Adjusted Series itself if it represents residuals + trend).
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

    # Extract data
    orig = result.original_series
    adj = result.adjusted_series
    eff = result.covariate_effect

    # Top Panel
    axes[0].plot(orig, label='Original', color='black', alpha=0.5)
    axes[0].plot(adj, label='Adjusted (Decanted)', color='blue', linewidth=1.5)
    axes[0].set_title("Original vs Adjusted Series")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Middle Panel
    axes[1].plot(eff, label='Covariate Effect', color='green')
    axes[1].set_title("Isolated Covariate Effect")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Bottom Panel
    # If Adjusted = Trend + Noise, we might want to see if it looks cleaner?
    # Or plot the difference between Original and Adjusted which IS the effect (redundant?)
    # Let's plot the Adjusted series again but maybe checking for stationarity or just detail?
    # The plan says "Residuals of Adjusted Series".
    # Since we don't know the Trend explicitly in the Result (unless we parse the model),
    # let's just plot the Adjusted Series.
    # Optionally, if the model isolated a "Trend" we could plot that.
    # For GAM, term 0 is trend. We could extract it?
    # For now, just plot Adjusted.
    axes[2].plot(adj, label='Adjusted Series', color='red', alpha=0.7)
    axes[2].set_title("Adjusted Series (Outcome)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
