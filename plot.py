import matplotlib.pyplot as plt


def plot_series(series, ax, color="rx"):
    ax.plot([i for i, _ in enumerate(series)], series)
    return ax
