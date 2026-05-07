import os

import numpy as np

from .utils import plot_moment_comparison, resolve_history_path, save_metrics, select_device

REFERENCE_VALUES = np.array(
    [
        20231.2507065272, -2947.12234630892, 38187.7425633891, 117802.633917250, 2667.95568642847,
        111638.853710749, 175532.834956306, 12600.2512004678, 93015.6636397448, 203623.336920169,
        -6900.53640323294, 108408.091894997, 189957.209629249, -4950.82140443995, 103761.017273395,
        183795.400634641, 1581.94003720943, 122733.758574282, 159353.142770030, -6663.90711810087,
        111460.883330548, 175405.486039242, -13848.6007158883, 125693.773876472, 189926.248421888,
        -11523.9822060260, 127185.193121587, 145970.641567344, 410.994686038808, 77203.2307401270,
        86869.1020809404, -17402.2213290882, 70367.5172760106, 39273.2399627759, 4525.60488665819,
        97256.6189041015, 17090.6561322898, -6826.06421577401, 54691.8413364768, 2182.92029630350,
        -5987.49620710858, 19089.6325292373, 7128.68804620302, 1040.15420438779, 10285.8550869701,
        271081.763364894, 38809.7981653962, -779255.364779940,
    ],
    dtype=np.float64,
)


def _relative_error(pred, truth, indices):
    denom = np.maximum(np.abs(truth[indices]), 1e-12)
    return float(np.mean(np.abs(pred[indices] - truth[indices]) / denom))


def run(args):
    device = select_device(args.device)
    print(f"Using device: {device}")

    history_path = resolve_history_path(args)
    if not history_path:
        raise FileNotFoundError(
            "History .npy not found. Please run train first, or pass --history_path explicitly."
        )

    history = np.load(history_path)
    pred = history[-1, -1, :] / 10**9

    truth = REFERENCE_VALUES

    idx_mx = np.arange(0, 48, 3)
    idx_my = np.arange(1, 48, 3)
    idx_mz = np.arange(2, 48, 3)

    metrics = {
        "history_path": history_path,
        "relative_error_mx": _relative_error(pred, truth, idx_mx),
        "relative_error_my": _relative_error(pred, truth, idx_my),
        "relative_error_mz": _relative_error(pred, truth, idx_mz),
    }

    plot_path = os.path.join(args.output_dir, "img.png")
    metrics_path = os.path.join(args.output_dir, "metrics.txt")
    plot_moment_comparison(pred, truth, plot_path)
    save_metrics(metrics, metrics_path)

    print(f"relative_error_mx: {metrics['relative_error_mx']:.6f}")
    print(f"relative_error_my: {metrics['relative_error_my']:.6f}")
    print(f"relative_error_mz: {metrics['relative_error_mz']:.6f}")
    print(f"Saved plot to {plot_path}")
    print(f"Saved metrics to {metrics_path}")
