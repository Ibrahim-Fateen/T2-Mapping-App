from scipy.optimize import curve_fit
import numpy as np

def calculate_t2_map(slice_data, te_sec, bounds=((0, 0), (np.inf, 3)), even_numbered_echos=True):
    if even_numbered_echos:
        te_sec = te_sec[::2]

    def exp_model(te, S0, T2):
        return S0 * np.exp(-te/T2)

    mnr_signals = np.zeros((slice_data.shape[0], slice_data.shape[1], len(te_sec)))
    T2_map = np.zeros(slice_data.shape[:2])
    S0_map = np.zeros(slice_data.shape[:2])

    for i in range(slice_data.shape[0]):
        for j in range(slice_data.shape[1]):
            if even_numbered_echos:
                mnr_signal = slice_data[i, j, ::2]
            else:
                mnr_signal = slice_data[i, j, :]
            try:
                initial_guess = (mnr_signal.max(), 0.1)
                popt, _ = curve_fit(exp_model, te_sec, mnr_signal, p0=initial_guess, bounds=bounds)
            except:
                print(f"Fit failed for ({i}, {j})")
                popt = (0, 0)
            mnr_signals[i, j, :] = mnr_signal
            S0, T2 = popt

            T2_map[i, j] = T2
            S0_map[i, j] = S0

    return T2_map, S0_map

