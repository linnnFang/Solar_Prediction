"""Naive persistence: predict the last observed power, y_hat(t+h) = P_t."""

import numpy as np


class NaivePersistence:
    name = "naive_persistence"

    def predict_kw(self, frame):
        """Input: pv_window_frame dict. Output: y_pred in kW (== power at t)."""
        return np.asarray(frame["p_now_kw"], dtype=np.float32).copy()
