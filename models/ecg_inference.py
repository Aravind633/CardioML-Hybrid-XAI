# import numpy as np
# import tensorflow as tf

# # Load trained ECG model
# ecg_model = tf.keras.models.load_model("models/ecg_lstm_model.h5")

# def get_ecg_risk_score(ecg_signal):
#     """
#     ecg_signal: numpy array of shape (187,)
#     returns: scalar risk score in [0,1]
#     """

#     # Reshape for model
#     ecg_signal = ecg_signal.reshape(1, 187, 1)

#     probs = ecg_model.predict(ecg_signal, verbose=0)[0]

#     # Probability of normal rhythm (class 0)
#     p_normal = probs[0]

#     # ECG-based risk score
#     risk_score = 1.0 - p_normal

#     return float(risk_score)


import numpy as np
import tensorflow as tf

# Load ECG model once
_ecg_model = tf.keras.models.load_model("models/ecg_lstm_model.h5")

def get_ecg_risk_score(ecg_signal):
    """
    ecg_signal: numpy array of shape (187,)
    returns: float in [0, 1]
    """
    ecg_signal = np.array(ecg_signal).reshape(1, 187, 1)

    probs = _ecg_model.predict(ecg_signal, verbose=0)[0]

    # Class 0 = Normal
    p_normal = probs[0]
    risk_score = 1.0 - p_normal

    return float(risk_score)
