"""
    Parts of this code were developed with the assistance of OpenAI's GPT model (ChatGPT, GPT-5).
    The author verified and adapted all generated code to ensure correctness and suitability for this study.

"""

import P56_VAD as VAD
import librosa
import numpy as np

def cal_snr(path):
    eps = np.finfo(float).eps

    x, fs = librosa.load(path, sr=None)

    asl_msq, actfact, c0 = VAD.asl_P56(x, fs, 16)

    if actfact == 0.0:

        return 0.0
    if actfact >= 1.0:

        return 99.0

    # RMS = sqrt(Mean Square)
    rms_signal = np.sqrt(asl_msq)


    n_total = len(x)
    total_energy = np.sum(x**2)

    speech_energy = asl_msq * n_total * actfact


    noise_energy = total_energy - speech_energy


    if noise_energy <= 0:
        noise_energy = eps


    n_noise = n_total * (1 - actfact)


    ms_noise = noise_energy / n_noise
    rms_noise = np.sqrt(ms_noise) # 소음의 RMS

    # SNR(dB) = 20 * log10(RMS_signal / RMS_noise)
    snr_db = 20 * np.log10(rms_signal / (rms_noise + eps))

    if snr_db > 99.0:
        snr_db = 99.0

    return snr_db