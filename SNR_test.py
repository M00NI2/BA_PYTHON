import P56_VAD as VAD
import librosa
import numpy as np

def cal_snr(path):
    eps = np.finfo(float).eps  # 0으로 나누는 것을 방지하기 위한 아주 작은 값

    x, fs = librosa.load(path, sr=None)

    asl_msq, actfact, c0 = VAD.asl_P56(x, fs, 16)

    if actfact == 0.0:
        # 음성이 전혀 감지되지 않음. SNR은 0 또는 매우 낮음.
        return 0.0
    if actfact >= 1.0:
        # 100% 음성으로 감지됨 (노이즈 추정 불가). 매우 높은 값 반환
        return 99.0 # 사실상 무한대(Inf)에 가까움

    # 4. 신호 RMS (RMS_signal) 계산
    # RMS = sqrt(Mean Square)
    rms_signal = np.sqrt(asl_msq)

    # 5. 노이즈 RMS (RMS_noise) 계산
    n_total = len(x) # 전체 샘플 수
    total_energy = np.sum(x**2) # 파일의 총 에너지

    # 음성 부분의 총 에너지 추정
    # (음성 에너지 = 음성의 평균제곱 * 음성 샘플 수)
    speech_energy = asl_msq * n_total * actfact

    # 소음 부분의 총 에너지 추정
    noise_energy = total_energy - speech_energy

    # VAD 추정 오류로 소음 에너지가 0 이하일 경우, 아주 작은 양수로 처리
    if noise_energy <= 0:
        noise_energy = eps

    # 소음 부분의 샘플 수
    n_noise = n_total * (1 - actfact)

    # 소음의 평균 제곱(MS)
    ms_noise = noise_energy / n_noise
    rms_noise = np.sqrt(ms_noise) # 소음의 RMS

    # 6. 최종 SNR (dB) 계산
    # SNR(dB) = 20 * log10(RMS_signal / RMS_noise)
    # rms_noise가 0에 가까울 수 있으므로 eps 추가
    snr_db = 20 * np.log10(rms_signal / (rms_noise + eps))

    # 비정상적으로 높은 값이 나오는 것을 방지 (예: 99dB 이상)
    if snr_db > 99.0:
        snr_db = 99.0

    return snr_db