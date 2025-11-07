import P56_VAD as VAD
import SRVT_utils as SRVT
import SNR_test as SNR
from tqdm import tqdm
import pandas as pd #CSV, Dataframe
import os

# read "participant_metadata.csv"
path = "participant_metadata.csv"
metadata = pd.read_csv(path, sep=";")

# the main folder where all the audio files are actually stored
audiofile_path = "/Users/muni/Desktop/BA_PYTHON/recordings"

####### --- Function Definitions --- #######
## 1. Function to create the full file path for each audio file ##
def path_join(row):
    """
        It takes one row from the metadata and combines the main audio path, the participant's folder name,
        and the audio file's name to make a full, complete path.

        Example:
        audiofile_path (global) = "/Users/muni/Desktop/BA_PYTHON/recordings"
        row.Participant_ID = "P01"
        row.File_Name = "P01_condition_A_mic.wav"
    """

    participant_folder = row.Participant_ID
    return os.path.join(audiofile_path, participant_folder, row.File_Name)


## 2. Function to analyze a single audio file ##
def analyze_single_row(row):
    """
        This function does the main work for just ONE audio file.
        It takes a single row (from the metadata) as input.
        1. It finds the full path to the audio file using my path_join function.
        2. It runs the analysis functions from the VAD and SRVT scripts.
        3. It returns all the results in a dictionary (a key-value list).
    """

    filepath = path_join(row)

    asldB, activityFactor, duration = VAD.P56_fileAnalysis(filepath)
    loudness = SRVT.fileLoudness(filepath)
    band = SRVT.fileFreqBandRelative(filepath)
    snr = SNR.cal_snr(filepath)

    return {
        "File_Name": row.File_Name,
        "Condition": row.Condition,
        "Device": row.Device,
        "ASL": asldB,
        "Percentage of Speech": activityFactor,
        "Duration": duration,
        "Loudness": loudness,
        "Bandwidth": band,
        "SNR": snr,
    }


## 3. 분석 결과 내는 함수 ##
def analyze_result():
    """
    ## 전체 참가자 녹음 파일 분석 ##

    results = []
    rows = list(metadata.itertuples(index=False))

    # for문으로 순차 실행
    for row in tqdm(rows, total=len(rows)):
        res = analyze_single_row(row)
        if not res == None:
            results.append(res)
    """

    ## each participant 녹음 파일 분석 ##
    participant_to_run = "P17"
    sub = metadata[metadata["Participant_ID"] == participant_to_run].copy()
    results = []

    for row in tqdm(sub.itertuples(index=False), total=len(sub)):
        res = analyze_single_row(row)
        if res is not None:
            results.append(res)


    df = pd.DataFrame(results)
    df.to_csv("analysis_results.csv", index=False)
    print("saved to analysis_results.csv")
    return df

if __name__ == "__main__":
    analyze_result()