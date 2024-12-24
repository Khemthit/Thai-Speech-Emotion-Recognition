import os
import time
import json
import zipfile
import requests
import numpy as np
import pandas as pd
from glob import glob
from urllib.error import HTTPError
from typing import Dict, List, Union

# Use absolute import based on your project structure
from module.utils.logger import get_logger

# Emotion mappings
emo2idx = {emo: i for i, emo in enumerate(['Neutral', 'Angry', 'Happy', 'Sad', 'Frustrated'])}
idx2emo = {v: k for k, v in emo2idx.items()}
correctemo = {
    'Neutral': 'Neutral',
    'Angry': 'Anger',
    'Happy': 'Happiness',
    'Sad': 'Sadness',
    'Frustrated': 'Frustration'
}


class InitialData:
    """
    The `InitialData` class is designed to manage the preparation and configuration of a dataset for Speech Emotion Recognition (SER) tasks. It encapsulates functionality for downloading, organizing, and preparing data, as well as generating labels for emotions. Below is a detailed explanation of the class and its components:

    Attributes
    -------
    + test_fold: int
        fold number of dataset used as test set. This dataset is separated
        into 10 folds. See self.fold_config() for more info.
    + agreement_threshold: float
        agreement threshold of the dataset. The dataset consists of multiple
        annotators. Thus, some data might have different annotation consensus
        among multiple annotators. `agreement_threshold` param is a threshold
        that will filter out label that multiple annotators did not agree upon
        selected ratio (default as 0.71). The default value was calculated from
        inter-rater score. We recommend not to change this value.
    + mic_type: str
        microphone type used for training dataset. Two choices avilable: `con` and `clip`,
        where `con` denotes condensor mic and `clip` denotes clip mic
    + download_dir: str
        dataset download directory
    + include_zoom: bool
        specify whether to include zoom experiment in dataset fold or not
    + emotions: List[str]
        list of allowed emotion. There are 5 emotions available
        [neutral, anger, happiness, sadness, frustration] but default
        as four emotions: neutral, anger, happiness, sadness

    Methods
    -------
    + extract()
        Run once as a preparation: download dataset, generate csv labels

    Usage
    -------
    + Initialize the class with the desired configuration:
        ```python
        data = InitialData(test_fold=0, emotions=["neutral", "happiness", "anger"])
        ```
    + Call the `extract` method to download, extract, and prepare the dataset:
        ```python
        data.extract()
        ```
    """

    def __init__(
            self,
            test_fold: int,
            agreement_threshold: float = 0.71,
            mic_type: str = 'clip',
            download_dir: str = "./dataset",
            include_zoom: bool = True,
            emotions: List[str] = ["neutral", "anger", "happiness", "sadness", "frustration"],
            *args,
            **kwargs):
        
        # Configure logging
        self.logger = get_logger("Data Wrangling")

        # Loading dataset config
        self.agreement_threshold = agreement_threshold
        self.mic_type = mic_type
        self.test_fold = test_fold

        # Config n_classes, available emotions
        self.include_zoom = include_zoom
        self.emotions = emotions
        self.n_classes = len(self.emotions)

        # Config download dir
        self.download_dir = download_dir
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        # Define download URL
        version = 1
        release_url = f"https://github.com/vistec-AI/dataset-releases/releases/download/v{version}"
        self.github_url = {
            "studio1-10": f"{release_url}/studio1-10.zip",
            "studio11-20": f"{release_url}/studio11-20.zip",
            "studio21-30": f"{release_url}/studio21-30.zip",
            "studio31-40": f"{release_url}/studio31-40.zip",
            "studio41-50": f"{release_url}/studio41-50.zip",
            "studio51-60": f"{release_url}/studio51-60.zip",
            "studio61-70": f"{release_url}/studio61-70.zip",
            "studio71-80": f"{release_url}/studio71-80.zip",
            "zoom1-10": f"{release_url}/zoom1-10.zip",
            "zoom11-20": f"{release_url}/zoom11-20.zip",
        }
        self.labels_url = f"{release_url}/emotion_label.json"

        # Define fold split
        self.fold_config = {
            0: [f"studio{s:03d}" for s in range(1, 11)],
            1: [f"studio{s:03d}" for s in range(11, 21)],
            2: [f"studio{s:03d}" for s in range(21, 31)],
            3: [f"studio{s:03d}" for s in range(31, 41)],
            4: [f"studio{s:03d}" for s in range(41, 51)],
            5: [f"studio{s:03d}" for s in range(51, 61)],
            6: [f"studio{s:03d}" for s in range(61, 71)],
            7: [f"studio{s:03d}" for s in range(71, 81)],
            8: [f"zoom{s:03d}" for s in range(1, 11)],
            9: [f"zoom{s:03d}" for s in range(11, 21)]
        }
        assert self.test_fold in self.fold_config.keys(), "Invalid test_fold number."
        self.studio_list = [s for studios in self.fold_config.values() for s in studios]

    def extract(self):
        """Run once as a preparation: download dataset, generate csv labels"""
        self._download()
        self._prepare_labels()

    def _get_audio_path(self, audio_name: str) -> Union[str, None]:
        if not isinstance(audio_name, str):
            raise TypeError(f"audio name must be string but got {type(audio_name)}")
        studio_type = audio_name[0]
        studio_num = audio_name.split('_')[0][1:]
        if studio_type == "s":
            directory = f"studio{studio_num}"
        elif studio_type == "z":
            directory = f"zoom{studio_num}"
        else:
            raise NameError(f"Error reading file name {audio_name}")
        audio_path = os.path.join(self.download_dir, directory, "con", f"{audio_name}".replace(".wav", ".flac"))
        if studio_type == "s":
            audio_path = audio_path.replace("con", self.mic_type)
        elif studio_type == "z":
            audio_path = audio_path.replace("con", "mic")
        else:
            raise NameError(f"Error reading file name {audio_name}")
        if not os.path.exists(audio_path):
            self.logger.warning(f"{audio_path} not found, skipping...")
            return None
        return audio_path 
    
    def find_fold_key(self, file_name: str) -> Union[int, None]:
        prefix = ''.join(filter(str.isalpha, file_name))
        number = int(''.join(filter(str.isdigit, file_name)))
        if prefix == "s":
            key = (number - 1) // 10
        elif prefix == "z":
            key = 8 + (number - 1) // 10
        else:
            key = None
        return key

    def _prepare_labels(self):
        # Check if labels.csv exists
        labels_csv_path = os.path.join(self.download_dir, "labels.csv")
        if not os.path.exists(labels_csv_path):
            self.logger.info("Preparing labels...")
            json_path = os.path.join(self.download_dir, "labels.json")
            if not os.path.exists(json_path):
                raise FileNotFoundError(f"labels.json not found at {self.download_dir}")

            self.logger.info(f"Formatting {json_path} ...")
            data = read_json(json_path)
            # Filter studio that doesn't appear in download_dir
            available_studios = []
            for std in sorted(glob(f"{self.download_dir}/*/")):
                std = std[:-1].split("/")[-1]
                std = std[0] + std[-3:]
                available_studios.append(std)
            data = {k: v for k, v in data.items() if k.split("_")[0].lower() in available_studios}
            agreements = get_agreements(data)
            labels = pd.DataFrame([
                (self.find_fold_key(file_name=k.split("_")[0]), self._get_audio_path(k), correctemo[idx2emo[v]])
                for k, v in {
                    k: convert_to_hardlabel(v, thresh=self.agreement_threshold)
                    for k, v in agreements.items()
                }.items()
                if v != -1 and self._get_audio_path(k) is not None
            ], columns=['FoldID', 'Path', 'Emotion'])

            labels.to_csv(labels_csv_path, index=False)
            self.logger.info(f"Labels saved to {labels_csv_path}.")
        else:
            self.logger.info(f"Labels already exist")
            labels = pd.read_csv(labels_csv_path)

    def download_file(self, url: str, output_path: str):
        """
        Download a file from a given URL and save it to the specified output path.

        Parameters
        ----------
        url : str
            URL of the file to download.
        output_path : str
            Path to save the downloaded file.
        """
        with requests.get(url, stream=True) as response:
            try:
                response.raise_for_status()
            except HTTPError as e:
                self.logger.error(f"HTTP Error: {e}")
                raise

            # Get total file size from the Content-Length header
            total_size = int(response.headers.get('Content-Length', 0))
            downloaded_size = 0
            chunk_size = 1024 * 1024  # 1 MB per chunk
            last_logged_time = time.time()  # Record the start time

            self.logger.info(f"Starting download: {url}")
            self.logger.info(f"Total file size: {total_size / (1024 * 1024):.2f} MB")

            # Write the content to the output file
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:  # Filter out keep-alive new chunks
                        f.write(chunk)
                        downloaded_size += len(chunk)

                        # Calculate progress
                        percent_done = (downloaded_size / total_size) * 100 if total_size else 0

                        # Log every 30 seconds
                        current_time = time.time()
                        if current_time - last_logged_time >= 30:
                            last_logged_time = current_time
                            self.logger.info(
                                f"Downloading {os.path.basename(output_path)}: "
                                f"{percent_done:.0f}% ({downloaded_size / (1024 * 1024):.2f} MB / {total_size / (1024 * 1024):.2f} MB)"
                            )

            self.logger.info(f"Download complete: {output_path}")

    def _download(self):
        # Check for missing files
        missing_files = [
            studio for studio, download_url in self.github_url.items()
            if not os.path.exists(os.path.join(self.download_dir, f"{studio}.zip"))
        ]
        # Download missing files
        if missing_files:
            self.logger.info("Downloading missing dataset files...")
            for f in missing_files:
                self.logger.info(f"Downloading {f}.zip ...")
                out_name = os.path.join(self.download_dir, f"{f}.zip")
                self.download_file(url=self.github_url[f], output_path=out_name)
        else:
            self.logger.info("All dataset ZIP files are already downloaded.")

        # Download labels.json if not present
        labels_json_path = os.path.join(self.download_dir, "labels.json")
        if not os.path.exists(labels_json_path):
            self.logger.info("Downloading labels.json ...")
            try:
                self.download_file(url=self.labels_url, output_path=labels_json_path)
            except HTTPError:
                self.logger.error(f"404 Error: Cannot download {self.labels_url}")
                raise
        else:
            self.logger.info("labels.json already exists.")
        
        def generate_names(name_prefix, range_start, range_end):
            return [f"{name_prefix}{str(i).zfill(3)}" for i in range(range_start, range_end + 1)]
        
        # Extract ZIP files if directories do not exist
        studios_present = [os.path.basename(std.rstrip('/')) for std in glob(os.path.join(self.download_dir, "*")) if os.path.isdir(std)]
        studios_to_extract = [
            f for f in self.github_url.keys() if any(
                generated_name not in studios_present 
                    for generated_name in generate_names(
                        ''.join(filter(str.isalpha, f)),
                        int(''.join(filter(str.isdigit, f.split('-')[0]))),
                        int(f.split('-')[1])
                    )
            )
        ]

        if studios_to_extract:
            self.logger.info("Extracting files...")
            for f in sorted(glob(os.path.join(self.download_dir, "*.zip"))):
                studio_key = os.path.basename(f).split(".")[0]
                if studio_key not in studios_present:
                    self.logger.info(f"Unzipping {f} ...")
                    with zipfile.ZipFile(f, 'r') as zip_ref:
                        zip_ref.extractall(self.download_dir)
            self.logger.info("All files have been extracted.")
        else:
            self.logger.info("All files are already extracted.")

        self.logger.info("All files are downloaded and extracted.")

# Utility Functions
def read_json(json_path: str) -> Dict[str, dict]:
    """Read label JSON"""
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        # Ensure all values are lists with at least one element
        assert all(isinstance(v, list) and len(v) > 0 for v in data.values()), 'All values in JSON must be non-empty lists.'
        return {k: v[0] for k, v in data.items()}
    else:
        raise FileNotFoundError(f"File not found: {json_path}")

def convert_to_softlabel(evals: List[str]) -> List[float]:
    """Converts a list of emotions into a distribution"""
    softlabel = [0 for _ in range(len(emo2idx.keys()))]
    for emo in evals:
        emo_capitalized = emo.capitalize()
        if emo_capitalized in emo2idx:
            softlabel[emo2idx[emo_capitalized]] += 1
    total = sum(softlabel)
    if total != 0:
        softlabel = [count / total for count in softlabel]
    return softlabel

def get_score_from_emo_list(emo_list: List[List[str]]) -> List[float]:
    """Aggregate a list of evaluations (which are lists of emotions) into a distribution"""
    softlabels = [convert_to_softlabel(evals) for evals in emo_list]
    if not softlabels:
        return [0.0] * len(emo2idx)
    aggregated = np.mean(softlabels, axis=0).tolist()
    return aggregated

def get_agreements(data: Dict[str, dict]) -> Dict[str, List[float]]:
    """Get agreement distribution from provided labels"""
    softlabel = {k: get_score_from_emo_list(v['annotated']) for k, v in data.items()}
    return softlabel

def convert_to_hardlabel(agreement_dist: List[float], thresh: float = 0.7) -> Union[int, int]:
    """
    Convert a distribution of agreements to a hard label.

    Parameters
    ----------
    agreement_dist : List[float]
        The distribution of agreements across emotions.
    thresh : float, optional
        The threshold above which an emotion is considered, by default 0.7.

    Returns
    -------
    Union[int, int]
        The index of the dominant emotion if conditions are met; otherwise, -1.
    """
    arr = np.array(agreement_dist)
    max_val = arr.max()
    if max_val < thresh:
        return -1
    # Use a small epsilon to handle floating point precision
    epsilon = 1e-6
    num_max = np.sum(np.isclose(arr, max_val, atol=epsilon))
    if num_max > 1:
        return -1
    return int(np.argmax(arr))