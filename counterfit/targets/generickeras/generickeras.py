import os
import numpy as np
import gzip

from counterfit.core.state import ArtTarget
from counterfit.core import config
from zipfile import ZipFile
from ember import PEFeatureExtractor
import tensorflow as tf
import tqdm
import requests


def download_samples_to(filename, url):
    r = requests.get(url)
    if r.ok:
        print(f"\n[+] Successfully downloaded malware samples to the location {filename}\n")
        # open method to open a file on your system and write the contents to the file location specified
        with open(filename, "wb") as code:
            code.write(r.content)
    else:
        print(r.status_code)

class GenericKeras(ArtTarget):
    model_name = 'generickeras'
    model_data_type = "pe"
    model_output_classes = ["benign", "malicious"]
    model_input_shape = (1,)
    model_location = "local"
    model_endpoint = os.path.join(
        config.targets_path, 'generickeras/model.h5')

    sample_input_path = f"{config.targets_path}/generickeras/malware_samples.zip"
    encryption_password = b'infected'

    X = []
    zip_info = []

    def __init__(self):
        # load model (gunzip)
        # with gzip.open(self.model_endpoint, 'rb') as f_in:
            # model_data = f_in.read().decode('ascii')

        self.model = tf.keras.models.load_model(self.model_endpoint)
        self.extractor = PEFeatureExtractor(2)  # feature_version=2
        self.thresh = 0.5

        # download samples
        # if not os.path.exists(self.sample_input_path):
        #     download_samples_to(self.sample_input_path)
            
        print(f"scanning malware sample info from {self.sample_input_path}")
        with ZipFile(self.sample_input_path) as thezip:
            thezip.setpassword(self.encryption_password)
            for zipinfo in tqdm.tqdm(thezip.infolist()):
                self.zip_info.append(zipinfo)

        self.X = self.read_input_data()
    
    def read_input_data(self):
        out = []
        with ZipFile(self.sample_input_path) as thezip:
            for i in range(len(self.zip_info)):
                with thezip.open(self.zip_info[i], pwd=self.encryption_password) as thefile:
                    out.append(thefile.read())
        return out

    def outputs_to_labels(self, output):
        # override default to incorporate threshold
        output = np.atleast_2d(output)
        return [self.model_output_classes[int(o[1] > self.thresh)] for o in output]

    def set_attack_samples(self, index=0):
        # JIT loading of samples
        if hasattr(index, "__iter__"):
            # list of indices
            to_fetch = index
        else:
            to_fetch = [index]

        out = []
        for i in to_fetch:
            out.append(self.X[i])
        self.active_attack.sample_index = index
        self.active_attack.samples = out

    def __call__(self, batch):
        features = []
        for bytez in batch:
            feat = np.array(self.extractor.feature_vector(bytez.tobytes()), dtype=np.float32)
            features.append(feat)

        scores = self.model.predict(np.array(features).reshape(-1, 2381)).T  # shape: (batch_size, )
        scores = np.vstack([1-scores, scores]).T  # shape: (batch_size, 2)
        return scores
