import os
import json

import numpy as np
from enum import Enum
from datetime import datetime as dt
from typing import (
    Any,
    Dict,
    List,
    Tuple,
    Union,
    )

import torch
from torch.utils.data import DataLoader

from model.AMCModel import AMCModel
from model_baseline.AMCModelBaseline import AMCModelBaseline
from data.dataset.OFDMDataset import (
    OFDMDataset,
    get_dataloader
    )

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    )

class ModelTypes(str, Enum):
    AMCModel = "AMCModel"
    AMCModelBaseline = "AMCModelBaseline"
    AMCModelBaselineAlternative = "AMCModelBaselineAlternative"

    def __str__(self) -> str:
        return self.name

class Test:

    SNR_VALUES = list(range(-4, 28, 4))

    def __init__(self, model_state_dict_path: str, write_dir: str) -> None:
        self.__model = self.__get_model(model_state_dict_path)
        self.__write_dir = write_dir

    def run(self) -> None:
        tr_te__snr_map = dict()
        for is_training in (True, False):
            snr__metric_map = dict()
            for snr in Test.SNR_VALUES:
                snr__metric_map[str(snr)] = self.__test_for_snr(snr, is_training)
            tr_te__snr_map["train" if is_training else "test"] = snr__metric_map 
        self.__write_json(tr_te__snr_map)
    
    def __write_json(self, res: Dict[str, Any]) -> None:
        date = dt.today().strftime("%Y%m%d")
        filename = f"stats__{self.__model_type}__{date}.json"
        filepath = os.path.join(self.__write_dir, filename)

        os.makedirs(self.__write_dir, exist_ok=True)
        with open(file=filepath, mode="w") as fp:
            json.dump(res, fp, indent=3)

    def __test_for_snr(self, snr_db: int, is_training: bool) -> Dict[str, Any]:
        dataset = self.__get_dataset(snr_db, is_training)
        dloader = self.__get_dloader(dataset)

        x, y = self.__test_model(self.__model, dloader)

        res = dict()
        res["conf_matrix"] = confusion_matrix(y, x).tolist()
        res["accuracy"] = accuracy_score(y, x, normalize=True)
        res["f_score"] = f1_score(y, x, average="micro")

        return res
    
    def __test_model(self, model: Union[AMCModel, AMCModelBaseline], dloader: DataLoader) -> Tuple[List[int], List[int]]:
        preds, labels = [], []
        for idx, (x, y) in enumerate(dloader):
            pred = self.__get_prediction(model(x))
            label = self.__get_label(y)

            preds.append(pred)
            labels.append(label)

            if idx == 1000:
                break

        return labels, preds

    def __get_dataset(self, snr_db: int, is_training: bool) -> OFDMDataset:
        assert snr_db in Test.SNR_VALUES, f'Invalid SNR value: {snr_db}!'
        dataset = OFDMDataset(snr_db=snr_db, is_training=is_training)
        return dataset

    def __get_dloader(self, dataset: OFDMDataset) -> DataLoader:
        return get_dataloader(dataset, batch_size=1, shuffle=1)

    def __get_model(self, path: str) -> Union[AMCModel, AMCModelBaseline]:
        assert os.path.exists(path), f'Invalid path: {path}!'

        state_dict = torch.load(path)
        model_type = list(state_dict.keys())[0].split("_")[1]
        assert model_type in [str(t) for t in ModelTypes], "Invalid model type!"

        self.__model_type = model_type
        if model_type == ModelTypes.AMCModel: model = AMCModel()
        elif model_type == ModelTypes.AMCModelBaseline: model = AMCModelBaseline()
        elif model_type == ModelTypes.AMCModelBaselineAlternative: pass     # TODO: add new model

        model.load_state_dict(state_dict)
        return model

    def __get_prediction(self, prediction: torch.tensor) -> int:
        pred = torch.zeros_like(prediction)
        pred[torch.argmax(torch.softmax(prediction, dim=0)).item()] = 1
        return np.argmax(pred.numpy())

    def __get_label(self, label: torch.tensor) -> int:
        return np.argmax(label.numpy())