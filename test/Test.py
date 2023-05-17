import os

from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from model.AMCModel import AMCModel
from data.dataset.OFDMDataset import OFDMDataset, get_dataloader


class Test:

    SNR_VALUES = list(range(-4, 28, 4))

    def __init__(self, model_state_dict_path: str) -> None:
        self.__model = self.__get_model(model_state_dict_path)

    def __test_for_snr(self, snr_db: int, is_training: bool) -> Dict[str, Any]:
        dataset = self.__get_dataset(snr_db, is_training)
        dloader = self.__get_dloader(dataset)

        x, y = self.__test_model(self.__model, dloader)
        
    
    def __test_model(self, model: AMCModel, dloader: DataLoader) -> Tuple[List[int], List[int]]:
        preds, labels = [], []
        for idx, (x, y) in enumerate(dloader):
            pred = self.__get_prediction(model(x))
            label = self.__get_label(y)

            preds.append(pred)
            labels.append(label)

        return labels, preds

    def __get_dataset(self, snr_db: int, is_training: bool) -> OFDMDataset:
        assert snr_db in Test.SNR_VALUES, f'Invalid SNR value: {snr_db}!'
        dataset = OFDMDataset(snr_db=snr_db, is_training=is_training)
        return dataset

    def __get_dloader(self, dataset: OFDMDataset) -> DataLoader:
        return get_dataloader(dataset, batch_size=1, shuffle=1)

    def __get_model(self, path: str) -> AMCModel:
        assert os.path.exists(path), f'Invalid path: {path}!'
        model = AMCModel()
        model.load_state_dict(torch.load(path)())
        return model

    def __get_prediction(self) -> int:
        pass

    def __get_label(self) -> int:
        pass