#!/usr/bin/env python

import torch
from torch import nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple

from pytorch_lightning import LightningDataModule

import os
import time
import datetime
from collections import defaultdict
import copy

import numpy as np
from sklearn.metrics import matthews_corrcoef

from eolearn.core import LoadTask

import eotasks

log = utils.get_logger(__name__)

# Data set
class EODataset(Dataset):
    def __init__(self, 
                 options,
                ):
        '''
        Create an EODataset

        options : dictionary with the following entries
            flag : train / valid / test
            data_dir : data directory
            seed : random seed
            n_validation_patches : Number of EOPatches selected for validation
            cropped_length : Cropped EOPatch samples side length
            cropped_patch_number : number of smaller EOPatches to subsample
            cropped_is_random : True: Randomly select overlapping patches (False: systematically select non overlapping patches)
            n_time_frames : Number of time frames in EOPatches
            bands : Sentinel band names
            indices : Processed indices (choice of NDVI, NDWI, NDBI)

        '''

        if flag=='test':
            log.info(f'not implemented: {flag}')
            return
        # band specification
        # https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi/resolutions/spatial
        band_names = ['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B11','B12'] # from starter notebook
        band_wavelength = [443, 490, 560, 665, 705, 740, 783, 842, 865, 940, 1610, 2190] # nm
        band_spatial_resolution = [60, 10, 10, 10, 20, 20, 20, 10, 20, 60, 20, 20] # m

        # read from data_dir
        # division in train / valid or test
        # all available eopatches
        f_patches = os.listdir(data_dir)
        assert seed # else the shuffle needs to go somewhere else

        np.random.shuffle(f_patches) 

        if flag=='train':
            f_patches = f_patches[n_validation_patches:]
        elif flag=='valid':
            f_patches = f_patches[:n_validation_patches]
            log.info('EOPatches used for validation:', f_patches)
        else:
            raise ValueError("not implemented: ", flag)

        # load patches
        eo_load = LoadTask(path=data_dir)
        large_patches = []

        start_time = time.time()
        
        for f_patch in f_patches:
            eopatch = eo_load.execute(eopatch_folder=f_patch)
            large_patches.append(eopatch)

        log.info(f'loading {flag} data took {time.time()-start_time:.1f} seconds')

        # subsample to smaller images
        eo_sample = eotasks.SamplePatchletsTask(s2_patchlet_size=cropped_length, 
                                                num_samples=cropped_patch_number, 
                                                random_mode=cropped_is_random)

        small_patches = []

        start_time = time.time()

        min_patches = 100

        for patch in large_patches:
            min_patches = min(len(patch.data['BANDS']), min_patches)
            sp = eo_sample.execute(patch)
            small_patches.extend(sp)

        log.info(f'creating {len(small_patches)} small patches from {len(large_patches)} patches in {time.time()-start_time:.1f} seconds')
        log.info(f'minimum time frames: {min_patches}')

        # subsample time frame TODO
        tidx = list(range((n_time_frames+1)//2)) + list(range(-1*(n_time_frames//2), 0))
        #tidx = list(range(n_time_frames))
        log.info(f'selecting the first N//2 and the last N//2 time stamps: {tidx}')

        # subsample bands and other channels
        log.info('-- selecting bands --')
        for band in bands:
            log.info(f'  {band}')
        log.info('-- selecting normalized indices --')
        for index in indices:
            log.info(f'  {index}')

        lowres = []
        target = []
        weight = []
               
        for patch in small_patches:
            #log.info(f"time indices: {len(patch.data['BANDS'])}")
            x = []
            for ix in tidx: # outer most group: time index

class EODataModule(LightningDataModule):
    """
    LightningDataModule for EODatasets

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        dataset: dict = {},
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        get_variable_options(self.hparams.dataset)

    def prepare_data(self):
        """
        Adapt if necessary

        Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y).
        """

        #EODataset(self.hparams.data_dir, train=True, download=True)
        #EODataset(self.hparams.data_dir, train=False, download=True)
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning twice for `trainer.fit()` and `trainer.test()`, so be careful if you do a random split!
        The `stage` can be used to differentiate whether it's called before trainer.fit()` or `trainer.test()`."""

        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            # load the EODataset for train and test
            # Split the train set in train and validation
            # assign the variables data_train etc
            pass

        if stage in (None, 'fit'):
            self.data_train = EODataset(options=self.hparams.dataset)
            self.data_val   = EODataset(options=self.hparams.dataset)
        if stage in (None, 'test'):
            self.data_test  = EODataset(options=self.hparams.dataset)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
