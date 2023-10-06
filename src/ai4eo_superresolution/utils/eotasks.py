#!/usr/bin/env python
# Built-in modules
import os
import json
import datetime as dt
from typing import Tuple, List

# Basics of Python data handling and visualization
import numpy as np
import pandas as pd
import geopandas as gpd

# Imports from eo-learn and sentinelhub-py
from sentinelhub import CRS, BBox, SHConfig, DataCollection

from eolearn.core import (FeatureType,
                          EOPatch, 
                          EOTask, 
                          EONode,
                          EOWorkflow,
                          EOExecutor, 
                          LoadTask,
                          SaveTask)

# Visualisation utilities from utils.py
from ai4eo_superresolution.utils.starter_pack_utils import get_extent

from scipy.stats import skewnorm

from skimage import measure
from skimage.morphology import binary_dilation, disk

import torch

class SamplePatchletsTask(EOTask):
    '''
    Sample patchlets from EOTask
    '''

    SCALE_FACTOR = 4 # do not change

    def __init__(self, s2_patchlet_size: int, num_samples: int, random_mode: bool):
        """ Set-up of task 
        
        :param s2_patchlet_size: Size in pixels of resulting patchlet
        :param num_samples: Number of patchlets to sample
        """
        self.s2_patchlet_size = s2_patchlet_size
        self.num_samples = num_samples
        self.random_mode = random_mode

    def _calculate_sampled_bbox(self, bbox: BBox, r: int, c: int, s: int,
                                resolution: float) -> BBox:
        """ Calculate bounding box of smaller patchlets """
        return BBox(((bbox.min_x + resolution * c,  bbox.max_y - resolution * (r + s)),
                     (bbox.min_x + resolution * (c + s), bbox.max_y - resolution * r)),
                    bbox.crs)

    def _sample_s2(self, eop: EOPatch, row: int, col: int, size: int, 
                   resolution: float = 10):
        """ Randomly sample a patchlet from the EOPatch """
        # create a new eopatch for each sub-sample
        sampled_eop = EOPatch(timestamp=eop.timestamp, 
                              scalar=eop.scalar, 
                              meta_info=eop.meta_info)
        
        # sample S2-related arrays
        features = eop.get_feature_list()
        s2_features = [feature for feature in features 
                       if isinstance(feature, tuple) and 
                       (feature[0].is_spatial() and feature[0].is_time_dependent())]
        
        for feature in s2_features:
            sampled_eop[feature] = eop[feature][:, row:row + size, col:col + size, :]
        
        # calculate BBox for new sub-sample
        sampled_eop.bbox = self._calculate_sampled_bbox(eop.bbox, 
                                                        r=row, c=col, s=size, 
                                                        resolution=resolution)
        sampled_eop.meta_info['size_x'] = size
        sampled_eop.meta_info['size_y'] = size
        
        # sample from target maps, beware of `4x` scale factor
        target_features = eop.get_feature(FeatureType.MASK_TIMELESS).keys()
        
        for feat_name in target_features:
            sampled_eop.mask_timeless[feat_name] = \
            eop.mask_timeless[feat_name][self.SCALE_FACTOR*row:self.SCALE_FACTOR*row + self.SCALE_FACTOR*size, 
                                         self.SCALE_FACTOR*col:self.SCALE_FACTOR*col + self.SCALE_FACTOR*size]
        
        # sample from weight maps, beware of `4x` scale factor
        target_features = eop.get_feature(FeatureType.DATA_TIMELESS).keys()
        
        for feat_name in target_features:
            sampled_eop.data_timeless[feat_name] = \
            eop.data_timeless[feat_name][self.SCALE_FACTOR*row:self.SCALE_FACTOR*row + self.SCALE_FACTOR*size, 
                                         self.SCALE_FACTOR*col:self.SCALE_FACTOR*col + self.SCALE_FACTOR*size]
        
        return sampled_eop

    def execute(self, eopatch_s2: EOPatch, buffer: int=0,  seed: int=42, random_mode: bool=1) -> List[EOPatch]:
        """ Sample a number of patchlets from the larger EOPatch. 
        
        :param eopatch_s2: EOPatch from which patchlets are sampled
        :param buffer: Do not sample in a given buffer at the edges of the EOPatch
        :param seed: Seed to initialise the pseudo-random number generator
        :param random_mode: Select the upper left corner at random (default: True)
        """
        _, n_rows, n_cols, _ = eopatch_s2.data['BANDS'].shape
        np.random.seed(seed)
        eops_out = []
        
        if not self.random_mode:
            max_per_row = n_rows // self.s2_patchlet_size
            max_per_col = n_cols // self.s2_patchlet_size
        
        # random sampling of upper-left corner. Added: Change this for non-overlapping patchlets
        for patchlet_num in range(0, self.num_samples):
            if self.random_mode:
                row = np.random.randint(buffer, n_rows - self.s2_patchlet_size - buffer)
                col = np.random.randint(buffer, n_cols - self.s2_patchlet_size - buffer)
            else:
                row = (buffer + patchlet_num // int(np.floor((n_rows - buffer) / self.s2_patchlet_size)) * self.s2_patchlet_size)
                col = buffer + (patchlet_num * self.s2_patchlet_size) % (n_cols - buffer - self.s2_patchlet_size)
                
                row = (patchlet_num // max_per_row) * self.s2_patchlet_size
                col = (patchlet_num % max_per_col) * self.s2_patchlet_size
                
                
            sampled_s2 = self._sample_s2(eopatch_s2, row, col, self.s2_patchlet_size)
            eops_out.append(sampled_s2)

        return eops_out

class ComputeReflectances(EOTask):
    """ Apply normalisation factors to DNs (from starter notebook)"""
    def __init__(self, feature):
        self.feature = feature
        
    def execute(self, eopatch):
        eopatch[self.feature] = eopatch.scalar['NORM_FACTORS'][..., None, None] \
            * eopatch[self.feature].astype(np.float32)
        return eopatch

class SentinelHubValidData:
    """
    Combine 'CLM' mask with `IS_DATA` to define a `VALID_DATA_SH` mask
    The SentinelHub's cloud mask is asumed to be found in eopatch.mask['CLM']
    """
    def __call__(self, eopatch):
        return eopatch.mask['IS_DATA'].astype(bool) & np.logical_not(eopatch.mask['CLM'].astype(bool))

    
class AddValidCountTask(EOTask):
    """
    The task counts number of valid observations in time-series and stores the results in the timeless mask.
    """
    def __init__(self, count_what, feature_name):
        self.what = count_what
        self.name = feature_name

    def execute(self, eopatch):
        eopatch[(FeatureType.MASK_TIMELESS, self.name)] = np.count_nonzero(eopatch.mask[self.what], axis=0)
        return eopatch
    
    
class ValidDataFractionPredicate:
    """ Predicate that defines if a frame from EOPatch's time-series is valid or not. Frame is valid if the
    valid data fraction is above the specified threshold.
    """
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, array):
        coverage = np.sum(array.astype(np.uint8)) / np.prod(array.shape)
        return coverage > self.threshold

class NanDataPredicate:
    """ Predicate that defines if a frame from EOPatch's time-series contains nans --> invalid """
    def __init__(self):
        pass

    def __call__(self, array):
        nancount = np.sum(np.isnan(array))
        return nancount==0

def weighting_function(pix_size: int, median_pix_size: int, highest_weight_pix_size: int = 35,
                       skewness: int = 15) -> float:
    """ Creates weight to be applied to a parcel depending on its number of pixels (after pixelation) """
    if pix_size >= median_pix_size:
        return 1
    
    xs = np.linspace(1, median_pix_size, median_pix_size)
    y1 = skewnorm.pdf(xs, skewness, loc=highest_weight_pix_size-100/3.14, scale=100)
    y1 = y1 / max(y1)
    y1 = y1 + 1

    return y1[int(pix_size)].astype(np.float)


class AddWeightMapTask(EOTask):
    """ Computes the weight map used to compute the validation metric """

    def __init__(self, 
                 cultivated_feature: Tuple[FeatureType, str], 
                 not_declared_feature: Tuple[FeatureType, str], 
                 weight_feature: Tuple[FeatureType, str], 
                 radius: int = 2, seed: int = 4321):
        self.cultivated_feature = cultivated_feature
        self.not_declared_feature = not_declared_feature
        self.weight_feature = weight_feature
        self.radius = radius
        self.seed = seed
        
    def execute(self, eopatch: EOPatch) -> EOPatch:
        cultivated = eopatch[self.cultivated_feature].astype(np.uint8).squeeze()
        not_declared = eopatch[self.not_declared_feature].squeeze()

        np.random.seed(self.seed)

        # compute connected components on binary mask
        conn_comp = measure.label(cultivated, background=0)
        # number of connected components
        n_comp = np.max(conn_comp) + 1

        # Placeholder for outputs
        height, width = cultivated.shape
        weights = np.zeros((height, width), dtype=np.float32)
        contours = np.zeros((height, width), dtype=np.float32)
        counts = np.zeros((height, width), dtype=np.uint8)

        # Loop over connected components, ignoring background
        for ncc in np.arange(1, n_comp):
            parcel_mask = conn_comp == ncc
            # number of pixels of each component, i.e. parcel
            n_pixels = np.sum(parcel_mask)

            # compute external boundary of parcel 
            dilated_mask = binary_dilation(parcel_mask, selem=disk(radius=self.radius))
            contour = np.logical_and(~parcel_mask, dilated_mask)

            weight = weighting_function(n_pixels, median_pix_size=400)

            weights[parcel_mask] = weight
            contours += 2 * weight * contour
            # In case countours overlap, the average weight is taken
            counts += contour

        # combine weights from all parcels into a single map. First add (averaged) contours,
        # then weighted parcels, then background 
        weight_map = np.zeros((height, width), dtype=np.float32)
        weight_map[contours > 0] = contours[contours > 0] / counts[contours > 0]
        weight_map[weights > 0] = weights[weights > 0]
        weight_map[weight_map == 0] = 1

        # add zero weights at border and undeclared parcels
        weight_map[not_declared == True] = 0
        weight_map[:1, :] = 0
        weight_map[:, :1] = 0
        weight_map[-2:, :] = 0
        weight_map[:, -2:] = 0

        eopatch[self.weight_feature] = weight_map[..., np.newaxis]
        
        return eopatch

class PredictPatchTask(EOTask):
    """
    https://eo-learn.readthedocs.io/en/latest/examples/land-cover-map/SI_LULC_pipeline.html#6.-Model-construction-and-training
    Task to make model predictions on a patch. Provide the model 
    """
    def __init__(self, model, features_feature, args):
        self.model = model
        self.features_feature = features_feature
        self.args = args
    
    def execute(self, eopatch):
        pred_eopatch = EOPatch(bbox=eopatch.bbox)
        # TODO repeat the preprocessing from EODataset
        band_names = ['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B11','B12']
        tidx = list(range((self.args.n_time_frames+1)//2)) + list(range(-1*(self.args.n_time_frames//2), 0))
        #tidx = list(range(self.args.n_time_frames))
        print(f'selecting the first N//2 and the last N//2 time stamps: {tidx}')

        print(self.args.indices)

        x = []
        for ix in tidx: # outer most group: time index
            print(f'Time index {ix}')
            for band in self.args.bands:
                band_ix = band_names.index(band)
                if len(eopatch.data['BANDS']) > ix:
                    xx = eopatch.data['BANDS'][ix][:, :, band_ix]
                else:
                    xx = eopatch.data['BANDS'][0][:, :, band_ix]
                x.append(xx.astype(np.float32))
            for index in self.args.indices:
                if len(eopatch.data['BANDS']) > ix:
                    xx = eopatch.data[index][ix]
                else:
                    xx = eopatch.data[index][0]
                x.append(xx.astype(np.float32).squeeze())
                print(f'Normalized index {index} attached')
        x = np.expand_dims(np.stack(x), axis=0)
        x = torch.tensor(x.astype(np.float32))
        print('Input shape: ', x.shape)

        with torch.no_grad():
            prediction = self.model(x)
        print('Output shape: ', prediction.shape)
        # reshape to expected output shape
        prediction = prediction.numpy().squeeze()
        prediction = prediction[:, :, np.newaxis]
        prediction = np.round(prediction).astype(np.uint8)
        pred_eopatch[(FeatureType.MASK_TIMELESS, 'PREDICTION')] = prediction
        return pred_eopatch

