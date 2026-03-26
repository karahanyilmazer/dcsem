#!/usr/bin/env python3

"""
This module contains functions for reading and writing to image files
"""

import glob
import os
from warnings import warn

import numpy as np
from fsl.data.image import Image
from fsl.transform import fnirt
from fsl.wrappers import convertwarp
from typing import List
from joblib import Parallel, delayed


def read_image(fname, mask):
    """
    Reads a nifti image with a provided mask
    :param fname: adress to the image
    :param mask: address to the mask
    :return: data [n, d] where n is the number of voxels in the mask and d is the 4th dimension of data.
    """
    mask_img = np.nan_to_num(Image(mask).data)
    data_img = Image(fname).data

    if data_img.ndim == 3:
        data_img = data_img[..., np.newaxis]
    data = data_img[mask_img > 0, :]

    return data


class NNMapping:
    """
    Non-linear transform between native and standard space.

    Compute the voxel-wise rotations from src to target given a mask in the target space and the transformation
    from source to mask. This function doesnt load data, it returns functions
    Args:
         src: address of the source image
         mask: address of the mask image
         xfm: transformation (warp field) from source to mask space, None if they are in the same space
    Return:
        mapping object with the methods:
            load_native:
            load_std:
            clean:
            restore:

    """

    def __init__(self, src, mask, xfm):
        self.mask_img, self.src_img = Image(mask), Image(src)
        self.ref_coords = np.array(np.where(np.nan_to_num(self.mask_img.data) > 0)).T
        self.n_vox = self.ref_coords.shape[0]

        if xfm is None:
            self.src_coords = self.ref_coords.copy()
            self.valid_indices = np.ones(self.src_coords.shape[0], dtype=bool)
            self.rot_mats = np.tile(np.eye(3), (self.n_vox, 1, 1))
        else:
            warp_img = Image(xfm)
            transform = fnirt.readFnirt(fname=warp_img, src=self.src_img, ref=self.mask_img, intent=2006)
            self.src_coords = np.around(transform.transform(self.ref_coords, 'voxel', 'voxel')).astype(int)
            self.valid_indices = np.all((self.src_coords < self.src_img.shape[:3]) & (self.src_coords > 0), axis=1)

            dx, dy, dz, _ = warp_img.pixdim
            grads = np.stack(np.gradient(warp_img.data, dx, dy, dz, axis=(0, 1, 2)), axis=-1)
            self.rot_mats = np.zeros((self.n_vox, 3, 3)) + np.eye(3)  # set default vaules to eye(3)
            self.rot_mats[self.valid_indices] = grads[tuple(self.ref_coords[self.valid_indices].T)]

            if transform.deformationType == 'relative':
                self.rot_mats[self.valid_indices] += np.eye(3)

        self.valid_src_coords = self.src_coords[self.valid_indices]
        self.valid_ref_coords = self.ref_coords[self.valid_indices]
        self.valid_rot_mats = self.rot_mats[self.valid_indices]
        self.n_valid_vox = len(self.valid_src_coords)

    def load_native(self, new_img):
        """load data from new image"""
        img = Image(new_img)
        data = img.data[tuple(self.valid_src_coords.T)]
        return data

    def load_std(self, new_img):
        img = Image(new_img)
        data = img.data[tuple(self.valid_ref_coords.T)]
        return data

    def restore(self, data):
        """ gets cleaned data (n_valid_vox, ...) returns full version (n_vox)"""
        if data.ndim == 1:
            shape = self.n_vox
        elif data.ndim == 2:
            shape = (self.n_vox, data.shape[1])
        else:
            shape = (self.n_vox, *data.shape[1:])

        c = np.zeros(shape)
        c[self.valid_indices == 1] = data
        c[self.valid_indices == 0] = np.nan
        return c

    def write_native(self, data, fname):
        """ writes the data to the native space
        Args:
            data: must have n_unique_vox rows
            fname: name of the output file.
        """
        assert data.shape[0] == self.n_valid_vox
        write_nifti(data, self.valid_src_coords, self.src_img, fname)

    def write_std(self, data, fname):
        """
        writes fata to std space
        Args:
            data: must have same rows as the unique voxels.
        """
        assert data.shape[0] == self.n_valid_vox
        write_nifti(self.restore(data), self.ref_coords, self.mask_img, fname)


def read_summary_images(summary_dir: str, subject_list: list, mask: str, summary_names: list):
    """
    Reads summary measure images
    :param summary_dir: path to the summary measurements, assumes the summaries are all in the standard space
    :param mask: roi mask in the standard space
    :param subject_list:
    :param summary_names:
    :return: 3d numpy array containing summary measurements, inclusion mask
    """

    mask_img = np.nan_to_num(Image(mask).data)
    if summary_names is None:
        summary_names = glob.glob(f'{summary_dir}/{subject_list[0]}/*.nii.gz')
        summary_names = [os.path.basename(s)[:-7] for s in summary_names]

    summaries = np.zeros((len(subject_list), (mask_img > 0).sum(), len(summary_names)))
    for i, sub in enumerate(subject_list):
        for j, m in enumerate(summary_names):
            summary_file = f'{summary_dir}/{sub}/{m}.nii.gz'
            summaries[i, :, j] = Image(summary_file).data[mask_img > 0]

    print(f'loaded summaries from {len(subject_list)} subjects.')
    invalid_voxs = np.isnan(summaries).any(axis=(0, 2))

    return summaries, invalid_voxs, summary_names


def convert_warp_to_deformation_field(warp_field, std_image, def_field, overwrite=True):
    """
    Converts a warp wield to a deformation field

    This is required because fnirt.Readfnirt only accepts deformation field. If the def_field
     exists it does nothing.
    :param warp_field:
    :param std_image:
    :param def_field:
    :param overwrite:
    :return:
    """
    if not os.path.exists(def_field) or overwrite is True:
        convertwarp(def_field, std_image, warp1=warp_field)
        img = Image(def_field)
        new_hdr = img.header.copy()
        new_hdr['intent_code'] = 2006  # for displacement field style warps
        Image(img.data, header=new_hdr).save(def_field)


def transform_indices(native_image, std_mask, def_field):
    """
    Findes nearest neighbour of each voxel of a standard space mask in a native space image.
    :param native_image: images object of native space
    :param std_mask: image object of a standard space mask
    :param def_field: string
    :return:
    """
    std_indices = np.array(np.where(np.nan_to_num(std_mask.data) > 0)).T
    transform = fnirt.readFnirt(def_field, native_image, std_mask)
    native_indices = np.around(transform.transform(std_indices, 'voxel', 'voxel')).astype(int)

    valid_vox = [np.all([0 < native_indices[j, i] < native_image.shape[i] for i in range(3)])
                 for j in range(native_indices.shape[0])]

    if not np.all(valid_vox):
        warn('Some voxels in mask lie out of native diffusion space.')

    return native_indices, valid_vox


def sample_from_native_space(image, xfm, mask, def_field=None):
    """
    Sample data from native space using a mask in standard space
    :param image: address to the image file in native space
    :param xfm: address to the transformation from standard to native
    :param mask: adress to the mask in standard space
    :param def_field: adress to the deformation field file, if not exsits will creat it, otherwise will use it.
    :return: sampled data, and valid voxels
    """
    convert_warp_to_deformation_field(xfm, mask, def_field)
    data_img = Image(image)
    mask_img = Image(mask)
    subj_indices, valid_vox = transform_indices(data_img, mask_img, def_field)
    data_vox = data_img.data[tuple(subj_indices[valid_vox, :].T)].astype(float)
    os.remove(def_field)
    return data_vox, valid_vox


def write_glm_results(data, delta_data, sigma_n, summary_names, mask, invalid_vox, glm_dir):
    """
    Writes the results of GLM into files,
    :param data: baseline measurement matrix (n_vox, n_sm)
    :param delta_data:  change matrix (n_vox, n_sm)
    :param sigma_n noise covariance of change (n_vox, n_sm, n_sm)
    :param glm_dir: path to the glm dir, it write data.nii.gz, delta_data.nii.gz, variance.nii.gz,
    and valid_mask.nii.gz in the address.
    :param mask: path to the mask file.
    :param invalid_vox: should be of size of mask voxels, contains 0 for valid voxels and 1 for invalid ones.
    This is used for droping voxels that are not valid.

    """
    os.makedirs(glm_dir, exist_ok=True)
    tril_idx = np.tril_indices(sigma_n.shape[-1])
    covariances = np.stack([s[tril_idx] for s in sigma_n], axis=0)

    os.makedirs(glm_dir, exist_ok=True)
    coords = np.argwhere(Image(mask).data>0)[invalid_vox==0]
    write_nifti(data, coords, mask, glm_dir + '/baseline.nii.gz')
    write_nifti(delta_data, coords,  mask, glm_dir + '/change.nii.gz')
    write_nifti(covariances, coords, mask, glm_dir + '/noise_cov.nii.gz')


    write_nifti(np.ones(coords.shape[0]), coords, mask, glm_dir + '/valid_mask.nii.gz')
    with open(f'{glm_dir}/summary_names.txt', 'w') as f:
        for t in summary_names:
            f.write("%s\n" % t)
        f.close()


def read_glm_results(glm_dir, mask=None):
    """
    :param glm_dir: path to the glm dir, it must contain data.nii.gz, delta_data.nii.gz, variance.nii.gz,
    and valid_mask.nii.gz
    :param mask: address of mask file, by default it uses the mask in glm dir.
    :return: tuple (data (n, d), delta_data (n, d), sigma_n(n, d, d) )
    """

    if mask is None:
        mask = glm_dir + '/valid_mask.nii.gz'

    mask_img = np.nan_to_num(Image(mask).data)
    data = Image(f'{glm_dir}/baseline.nii.gz').data[mask_img > 0, :]
    delta_data = Image(f'{glm_dir}/change.nii.gz').data[mask_img > 0, :]
    variances = Image(f'{glm_dir}/noise_cov.nii.gz').data[mask_img > 0, :]

    coords = np.argwhere(mask_img)
    n_vox, n_dim = data.shape
    tril_idx = np.tril_indices(n_dim)
    diag_idx = np.diag_indices(n_dim)
    sigma_n = np.zeros((n_vox, n_dim, n_dim))
    for i in range(n_vox):
        sigma_n[i][tril_idx] = variances[i]
        sigma_n[i] = sigma_n[i] + sigma_n[i].T
        sigma_n[i][diag_idx] /= 2

    with open(f'{glm_dir}/summary_names.txt', 'r') as reader:
        summary_names = [line.rstrip() for line in reader]

    return data, delta_data, sigma_n, summary_names, coords


def read_glm_weights(data: List[str], xfm: List[str],  mask: str, save_xfm_path: str, parallel=True):
    """
    reads voxelwise glm weights for each subject in an arbitrary space and a transformation from that space to standard,
    then takes voxels that lie within the mask (that is in standard space).

    :param data: list of nifti files one per subject
    :param xfm: list of transformations from the native space to standad space
    :param mask: address of roi mask in standard space
    :param parallel: flag for computing transformations in parallel.
    :param save_xfm_path: path to save computed xfms
    :returns: weights matrix (n_subj , n_vox). For the voxels that lie outside of image boundaries it places nans.

    """

    os.makedirs(save_xfm_path, exist_ok=True)
    mask_img = np.nan_to_num(Image(mask).data)
    std_indices = np.array(np.where(mask_img > 0)).T

    n_vox = std_indices.shape[0]
    n_subj = len(data)
    weights = np.zeros((n_subj, n_vox)) + np.nan
    print('Reading GLM weights:')

    def func(s_idx):
        subjdata, valid_vox = sample_from_native_space(
            data[s_idx], xfm[s_idx], mask, f"{save_xfm_path}/def_field_{s_idx}.nii.gz")
        print('weights for', s_idx, 'loaded.')
        return subjdata, valid_vox

    if parallel:
        res = Parallel(n_jobs=-1, verbose=True)(delayed(func)(i) for i in range(n_subj))
    else:
        res = []
        for i in range(n_subj):
            res.append(func(i))

    for subj_idx in range(n_subj):
        weights[subj_idx, res[subj_idx][1]] = res[subj_idx][0]

    return weights


def write_nifti(data: np.ndarray, coords: np.ndarray, ref_img: str, fname: str):
    """
    writes data to a nifti file.

    :param data: data matrix to be written to the file (M, d).
    :param ref_img: address to the reference image in target space (used to define resolution, FOV, etc)
    :param fname: path to the output nifti file
    :param coords: coordinates to write the files.
    :return:
    """
    if data.ndim == 1:
        data = data[:, np.newaxis]

    mask = Image(ref_img)
    if not np.all((coords < mask.shape[:3]) & (coords >= 0)):
        raise ValueError(" coordinated are not matched with the reference space.")
    img = np.zeros((*mask.shape[:3], data.shape[1]))
    img[tuple(coords.T)] = data
    Image(img, header=mask.header).save(fname)


def write_inference_results(path, model_names, predictions, posteriors, peaks, mask, coords):
    """
    Writes the results of inference to nifti files
    :param path: full path to write the files.
    :param model_names: name of the change vectors. (m, )
    :param predictions: index of the winning model (n,)
    :param posteriors: estimated posterior probability for each model(n, m)
    :param peaks: etimated amount of change for each model (n, m)
    :param mask: mask file that hass the information about the voxel locations.
    :return: Nothing.
    """
    os.makedirs(path, exist_ok=True)
    write_nifti(predictions[:, np.newaxis], coords, mask, f'{path}/inferred_change.nii.gz')
    for i, m in enumerate(model_names):
        if m == '[]' or m == 'nochange':
            write_nifti(posteriors[:, i][:, np.newaxis], coords, mask, f'{path}/nochange_probability.nii.gz')
        else:
            write_nifti(posteriors[:, i][:, np.newaxis], coords, mask, f'{path}/{m}_probability.nii.gz')
            write_nifti(peaks[:, i][:, np.newaxis], coords, mask, f'{path}/{m}_amount.nii.gz')


def read_inference_results(maps_dir, mask=None):
    """
    Reads the results of inference (posterior probabilities and estimated amount of change) for further analyses
    :param maps_dir: path to the results dir that contains probability maps
    :param mask: address of mask file, by default it uses the mask in the maps dir (if exists).
    :return: tuple (dict(models > probabilities (n_vox,1 )), dict(models > amounts)
    """

    if mask is None:
        mask = maps_dir + '/valid_mask.nii.gz'
    file_names = glob.glob(f'{maps_dir}/*probability*')
    model_names = [f.split('/')[-1].replace('_probability.nii.gz', '') for f in file_names]
    mask_img = np.nan_to_num(Image(mask).data)
    posteriors = dict()
    amounts = dict()
    for i, m in enumerate(model_names):
        if m == 'nochange':
            posteriors[m] = Image(f'{maps_dir}/nochange_probability.nii.gz').data[mask_img > 0]
            amounts[m] = np.zeros_like(posteriors[m])
        else:
            posteriors[m] = Image(f'{maps_dir}/{m}_probability.nii.gz').data[mask_img > 0]
            amounts[m] = Image(f'{maps_dir}/{m}_amount.nii.gz').data[mask_img > 0]

    return posteriors, amounts


def read_pes(pe_dir, mask_add):
    """
    Read parameter estimates.
    :param pe_dir:
    :param mask_add:
    :return:
    """
    mask_img = Image(mask_add)
    n_subj = len(glob.glob(pe_dir + '/subj_*.nii.gz'))
    pes = list()
    for subj_idx in range(n_subj):
        f = f'{pe_dir}/subj_{subj_idx}.nii.gz'
        pes.append(Image(f).data[mask_img.data > 0, :])

    print(f'loaded summaries from {n_subj} subjects')
    pes = np.array(pes)
    invalids = np.any(np.isnan(pes), axis=(0, 2))
    pes = pes[:, invalids == 0, :]
    if invalids.sum() > 0:
        warn(f'{invalids.sum()} voxels are dropped because of lying outside of brain mask in some subjects.')
    return pes, invalids


