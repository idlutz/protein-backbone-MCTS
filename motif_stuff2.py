#!/usr/bin/env python

import os
import sys


sys.path.append("/home/bcov/from/derrick/getpy_motif/")
sys.path.append("/home/bcov/sc/random/npose")

import future_motif_hash as fmh

import npose_util as nu
import numpy as np

from numba import njit


@njit(fastmath=True, cache=True)
def get_good_xforms(frames, max_dist2, who_is, care_mask):
    inverses = []
    for i in range(len(frames)):
        inverses.append(np.linalg.inv(frames[i]))
    good_xforms = []
    misses = 0

    i_s = []
    j_s = []

    for i in range(len(frames)):
        i_care = care_mask[i]
        for j in range(len(frames)):
            if (i == j):
                continue
            if ( not i_care and not care_mask[j] ):
                continue
            frame = frames[j]
            inv = inverses[i]
            x = frame[0,3] - inv[0,3]
            y = frame[1,3] - inv[1,3]
            z = frame[2,3] - inv[2,3]
            if ( x*x + y*y + z*z < max_dist2 ):
                misses += 1
            # dist2 = np.sum(np.square(xform[:3,3]))
            # if ( dist2 > max_dist2 ):
            #     misses += 1
            #     continue

            xform = inverses[i] @ frames[j]
            good_xforms.append(xform)
            i_s.append(who_is[i])
            j_s.append(who_is[j])

    return good_xforms, i_s, j_s, misses


def get_all_xforms( frames ):
    xforms = np.zeros((len(frames), len(frames), 4, 4), np.float)

    inverses = np.linalg.inv(frames)

    for i in range(len(frames)):
        xforms[i,:,:,:] = inverses[i] @ frames

    return xforms

search_mask = fmh.get_search_mask("FILV", "FLIV")
# search_mask = fmh.get_search_mask("G", "G")


def motif_score_npose_all( npose, sm=search_mask ):

    frames = nu.npose_to_derp_hash_frames(npose)

    xforms = get_all_xforms(frames).reshape((-1, 4, 4))

    xform_hits = fmh.get_masked_hits( xforms, sm ).reshape((len(frames), len(frames)))


    xform_hits = xform_hits | xform_hits.T

    # distances = np.linalg.norm(xforms[:,:,:3,3], axis=-1)

    return xform_hits#, distances


def motif_score_npose( npose, care_mask=None, calc_mask=None ):
    max_valid_dist = 12 
    max_valid_dist2 = max_valid_dist**2

    if ( care_mask is None ):
        care_mask = np.ones(nu.nsize(npose), np.bool)
    if ( calc_mask is None ):
        calc_mask = np.ones(nu.nsize(npose), np.bool)

    frames = nu.npose_to_derp_hash_frames(npose)

    frames = frames[calc_mask]
    care_mask = care_mask[calc_mask]
    who_is = np.arange(nu.nsize(npose), dtype=np.int)[calc_mask]

    good_xforms, froms, tos, misses = get_good_xforms(frames, max_valid_dist2, who_is, care_mask)
    good_xforms = np.array(good_xforms)

    misses = 0
    hits = 0

    if ( len(good_xforms) == 0 ):
        return hits, np.array([]), np.array([]), misses

    xform_hits = fmh.get_masked_hits( good_xforms, search_mask )

    hits = np.sum( xform_hits )

    good_froms = np.array(froms)[xform_hits]
    good_tos = np.array(tos)[xform_hits]

    # uncomment this to find max_valid_dist
    # distances = np.linalg.norm(good_xforms[:,:3,3], axis=-1)
    # hit_distances = distances[xform_hits]
    # print(np.max(hit_distances))

    return hits, good_froms, good_tos, misses


def get_good_xforms_asym(frames1, frames2, max_dist2, who_is1, who_is2):
 
    inverses1 = [np.linalg.inv(frames1[i]) for i in range(len(frames1))]
    # we don't need inverses2

    good_xforms = []
    misses = 0

    i_s = []
    j_s = []

    for i in range(len(frames1)):
    
        for j in range(len(frames2)):

            frame = frames2[j]
            inv = inverses1[i]
            x = frame[0,3] - inv[0,3]
            y = frame[1,3] - inv[1,3]
            z = frame[2,3] - inv[2,3]
            if ( x*x + y*y + z*z < max_dist2 ):
                misses += 1
            # dist2 = np.sum(np.square(xform[:3,3]))
            # if ( dist2 > max_dist2 ):
            #     misses += 1
            #     continue

            xform = inverses1[i] @ frames2[j]
            good_xforms.append(xform)
            i_s.append(who_is1[i])
            j_s.append(who_is2[j])

    return good_xforms, i_s, j_s, misses




def motif_score_npose_asym( npose1, npose2 ):
    
    # added by Isaac Lutz 10/27/21 to speed up compute
    # only calc pairs across two nposes -- interfaces
    # removed mask -- I just mask outside of function
    # and didn't want to deal with 2x every data structure
    
    max_valid_dist = 12
    max_valid_dist2 = max_valid_dist**2

    frames1 = nu.npose_to_derp_hash_frames(npose1)
    frames2 = nu.npose_to_derp_hash_frames(npose2)

    who_is1 = np.arange(nu.nsize(npose1), dtype=np.int)
    who_is2 = np.arange(nu.nsize(npose2), dtype=np.int)

    good_xforms, froms, tos, misses = get_good_xforms_asym(frames1, frames2, max_valid_dist2, who_is1, who_is2)
    good_xforms = np.array(good_xforms)

    misses = 0
    hits = 0

    if ( len(good_xforms) == 0 ):
        return hits, np.array([]), np.array([]), misses

    xform_hits = fmh.get_masked_hits( good_xforms, search_mask )

    hits = np.sum( xform_hits )

    good_froms = np.array(froms)[xform_hits]
    good_tos = np.array(tos)[xform_hits]

    # uncomment this to find max_valid_dist
    # distances = np.linalg.norm(good_xforms[:,:3,3], axis=-1)
    # hit_distances = distances[xform_hits]
    # print(np.max(hit_distances))

    return hits, good_froms, good_tos, misses










