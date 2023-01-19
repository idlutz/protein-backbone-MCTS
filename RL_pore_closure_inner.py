#!/usr/bin/env python
# coding: utf-8


import sys

sys.path.append('/home/ilutz/BINDERS_RL/2_RL/tree_source/')

from math import cos,sin,tan,asin,acos,radians,sqrt,degrees,atan,atan2,copysign,exp
from math import pi as mPI
from math import pi
import numpy as np
import matplotlib.pyplot as plt
import io
import tempfile
import os
import glob
import seaborn as sns
import pandas as pd

import itertools
import operator
from functools import reduce
import pickle
from sklearn.cluster import KMeans
import scipy
from scipy.stats import norm
import random
import statistics
import time
import timeit
import math
import localization as lx
import gzip

import npose_util as nu
import motif_stuff2

import subprocess
import datetime

import math
from collections import defaultdict
import time
import argparse
import itertools
import subprocess
import getpy
import xbin
import h5py

import voxel_array
import npose_util as nu

from importlib import reload
reload(nu)

base_name = sys.argv[1].split('/')[-1][:-4]
t1 = nu.npose_from_file(sys.argv[1])
inner = t1[306*5*6:]
stub = np.append(t1[:306*5][10:], inner[:int(len(inner)/6)], 0)
wnum_hash = []

inner_r = inner.reshape(int(len(inner)/5),5,4)
l = int(len(inner_r) / 6)
inner_r = [inner_r[l*x:l*(x+1)] for x in range(6)]
inner_terminii = [x[-1][nu.CA][:3] for x in inner_r]


# vals for determining cone -- outer constant, inner variable
outer_maxz = 25
outer_minz = -25
outer_rad = 55

inner_maxz = max([x[2] for x in inner]) + 5
inner_minz = min([x[2] for x in inner]) - 5
inner_rad = 20

max_slope = (inner_maxz - outer_maxz) / (inner_rad - outer_rad)
min_slope = (inner_minz - outer_minz) / (inner_rad - outer_rad)
max_yint = inner_maxz - (max_slope * inner_rad)
min_yint = inner_minz - (min_slope * inner_rad)

def get_zlims(atom_rad):
    
    zmax = max_slope*atom_rad + max_yint
    zmin = min_slope*atom_rad + min_yint
    
    return zmax, zmin



zero_ih = nu.npose_from_file('/home/ilutz/BINDERS_RL/2_RL/zero_ih_long.pdb')

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def c_op(pt_set,c_num): # Cn symmetry operations
    
    # translation
    trans = [0,0,0]
    
    angle_inc = math.pi*2 / c_num
    concat_list = [pt_set]
    
    for i in range(c_num - 1):
        
        # rotation around z axis centered at origin -- for Rosetta Z sym scripts
        rot = euler_to_R(0,0,angle_inc*(i+1))

        # make xform from translation, rotation
        pre = np.concatenate((rot,np.array([[0,0,0]])), axis=0)
        xform = np.concatenate((pre,np.array([[trans[0]],[trans[1]],[trans[2]],[1]])), axis=1)

        # add it to list
        concat_list.append(nu.xform_npose(xform, pt_set))
    
    return np.concatenate(tuple(concat_list))


rr = np.load('/home/ilutz/BINDERS_RL/2_RL/all_loops_bCov.npz', allow_pickle=True)
all_loops = [rr[f] for f in rr.files][0]
    

def align_loop(build, end, loop):
    # returns loop aligned to assigned end of build
    tpose1 = nu.tpose_from_npose(loop)
    tpose2 = nu.tpose_from_npose(build)
    
    itpose1 = np.linalg.inv(tpose1)
    
    if end == 'C': # C term addition
        xform = tpose2[-1] @ itpose1[0]
        aligned_npose1 = nu.xform_npose(xform, loop)
        return aligned_npose1[5:]
    
    else: # N term addition
        xform = tpose2[0] @ itpose1[-1]
        aligned_npose1 = nu.xform_npose(xform, loop)
        return aligned_npose1[:-5]
    
    
def extend_helix(build, end, res):
    
    if end == 'C': # C term addition
        ext = align_loop(build, 'C', zero_ih)
        return ext[:(res*5)]
    
    else: # N term addition
        # note: can't handle 0 ext, but doesn't need to
        ext = align_loop(build, 'N', zero_ih)
        return ext[-(res*5):]


def check_hash(pt_set,wnum_hash): 
    
    if len(wnum_hash) == 0:
        return True
    
    for i in pt_set:
        
        key = str(int(round(i[0])))+'_'+str(int(round(i[1])))+'_'+str(int(round(i[2])))
        
        if key not in wnum_hash:
            return False
            
    return True


def check_clash(build_set, build_end, query_set, clash_threshold = 2.85, diameter_threshold = 999,seq_buff=5):
    if len(build_set) == 0 or len(query_set) == 0:
        return True

    if len(query_set) < seq_buff:
        seq_buff = len(query_set)
    elif len(build_set) < seq_buff:
        seq_buff = len(build_set)

    axa = scipy.spatial.distance.cdist(build_set,query_set)
    
    if seq_buff > 0:
        for i in range(seq_buff):
            for j in range(seq_buff-i):
                # add sequence buffer to different ends accordingly
                if build_end == 'C':
                    axa[-(i+1)][j] = clash_threshold + 0.1
                else:
                    axa[i][-(j+1)] = clash_threshold + 0.1

    if np.min(axa) < clash_threshold: # clash condition
        return False
    if np.max(axa) > diameter_threshold: # compactness condition
        return False
    
    return True


def get_avg_sc_neighbors(ca_cb, care_mask):
    
    conevect = (ca_cb[:,1] - ca_cb[:,0] )

    conevect /= 1.5

    maxx = 11.3
    max2 = maxx*maxx

    neighs = np.zeros(len(ca_cb))

    for i in range(len(ca_cb)):

        if ( not care_mask[i] ):
            continue
        
        vect = ca_cb[:,0] - ca_cb[i,1]
        vect_length2 = np.sum( np.square( vect ), axis=-1 )

        vect = vect[(vect_length2 < max2) & (vect_length2 > 4)]
        vect_length2 = vect_length2[(vect_length2 < max2) & (vect_length2 > 4)]

        vect_length = np.sqrt(vect_length2)
        
        vect = vect / vect_length[:,None]

        dist_term = np.zeros(len(vect))
        
        for j in range(len(vect)):
            if ( vect_length[j] < 7 ):
                dist_term[j] = 1
            elif (vect_length[j] > maxx ):
                dist_term[j] = 0
            else:
                dist_term[j] = -0.23 * vect_length[j] + 2.6

        angle_term = ( np.dot(vect, conevect[i] ) + 0.5 ) / 1.5
        angle_term[angle_term < 0] = 0

        neighs[i] = np.sum( dist_term * np.square( angle_term ) )

    return neighs




def fast_pore_score(bd):
    # most accurate yet -- max approximate percent volume filled for two atom bounded shell
    # use van der Waal radius for carbon (1.7 A) -- oxygen and nitrogen ~1.5, close enough
    # matrix operations for speed -- 5-10x faster
    # also includes a check for at least one decently exposed terminus -- if not, return 0
    # picked threshold both qualitatively and to discard ~20% as with first round -- 0.45 percentile
    
    # adapted for pore closure -- instead of using radial distance as in cages, use z dist instead
    # and occupancy of cylinder
    
    c_vdw = 1.7
    dists = [pt[2] for pt in bd[len(stub):]]
    rads = sorted(dists)
    
    nds = [x for x in range(len(rads))]

    inds = np.array(nds*len(nds)).reshape(len(nds),len(nds))
    jnds = np.transpose(inds)
    atoms_vols = 6 * (jnds-inds+1) * (4/3)*pi*c_vdw**3

    its = np.array(rads*len(rads)).reshape(len(rads),len(rads))
    jts = np.transpose(its)
    shell_vols = ((jts+c_vdw) - (its-c_vdw))*pi*(50**2) # cylinder height times

    mask = jnds > inds

    pore_vals = (atoms_vols / shell_vols)*mask

    return np.nanmax(pore_vals) # forget nans


# In[217]:


def interchain_motif_score(input_pose):
    
    # speed up -- take all by all dists of chA vs. subunit set
    # mask/remove everything above a distance threshold
    # motif score only what is left -- just potential interfaces
    # mask all self interactions
    
    # in motif score code, this appears to be 12 -- TEST
    dist_threshold = 12
    
    whole = c_op(input_pose,6)
    whole_reshape = whole.reshape(int(len(whole)/5),5,4)
    whole_cbs = whole_reshape[:,nu.CB]
    
    ip_reshape = input_pose.reshape(int(len(input_pose)/5),5,4)
    ip_cbs = ip_reshape[:,nu.CB]
    
    axa = scipy.spatial.distance.cdist(ip_cbs,whole_cbs[int(len(ip_cbs)):])
    
    # select parts of chA and neighboring subunits worth motif scoring interface
    inc_self = ip_reshape[[ind for ind,x in enumerate(np.any(axa<dist_threshold,axis=1)) if x]]
    inc_self = inc_self.reshape(len(inc_self)*5,4)
    inc_other_inds = [ind for ind,x in enumerate(np.any(axa<dist_threshold,axis=0)) if x]
    inc_other = whole_reshape[int(len(ip_cbs)):][inc_other_inds]
    inc_other = inc_other.reshape(len(inc_other)*5,4)
    
    hits, froms, tos, t_t = motif_stuff2.motif_score_npose_asym( inc_self, inc_other )
    

    return hits
    
    

def score_build(input_pose, input_ss=None):
    
    build_len = (len(input_pose)/5) - (len(stub)/5)

    return build_len, build_len / 1000



def test_builder(num_runs = 80000001):
    ct = 0
    while ct < num_runs:

#         if ct % 1000 == 0 and len(g.builds) > 0:
#             series = [ind for ind,_ in enumerate(g.builds)]

#             avg_lengths = []
#             best_lengths = []
#             avg_bts = []
#             avg_series = []
            
#             avg_overall = []
#             best_overall = []
            
#             if len(g.builds) < 10000:
#                 bin_size = 50
#             elif len(g.builds) < 25000:
#                 bin_size = 250
#             elif len(g.builds) < 50000:
#                 bin_size = 500
#             elif len(g.builds) < 100000:
#                 bin_size = 1000
#             else:
#                 bin_size = 5000

#             for i in series:
#                 if i % bin_size == 0 and i != 0:
#                     avg_series.append(i)
#                     avg_lengths.append(np.mean([len(x)/5 for x in g.builds[i-bin_size:i]]))
#                     best_lengths.append(max([len(x)/5 for x in g.builds[i-bin_size:i]]))
#                     avg_bts.append(np.mean(g.build_times[i-bin_size:i]))
                    
                    
#                     try:
#                         avg_overall.append(np.mean(g.overall_scores[i-bin_size:i]))
#                         best_overall.append(max(g.overall_scores[i-bin_size:i]))
#                     except:
#                         pass

#             print('avg lengths, best lengths')
#             sns.scatterplot(x = avg_series, y = avg_lengths)
#             sns.scatterplot(x = avg_series, y = best_lengths)
#             plt.show()
#             #plt.savefig('avglengths_bestlengths.png')
#             #plt.clf()
#             print('avg build times')
#             sns.scatterplot(x = avg_series, y = avg_bts)
#             plt.show()
#             #plt.savefig('avg_build_times.png')
#             #plt.clf()
            
#             try:
#                 print('avg overall')#, best pc scn')
#                 sns.scatterplot(x = avg_series, y = avg_overall)
#                 plt.show()
#                 #plt.savefig('avg_overall.png')
#                 #plt.clf()
#                 print('best overall')
#                 sns.scatterplot(x = avg_series, y = best_overall)
#                 plt.show()
#                 #plt.savefig('best_overall.png')
#                 #plt.clf()
#             except:
#                 pass
            
#             print('#############################################################')
            
        
        g.test_build()
        
        ct += 1

class tree_builder():
    def __init__(self, stub, wnum_hash):
        
        # starting point, wnum hash, build lists
        self.stub = stub
        self.wnum_hash = wnum_hash
        self.builds = []
        self.build_paths = []
        self.build_sstructs = []
        
        # score lists
        self.build_lens = []
        self.overall_scores = []
        self.score_lists = [self.build_lens,self.overall_scores]
        
        # general build params
        self.max_rad = outer_rad
        self.max_helix_length = 25
        self.min_helix_length = 7
        self.max_ext_length = self.max_helix_length - 4 # set by max helix length
        self.min_ext_length = self.min_helix_length - 4 # set by min helix length
        self.max_helices = 99999
        self.min_helices = 3
        self.max_residues = 99999
        self.min_score_residues = 0 # score at this length - higher is faster
        self.clash_threshold = 2.85 # 2.75 best? or too close
        self.diam_threshold = 99999 # plateaus build lengths
        
        # weight params
        self.base_wf = 100 # upweight factor for satisfying constraints (unscored)
        self.starting_wf = 10 # starting weights (changes how quickly choices become dominant) - don't change
        self.ext_multiplier = round((len(binned_loops)*2) / (self.max_ext_length+1-self.min_ext_length))
        self.ext_start_wf = self.ext_multiplier * self.starting_wf # reweight to match loop total
        self.zero_wf_initial = round(0.1*len(binned_loops)*2) # start at 10%
        self.zero_wf_scalar = self.zero_wf_initial / 2 # scales with distance down tree past minimum helices
        self.upweight_cutoff = 0.4 # how dominant choice can be relative to all other choices
        self.score_uw_scalar = 5000 # take score weight (between 0 and 1) and multiply for upweighting
        # score/constraint upweight is relative to starting_wf * len(binned_loops)*2 = sum initial wts
        
        # try/attempt params
        self.try_initial = 4 # starting try limit
        self.try_scalar = 3 # scales with distance down tree

        # choices - C ext, C loop, N ext, N loop, zero choice
        #self.all_choices = ['C_e_'+str(x) for x in range(self.min_ext_length,self.max_ext_length+1)] + ['C_l_'+str(x) for x in binned_loops] + ['N_e_'+str(x) for x in range(self.min_ext_length,self.max_ext_length+1)] + ['N_l_'+str(x) for x in binned_loops] + ['C_e_0'] # call zero choice a C extension of 0
        
        # choices - N ext, N loop, zero choice
        self.all_choices = ['N_e_'+str(x) for x in range(self.min_ext_length,self.max_ext_length+1)] + ['N_l_'+str(x) for x in binned_loops]
        
        # mapping of choices to weight indices for upweighting
        self.choice_wt_map = {}
        for cnd,chce in enumerate(self.all_choices):
            self.choice_wt_map[chce] = cnd

        # choice options and initial weights depending on previous choice (or start) - zero choice default 0
        self.initial_weights = {}
        #self.initial_weights['start'] = [self.ext_start_wf for x in range(self.min_ext_length,self.max_ext_length+1)] + [0 for _ in binned_loops] + [0 for _ in range(self.min_ext_length,self.max_ext_length+1)] + [0 for _ in binned_loops] + [0]
        #self.initial_weights['C_e'] = [0 for _ in range(self.min_ext_length,self.max_ext_length+1)] + [self.starting_wf for x in binned_loops] + [0 for _ in range(self.min_ext_length,self.max_ext_length+1)] + [self.starting_wf for x in binned_loops] + [0]
        #self.initial_weights['C_l'] = [self.ext_start_wf for x in range(self.min_ext_length,self.max_ext_length+1)] + [0 for _ in binned_loops] + [0 for _ in range(self.min_ext_length,self.max_ext_length+1)] + [0 for _ in binned_loops] + [0]
        #self.initial_weights['N_e'] = [0 for x in range(self.min_ext_length,self.max_ext_length+1)] + [self.starting_wf for _ in binned_loops] + [0 for x in range(self.min_ext_length,self.max_ext_length+1)] + [self.starting_wf for _ in binned_loops] + [0]
        #self.initial_weights['N_l'] = [0 for _ in range(self.min_ext_length,self.max_ext_length+1)] + [0 for _ in binned_loops] + [self.ext_start_wf for _ in range(self.min_ext_length,self.max_ext_length+1)] + [0 for _ in binned_loops] + [0]
        
        self.initial_weights['start'] = [self.ext_start_wf for x in range(self.min_ext_length,self.max_ext_length+1)] + [self.ext_start_wf for _ in binned_loops]
        self.initial_weights['C_e'] = []#[0 for _ in range(self.min_ext_length,self.max_ext_length+1)] + [self.starting_wf for x in binned_loops] + [0 for _ in range(self.min_ext_length,self.max_ext_length+1)] + [self.starting_wf for x in binned_loops] + [0]
        #self.initial_weights['C_l'] = [self.ext_start_wf for x in range(self.min_ext_length,self.max_ext_length+1)] + [0 for _ in binned_loops] + [0 for _ in range(self.min_ext_length,self.max_ext_length+1)] + [0 for _ in binned_loops] + [0]
        self.initial_weights['N_e'] = [0 for x in range(self.min_ext_length,self.max_ext_length+1)] + [self.starting_wf for _ in binned_loops]
        self.initial_weights['N_l'] = [self.ext_start_wf for _ in range(self.min_ext_length,self.max_ext_length+1)] + [0 for _ in binned_loops] 
        
        
        # set up build graph with first choice options, weights
        self.build_graph = [{},self.initial_weights['start'].copy()]
        
        # debug
        self.build_times = []

    
    def upweight(self, path, factor):
        
        for path_index,choice_name in enumerate(path):

            if (path_index+1)%2 == 0:
                
                # get appropriate index for upweighting
                wt_ind = self.choice_wt_map[choice_name]
                
                # access build path weights
                t1 = self.call_graph_path(path[:path_index-1])[1] 
                
                # number of choices down the tree, where first choice = 0
                num_choices = int((path_index-1)/2)
                
                # calc upweight cutoff based on position in path
                uw_cutoff = min( (self.upweight_cutoff + 0.025*num_choices), 0.6 )
                
                if (t1[int(wt_ind)]+int(factor)) < (sum(t1)*uw_cutoff):
                    t1[int(wt_ind)] += int(factor)
                else:
                    t1[int(wt_ind)] = int(sum(t1)*uw_cutoff)
                    
    
    def call_graph_path(self, path):
        
        return reduce(operator.getitem, path, self.build_graph)
    
    
    def test_build(self): # v4
        
        # start time, initialize single build
        start = time.time()
        build = self.stub.copy()
        build_path = []
        current_length = round(len(build)/5)
        current_helices = 0
        build_ss = 'L'*current_length # just call toroid loop to only worst helix score new parts
        prev_choice = []
        prev_choice_type = 'e'
        
        # while build does not exceed residue and helix limits
        while current_length < self.max_residues and current_helices < self.max_helices:

            # initialize no valid choice, 0 attempts
            valid_choice = False
            tries = 0
            
            # while valid choice not found
            while not valid_choice:
                
                # pick a choice, align it, keep track of end (N or C) and choice type (e or l)
                wts = self.call_graph_path(build_path)[1]
                choice_key = random.choices(self.all_choices, weights=wts, k=1)[0]
                choice_end, choice_type, choice_num = choice_key.split('_')
                
                if choice_type == 'e':
                    if len(build_path) == 0: # stub must be 5 residues, so reduce extension by 1
                        choice = extend_helix(build, choice_end, int(choice_num)-1)
                    else:
                        choice = extend_helix(build, choice_end, int(choice_num))
                    cc_build = build
                else:
                    loop_index = int(random.random()*len(binned_loops[int(choice_num)]))
                    rand_loop = binned_loops[int(choice_num)][loop_index]
                    if choice_end == 'C':
                        cc_build = build[:-4*5]
                    else:
                        cc_build = build[4*5:]
                    choice = align_loop(cc_build, choice_end, rand_loop)
                
                # check length, check hash, check clash - accept if all pass
                if current_length + round(len(choice)/5) <= self.max_residues:
                    if check_hash(choice, self.wnum_hash):
                        
                        # add icos op for cages, check addition vs. all
                        if choice_end == 'C': # add to C
                            # use different xform set to put chA at end, so that slice point still works
                            temp_build = c_op(np.append(cc_build,choice,0),6)
                            
                            slice_pt = -len(choice)
                            
                            if check_clash(temp_build[:slice_pt],choice_end,temp_build[slice_pt:],clash_threshold=self.clash_threshold,diameter_threshold=self.diam_threshold):
                            
                                rad_limit = False
                                for atom in choice:
                                    atom_rad = sqrt(atom[0]**2+atom[1]**2)
                                    max_z,min_z = get_zlims(atom_rad)
                                    if atom_rad  > self.max_rad or atom[2] > max_z or atom[2] < min_z:
                                        rad_limit = True
                                    
                                if not rad_limit:
                                    valid_choice = True
                        
                        else: # add to N
                            temp_build = c_op(np.append(choice,cc_build,0),6)
                            
                            slice_pt = len(choice)
                            
                            if check_clash(temp_build[slice_pt:],choice_end,temp_build[:slice_pt],clash_threshold=self.clash_threshold,diameter_threshold=self.diam_threshold):
                                
                                rad_limit = False
                                for atom in choice:
                                    atom_rad = sqrt(atom[0]**2+atom[1]**2)
                                    max_z,min_z = get_zlims(atom_rad)
                                    if atom_rad > self.max_rad or atom[2] > max_z or atom[2] < min_z:
                                        rad_limit = True
                                        
                                if not rad_limit:
                                    valid_choice = True
                
                # update attempts
                tries += 1
                
                # if tries exceed limit and no choice found, or zero choice picked - end condition
                if ((tries >= (self.try_scalar*current_helices+self.try_initial)) and (not valid_choice)) or (choice_key == 'C_e_0'):
                    
                    # add zero choice to build path if picked (to allow it to be upweighted)
                    if choice_key == 'C_e_0':
                        build_path.append(0)
                        build_path.append(choice_key)
                        
                    # remove loop if it was last choice, update build path, length, secondary struct, helices
                    if prev_choice_type == 'l' and len(prev_choice) > 0:
                        # remove loop, re-extend by loop cap helix residues
                        if prev_choice_end == 'C': 
                            cc_build = build[:-len(prev_choice)]
                            build = np.append(cc_build,extend_helix(cc_build,'C',4),0)
                            build_ss = build_ss[:-round(len(prev_choice)/5)] + 'H'*4
                        else:
                            cc_build = build[len(prev_choice):]
                            build = np.append(extend_helix(cc_build,'N',4),cc_build,0)
                            build_ss = 'H'*4 + build_ss[round(len(prev_choice)/5):]
                        build_path = build_path[:-2]
                        current_length = round(len(build)/5)
                        current_helices = ((len(build_path)/2)+1)/2
                
                    # if above score length threshold
                    if current_length >= self.min_score_residues and current_helices >= self.min_helices:

                        # score build, upweight accordingly
                        scores = score_build(build, input_ss=build_ss)
                        if scores[-1] > 0:
                            self.upweight(build_path,scores[-1]*self.score_uw_scalar)
                        for s_ind,score in enumerate(scores):
                            self.score_lists[s_ind].append(score)

                    
                    else: # append zeroes otherwise
                        for s_list in self.score_lists:
                            s_list.append(0)
                    
                    # record build, details, stop time
                    self.builds.append(build)
                    self.build_paths.append(build_path)
                    self.build_sstructs.append(build_ss)
                    end = time.time()
                    self.build_times.append(end-start)
                    
                    return build
            
            # found valid choice
            # update build path, build, length, helices, secondary struct
            build_path.append(0)
            build_path.append(choice_key)
            if choice_end == 'C': # add to C
                build = np.append(cc_build,choice,0)
                if choice_type == 'e':
                    build_ss = build_ss + round(len(choice)/5)*'H'
                else:
                    build_ss = build_ss + (round(len(choice)/5)-8)*'L' + 'H'*4
            else: # add to N
                build = np.append(choice,cc_build,0)
                if choice_type == 'e':
                    build_ss = round(len(choice)/5)*'H' + build_ss
                else:
                    build_ss = 'H'*4 + (round(len(choice)/5)-8)*'L' + build_ss
            current_length = round(len(build)/5)
            current_helices = ((len(build_path)/2)+1)/2
            
            # keep track of previous choice, end, type to trim loops
            prev_choice, prev_choice_end, prev_choice_type = choice, choice_end, choice_type
            
            # if new branch point in tree:
            if choice_key not in self.call_graph_path(build_path[:-2])[0]:
            
                # find new choice list, weights according to previous choice
                new_weights = self.initial_weights[choice_key[:3]].copy()
                
                # add zero choice if path is long enough (at least 3 helices) and last addition was helix
                if current_helices >= self.min_helices and choice_type == 'e':
                    # weight according to build path length (increase bias to terminate if longer)
                    new_weights[-1] == self.zero_wf_initial + max(0,self.zero_wf_scalar*(current_helices-self.min_helices))
                
                # create new branch point in tree with new choices/weights
                t1 = self.call_graph_path(build_path[:-2])[0][choice_key] = [{},new_weights]
                
            # upweight for satisfying constraints
            self.upweight(build_path,self.base_wf)
            
        # reached length or helix limit
        # remove loop if it was last choice, update build path, length, secondary struct, helices
        if prev_choice_type == 'l' and len(prev_choice) > 0:
            # remove loop, re-extend by loop cap helix residues
            if prev_choice_end == 'C': 
                cc_build = build[:-len(prev_choice)]
                build = np.append(cc_build,extend_helix(cc_build,'C',4),0)
                build_ss = build_ss[:-round(len(prev_choice)/5)] + 'H'*4
            else:
                cc_build = build[len(prev_choice):]
                build = np.append(extend_helix(cc_build,'N',4),cc_build,0)
                build_ss = 'H'*4 + build_ss[round(len(prev_choice)/5):]
            build_path = build_path[:-2]
            current_length = round(len(build)/5)
            current_helices = ((len(build_path)/2)+1)/2
        
        # if above score length threshold
        if current_length >= self.min_score_residues and current_helices >= self.min_helices:

            # score build, upweight accordingly
            scores = score_build(build, input_ss=build_ss)
            if scores[-1] > 0:
                self.upweight(build_path,scores[-1]*self.score_uw_scalar)
            for s_ind,score in enumerate(scores):
                self.score_lists[s_ind].append(score)


        else: # append zeroes otherwise
            for s_list in self.score_lists:
                s_list.append(0)

        # record build, details, stop time
        self.builds.append(build)
        self.build_paths.append(build_path)
        self.build_sstructs.append(build_ss)
        end = time.time()
        self.build_times.append(end-start)

        return build
        

# # make loop bins
# binned_loops = kClusBin(clusts1=1,clusts2=32,clusts3=16)

# # save loop bins, for easier use later
# save_obj(binned_loops,'/home/ilutz/BINDERS_RL/2_RL/binned_loops_no0')

# loading binned loops from .pkl file
binned_loops = load_obj('/home/ilutz/BINDERS_RL/2_RL/binned_loops_no0')


# build_mesh = '../build_vol.obj'


# # # wnum_hash is slow to compute -- can save it as .pkl and just load to save time
# # # buffer distance prevents building directly on surface -- set to 0 currently
# if len(glob.glob('../wnum_hash*')) == 0: # only bother making it if it's not there yet
#     wnum_hash = voxelize_mesh(build_mesh,0) # buff dist = 0
#     save_obj(wnum_hash,'../wnum_hash')
# else:
#     # loading wnum_hash from .pkl file
#     # for sequence buffer of 0:
#     wnum_hash = load_obj('../wnum_hash')
    
# # for sequence buffer of 1:
# #wnum_hash = load_obj('/home/ilutz/RL_RIF/apoe_test_auto/apoe_wnum_hash2_buff1')


def dump_holigomer(npose, chains, out_name):
    
    # MODIFIED FOR PORE CLOSURE WITH INNER
    
    slice_len = len(npose)/chains
    sub_name = out_name.split('/')[-1]
    
    dump_file = []
    chain_ops = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','0','1','2','3','4','5','6','7','8','9']
    for i in range(chains):
        ch_name = f'scratch/scratch_{sub_name}_{i}.pdb'
        nu.dump_npdb(npose[int(i*slice_len):int((i+1)*slice_len)],ch_name)
        
        # read files, append to new pdb with chains labelled
        with open(ch_name,'r') as ch_file:        
            for line in ch_file:        
                dump_file.append(line[:21]+chain_ops[i]+line[22:])
            dump_file.append('TER\n')
    
    # ADD INNER AS SEPARATE CHAINS
    slice_len = len(inner)/chains
    for i in range(chains):
        ch_name = f'scratch/scratch_{sub_name}_{i+chains}.pdb'
        nu.dump_npdb(inner[int(i*slice_len):int((i+1)*slice_len)],ch_name)
        
        # read files, append to new pdb with chains labelled
        with open(ch_name,'r') as ch_file:        
            for line in ch_file:        
                dump_file.append(line[:21]+chain_ops[i+chains]+line[22:])
            dump_file.append('TER\n')
    
    with open(f'{out_name}.pdb', 'w') as f:
        for item in dump_file:
            f.write(item)
    del_paths = glob.glob(f'scratch/scratch_{sub_name}_*.pdb')
    for dp in del_paths:
        os.remove(dp)



def euler_to_R(phi,theta,psi): # in radians
    R_x = np.array([[1,         0,                  0     ],
                    [0,         np.cos(phi), -np.sin(phi) ],
                    [0,         np.sin(phi), np.cos(phi)  ]
                    ])

    R_y = np.array([[np.cos(theta),    0,      np.sin(theta)  ],
                    [0,                1,      0              ],
                    [-np.sin(theta),   0,      np.cos(theta)  ]
                    ])

    R_z = np.array([[np.cos(psi),    -np.sin(psi),    0],
                    [np.sin(psi),    np.cos(psi),     0],
                    [0,              0,               1]
                    ])

    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R




for _ in range(1):
    
    g = tree_builder(stub, wnum_hash)
    
    test_builder(num_runs = 10000)
    
    max_dump_per_trial = 5
    dump_ct = 0
    best_builds = {}
    for ind,i in enumerate(g.builds):
        if g.overall_scores[ind] > 0.005:
            best_builds[g.overall_scores[ind]] = (i, g.build_sstructs[ind])

    sorted_best_builds = [x[1] for x in sorted(best_builds.items(), key = lambda kv: kv[0])]
    sorted_best_builds.reverse()

    for bind,pre_bd in enumerate(sorted_best_builds):
        if dump_ct < max_dump_per_trial:
            
            bd = pre_bd[0]
            build_len, score_weight = score_build(bd)
            
            
            bd_r = bd.reshape(int(len(bd)/5),5,4)
            bd_terminus = bd_r[0][nu.CA][:3]

            term_dist = min([np.linalg.norm(bd_terminus-x) for x in inner_terminii])
            
            if term_dist < 20:

                dump_holigomer(c_op(bd[:-int(len(inner)/6)],6),6,f'outputs/{base_name}_rl_{str(term_dist)[:6]}_{int(build_len)}')
                dump_ct += 1
            

