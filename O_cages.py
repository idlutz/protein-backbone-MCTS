#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

import sys

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


# In[90]:


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

icos_matrices = load_obj('xforms/O_matrices')
icosa_xforms = [np.append(x,np.array([[0,0,0,1]])).reshape(4,4) for x in icos_matrices]

zero_ih = nu.npose_from_file('/opt/conda/envs/env/rl_tree_source/zero_ih_long.pdb')

def icosahedral_op(pt_set,icosa_xform_set=icosa_xforms): # icosahedral symmetry operations

    i_out = []
    for xform in icosa_xform_set:
        i_out.append(nu.xform_npose(xform, pt_set))
    
    return np.concatenate(tuple(i_out))


rr = np.load('/opt/conda/envs/env/rl_tree_source/all_loops_bCov.npz', allow_pickle=True)
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
    
    c_vdw = 1.7
    dists = [np.linalg.norm(pt[:3]) for pt in bd]
    rads = sorted(dists)
    
    # at least one terminus exposed condition
    if (max(rads.index(dists[0]),rads.index(dists[-1])) / len(dists)) > 0.45:
    
        nds = [x for x in range(len(rads))]

        inds = np.array(nds*len(nds)).reshape(len(nds),len(nds))
        jnds = np.transpose(inds)
        atoms_vols = 24 * (jnds-inds+1) * (4/3)*pi*c_vdw**3 # change num for symmetry

        its = np.array(rads*len(rads)).reshape(len(rads),len(rads))
        jts = np.transpose(its)
        shell_vols = (4/3)*pi * ( (jts+c_vdw)**3 - (its-c_vdw)**3 )

        mask = jnds > inds

        pore_vals = (atoms_vols / shell_vols)*mask

        return np.max(pore_vals)
    
    else:
        return 0


def interchain_motif_score(input_pose,icosa_xform_set=icosa_xforms):
    
    # speed up -- take all by all dists of chA vs. subunit set
    # mask/remove everything above a distance threshold
    # motif score only what is left -- just potential interfaces
    # mask all self interactions
    
    # in motif score code, this appears to be 12 -- TEST
    dist_threshold = 6
    
    whole = icosahedral_op(input_pose,icosa_xform_set)
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
    
    # only return hits if more than one interface found
    if len(set([int(inc_other_inds[x]/len(ip_cbs)) for x in tos])) > 2:
        return hits
    
    # results slightly different than old ic_ms, but may be better
    # doesn't add integer error from subtracting base -- more accurate
    # may be more restrictive on counting number of interfaces
    else:
        return 0
    
    

def score_build(input_pose, input_ss=None, icosa_xform_set=icosa_xforms):

    pose = input_pose.reshape(int(len(input_pose)/5),5,4)

    ca_cb = pose[:,1:3,:3]
    care_mask = np.ones(len(pose),dtype=int)
    
    neighs = get_avg_sc_neighbors( ca_cb, care_mask )
    
    #surface = neighs < 2
    is_core_boundary = neighs > 2
    is_core = neighs > 5.2
    
    # to score core/boundary
    score_mask = is_core_boundary
    care_mask = is_core_boundary
    
    # percent of 9mers with hit in core/boundary (brian says >80%, maybe even >90%)
    hits, froms, tos, t_t = motif_stuff2.motif_score_npose( input_pose, score_mask, care_mask )
    no_hits = 0
    
    for i in range(len(care_mask)-8):
        hits = False
        for j in range(9):
            if i+j in tos or i+j in froms:
                hits = True
        if hits:
            no_hits += 1
    
    pc_scn = is_core.mean()
    motif_score = no_hits / (len(care_mask)-8)
    pore_score = fast_pore_score(input_pose)
    ic_ms = interchain_motif_score(input_pose,icosa_xform_set=icosa_xform_set)

    a_pc = 3
    b_pc = 0.25
    m_pc = 100
    score_pc = m_pc/(1+exp(-a_pc*10*(pc_scn - b_pc)))
    
    a_mot = 3
    b_mot = 0.9
    m_mot = 10
    score_mot = m_mot/(1+exp(-a_mot*10*(motif_score - b_mot))) + 1
    
    a_pore = 1.8
    b_pore = 0.61
    m_pore = 100
    score_pore = m_pore/(1+exp(-a_pore*10*(pore_score - b_pore)))
    
    a_icms = 0.03
    b_icms = 20
    m_icms = 2
    score_icms = m_icms/(1+exp(-a_icms*10*(ic_ms - b_icms)))
    
    score_wt_pre = score_pc*score_mot*score_pore*score_icms*0.05 # additional reweight -- may actually be key

    if input_ss:
        worst_helix = 999

        prev = 'Y'
        first = 0
        for ind,i in enumerate(input_ss):
            if i == 'H' and prev == 'L':
                first = ind
            if i == 'L' and prev == 'H':
                second = ind-1
                hel_neighs = neighs[first:second+1].mean()
                if hel_neighs < worst_helix:
                    worst_helix = hel_neighs
            elif i == 'H' and ind == len(input_ss)-1:
                second = ind
                hel_neighs = neighs[first:second+1].mean()
                if hel_neighs < worst_helix:
                    worst_helix = hel_neighs
            prev = i

        a_whel = 0.5
        b_whel = 2.1
        m_whel = 1.1
        whel_penalty = m_whel/(1+exp(-a_whel*10*(worst_helix - b_whel)))

        score_wt_pre = score_wt_pre * whel_penalty
    else:
        worst_helix = 0

    if score_wt_pre > 0.25:
        a_final,b_final,m_final = .003, 200, 1
        score_wt = (m_final/(1+exp(-a_final*10*(score_wt_pre - b_final))))
    else:
        score_wt = 0

    return pc_scn, motif_score, pore_score, ic_ms, worst_helix, score_wt



def test_builder(num_runs = 80000001):
    ct = 0
    while ct < num_runs:

#         if ct % 250 == 0 and len(g.builds) > 0:
#             series = [ind for ind,_ in enumerate(g.builds)]

#             avg_lengths = []
#             best_lengths = []
#             avg_bts = []
#             avg_series = []
            
#             avg_pc_scn = []
#             best_pc_scn = []
            
#             avg_mots = []
#             best_mots = []
            
#             avg_pores = []
#             best_pores = []
            
#             avg_icms = []
#             best_icms = []
            
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
                    
#                     avg_pc_scn.append(np.mean(g.pc_scns[i-bin_size:i]))
#                     best_pc_scn.append(max(g.pc_scns[i-bin_size:i]))
                    
#                     avg_mots.append(np.mean(g.motif_scores[i-bin_size:i]))
#                     best_mots.append(max(g.motif_scores[i-bin_size:i]))
                    
#                     avg_pores.append(np.mean(g.pore_scores[i-bin_size:i]))
#                     best_pores.append(max(g.pore_scores[i-bin_size:i]))
                    
#                     avg_icms.append(np.mean(g.icm_scores[i-bin_size:i]))
#                     best_icms.append(max(g.icm_scores[i-bin_size:i]))
                    
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
#             print('avg pc scn')#, best pc scn')
#             sns.scatterplot(x = avg_series, y = avg_pc_scn)
#             plt.show()
#             #plt.savefig('avg_pc_scn.png')
#             #plt.clf()
#             print('best pc scn')
#             sns.scatterplot(x = avg_series, y = best_pc_scn)
#             plt.show()
#             #plt.savefig('best_pc_scn.png')
#             #plt.clf()
#             print('avg motif score')#, best motif score')
#             sns.scatterplot(x = avg_series, y = avg_mots)
#             plt.show()
#             #sns.scatterplot(x = avg_series, y = best_mots)
#             #plt.savefig('avg_motif_score.png')
#             #plt.clf()
#             print('avg pore score')
#             sns.scatterplot(x = avg_series, y = avg_pores)
#             plt.show()
#             #plt.savefig('avg_pore_score.png')
#             #plt.clf()
#             print('best pore score')
#             sns.scatterplot(x = avg_series, y = best_pores)
#             plt.show()
#             #plt.savefig('best_pore_score.png')
#             #plt.clf()
#             print('avg ic ms')
#             sns.scatterplot(x = avg_series, y = avg_icms)
#             plt.show()
#             #plt.savefig('avg_icms.png')
#             #plt.clf()
#             print('best ic ms')
#             sns.scatterplot(x = avg_series, y = best_icms)
#             plt.show()
#             #plt.savefig('best_icms.png')
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
        
        # sublist of xforms for given start
        self.xforms = icosa_xforms
        self.clashC_xforms = icosa_xforms
        
        # score lists
        self.pc_scns = []
        self.motif_scores = []
        self.pore_scores = []
        self.icm_scores = []
        self.worst_helices = []
        self.overall_scores = []
        self.score_lists = [self.pc_scns,self.motif_scores,self.pore_scores,self.icm_scores,self.worst_helices,self.overall_scores]
        
        # general build params
        self.max_rad = 65
        self.max_helix_length = 22
        self.min_helix_length = 9
        self.max_ext_length = self.max_helix_length - 4 # set by max helix length
        self.min_ext_length = self.min_helix_length - 4 # set by min helix length
        self.max_helices = 7
        self.min_helices = 3
        self.max_residues = 80 # max for MS barcode chip order
        self.min_score_residues = int(len(stub)/5) + 50 # score at this length - higher is faster
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
        self.all_choices = ['C_e_'+str(x) for x in range(self.min_ext_length,self.max_ext_length+1)] + ['C_l_'+str(x) for x in binned_loops] + ['N_e_'+str(x) for x in range(self.min_ext_length,self.max_ext_length+1)] + ['N_l_'+str(x) for x in binned_loops] + ['C_e_0'] # call zero choice a C extension of 0
        
        # mapping of choices to weight indices for upweighting
        self.choice_wt_map = {}
        for cnd,chce in enumerate(self.all_choices):
            self.choice_wt_map[chce] = cnd

        # choice options and initial weights depending on previous choice (or start) - zero choice default 0
        self.initial_weights = {}
        self.initial_weights['start'] = [self.ext_start_wf for x in range(self.min_ext_length,self.max_ext_length+1)] + [0 for _ in binned_loops] + [0 for _ in range(self.min_ext_length,self.max_ext_length+1)] + [0 for _ in binned_loops] + [0]
        self.initial_weights['C_e'] = [0 for _ in range(self.min_ext_length,self.max_ext_length+1)] + [self.starting_wf for x in binned_loops] + [0 for _ in range(self.min_ext_length,self.max_ext_length+1)] + [self.starting_wf for x in binned_loops] + [0]
        self.initial_weights['C_l'] = [self.ext_start_wf for x in range(self.min_ext_length,self.max_ext_length+1)] + [0 for _ in binned_loops] + [0 for _ in range(self.min_ext_length,self.max_ext_length+1)] + [0 for _ in binned_loops] + [0]
        self.initial_weights['N_e'] = [0 for x in range(self.min_ext_length,self.max_ext_length+1)] + [self.starting_wf for _ in binned_loops] + [0 for x in range(self.min_ext_length,self.max_ext_length+1)] + [self.starting_wf for _ in binned_loops] + [0]
        self.initial_weights['N_l'] = [0 for _ in range(self.min_ext_length,self.max_ext_length+1)] + [0 for _ in binned_loops] + [self.ext_start_wf for _ in range(self.min_ext_length,self.max_ext_length+1)] + [0 for _ in binned_loops] + [0]
        
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
        build_ss = 'H'*current_length
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
                            temp_build = icosahedral_op(np.append(cc_build,choice,0),icosa_xform_set=self.clashC_xforms)
                            
                            slice_pt = -len(choice)
                            
                            if check_clash(temp_build[:slice_pt],choice_end,temp_build[slice_pt:],clash_threshold=self.clash_threshold,diameter_threshold=self.diam_threshold):

                                rad_limit = False
                                for atom in choice:
                                    if sqrt(atom[0]**2+atom[1]**2+atom[2]**2) > self.max_rad:
                                        rad_limit = True
                                if not rad_limit:
                                    valid_choice = True
                        
                        else: # add to N
                            temp_build = icosahedral_op(np.append(choice,cc_build,0),icosa_xform_set=self.xforms)
                            
                            slice_pt = len(choice)
                            
                            if check_clash(temp_build[slice_pt:],choice_end,temp_build[:slice_pt],clash_threshold=self.clash_threshold,diameter_threshold=self.diam_threshold):

                                rad_limit = False
                                for atom in choice:
                                    if sqrt(atom[0]**2+atom[1]**2+atom[2]**2) > self.max_rad:
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
                        scores = score_build(build, input_ss=build_ss, icosa_xform_set=self.xforms)
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
            scores = score_build(build, input_ss=build_ss, icosa_xform_set=self.xforms)
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
# save_obj(binned_loops,'binned_loops_no0')

# loading binned loops from .pkl file
binned_loops = load_obj('/opt/conda/envs/env/rl_tree_source/binned_loops_no0')

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
    

def dump_holigomer(npose, chains, out_name):
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


# In[4]:


def closest_xforms(input_pose):
    
    n_closest = 12 # 31 works empirically tested, compared to full xform set to see conflict
    # this number could realistically be lower, given extended monomers probably fail on other metrics
    # 12 seems to work too -- saw one clash violation at 11, haven't fully tested 12
    
        
    whole = icosahedral_op(input_pose)
    il = int(len(input_pose))
    centroid_1 = np.mean(input_pose,0)[:3]
    
    xform_dists = {}

    for i in range(1,24): # change num for symmetry
        centroid_i = np.mean(whole[il*i:il*(i+1)],0)[:3]
        dist = np.linalg.norm(centroid_1 - centroid_i)
        xform_dists[i] = dist
        
    in_order = sorted(xform_dists.items(), key=lambda x:x[1])

    return np.array([icosa_xforms[x] for x in [0]+[x[0] for x in in_order[:n_closest]]])


# In[5]:


def random_init_build(runs):
    
    final_run = False
    
    while not final_run:
        
        # initialize with random origin, then pick random start
        g.__init__(stub + [0,30,0,0], wnum_hash)
        start = False  

        while not start:
            
            # random translation within radius:
            trans = [999,999,999]
            while np.linalg.norm(trans) > stub_radius:
                trans = [random.random()*stub_radius,
                         random.random()*stub_radius,
                         random.random()*stub_radius]
            
            # add a random rotation
            rot = euler_to_R(random.random()*2*math.pi-math.pi,
                             random.random()*2*math.pi-math.pi,
                             random.random()*2*math.pi-math.pi)
            
            # make xform from translation, rotation
            pre = np.concatenate((rot,np.array([[0,0,0]])), axis=0)
            rand_xform = np.concatenate((pre,np.array([[trans[0]],[trans[1]],[trans[2]],[1]])), axis=1)
            
            # generate potential start with xform
            pot_start = nu.xform_npose(rand_xform, ori_1res)

            if check_hash(pot_start,g.wnum_hash):

                # add reasonable short extension to check too
                ext = extend_helix(pot_start, 'C', 5) # reasonable?
                if check_hash(ext, g.wnum_hash):
            
                    trial_build = icosahedral_op(np.concatenate((pot_start,ext)))
                        
                    slice_pt = -(len(ext)+len(pot_start))
                    if check_clash(trial_build[:slice_pt],'C',trial_build[slice_pt:],clash_threshold=g.clash_threshold,diameter_threshold=g.diam_threshold,seq_buff=0):

                        if g.max_rad:
                            radius_limit = False
                            for atom in ext:
                                if sqrt(atom[0]**2+atom[1]**2+atom[2]**2) > g.max_rad:
                                    radius_limit = True
                            if not radius_limit:
                                start = True
                        else:
                            start = True

        g.stub = pot_start
        g.xforms = closest_xforms(g.stub)
        g.clashC_xforms = np.append(g.xforms[1:],[g.xforms[0]],0)
        
        # test with x iterations to make sure builds are working
        test_iters = 500
        test_builder(num_runs = test_iters)

        # if builds are getting longer than x residues and scoring well enough, do full number of iterations
        if max(g.overall_scores) > 0.0001:
            
            final_run = True

            test_builder(num_runs = runs)

            max_dump_per_trial = 5
            dump_ct = 0
            best_builds = {}
            for ind,i in enumerate(g.builds):
                if g.overall_scores[ind] > 0.001:
                    best_builds[g.overall_scores[ind]] = (i, g.build_sstructs[ind])

            sorted_best_builds = [x[1] for x in sorted(best_builds.items(), key = lambda kv: kv[0])]
            sorted_best_builds.reverse()

            for bind,pre_bd in enumerate(sorted_best_builds):
                if dump_ct < max_dump_per_trial:

                    bd = pre_bd[0]
                    bd_ss = pre_bd[1]

                    pc_scn, motif_hits, pore_score, ic_ms, worst_hel, score_weight = score_build(bd,input_ss=bd_ss,icosa_xform_set=g.xforms)

                    if pc_scn >= 0.2 and motif_hits >= 0.9 and pore_score >= 0.3 and ic_ms > 10 and worst_hel > 2.0:
                        dump_holigomer(icosahedral_op(bd),24,f'outputs/build_{str(pc_scn)[:6]}_{str(motif_hits)[:4]}_{str(pore_score)[:6]}_{str(ic_ms)[:6]}_{str(worst_hel)[:4]}_{str(score_weight)[:8]}')
                        dump_ct += 1

tt = zero_ih.reshape(int(len(zero_ih)/5),5,4)
stub = tt[7:10].reshape(15,4)
ori_1res = tt[10:15].reshape(25,4)

stub_radius = 50
wnum_hash = []
g = tree_builder(stub + [0,30,0,0], wnum_hash)

# can also use a mesh to get this
maxs = [75,75,75]
mins = [0,0,0]



for _ in range(99999999):
    random_init_build(10000)

