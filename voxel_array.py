#!/usr/bin/env python

import os
import sys
import itertools
import numpy as np
import random

from numba import njit

# OOB gets clipped to the edges. Be careful to leave them at 0
class VoxelArray:

    def __init__(self, lbs, ubs, cbs, dtype="f8", arr=None):

        self.dim = len(lbs)
        self.lb = lbs
        self.ub = ubs
        self.cs = cbs

        if ( arr is None ):
            extents = self.floats_to_indices_no_clip(np.array([self.ub]))[0]
            extents += 1
            self.arr = np.zeros(extents, dtype=dtype)
        else:
            self.arr = arr

    def copy(self):
        vx = VoxelArray(self.lb, self.ub, self.cs, self.arr.dtype, self.arr.copy())
        return vx

    def save(self, fname):
        save_dict = {
            "lb":self.lb,
            "ub":self.ub,
            "cs":self.cs,
            "arr":self.arr
        }
        np.save(fname, save_dict)

    @classmethod
    def load(cls, fname):
        save_dict = np.load(fname, allow_pickle=True).item()
        lb = save_dict["lb"]
        ub = save_dict["ub"]
        cs = save_dict["cs"]
        arr = save_dict["arr"]

        return cls(lb, ub, cs, arr=arr)


    # only used in __init__ 
    def floats_to_indices_no_clip(self, pts):
        inds = np.zeros((len(pts), self.dim), dtype=np.int)
        for i in range(self.dim):
            inds[:,i] = ((pts[:,i] - self.lb[i] ) / self.cs[i])
        return inds

    def floats_to_indices(self, pts, out=None):
        if ( out is None ):
            out = np.zeros((len(pts), self.dim), dtype=np.int)

        return xform_vectors_w_out(pts, self.lb, self.cs, self.arr.shape, out)

    def indices_to_centers(self, inds ):
        return numba_indices_to_centers(inds, self.lb, self.cs)

    def all_indices(self):
        ranges = []
        for i in range(self.dim):
            ranges.append(list(range(self.arr.shape[i])))
        inds = np.array(list(itertools.product(*ranges)))
        return inds

    def all_centers(self):
        inds = self.all_indices()
        return self.indices_to_centers(inds)


    # One would usuallly type assert(voxel.oob_is_zero())
    def oob_is_zero(self):
        # This could certainly be made more efficient
        all_indices = self.all_indices()
        is_good = np.zeros(len(all_indices))
        for i in range(self.dim):
            is_good |= (all_indices[:,i] == 0) | (all_indices[:,i] == self.arr.shape[i]-1)

        good_indices = all_indices[is_good]
        return np.any(self.arr[good_indices])

    # This uses the centers as measurement
    def indices_within_x_of(self, _x, pt):
        low = pt - _x
        high = pt + _x

        # If you hit these, you are about to make a mistake
        assert( not np.any( low <= self.lb + self.cs))
        assert( not np.any( high >= self.ub - self.cs ) )

        bounds = self.floats_to_indices( np.array( [low, high] ) )

        ranges = []
        size = 1
        for i in range(self.dim):
            ranges.append(np.arange(bounds[0, i], bounds[1, i] + 1) )
            size *= (len(ranges[-1]))
        ranges = np.array(ranges)

        #in numba version, this whole bottom part is tested for loops

        # indices = np.array(itertools.product(*ranges))
        indices = np.array(np.meshgrid(*ranges)).T.reshape(-1, len(ranges))

        centers = self.indices_to_centers(indices)

        return indices[ np.sum(np.square(centers - pt), axis=-1) < _x*_x ]



    def dump_mask_true(self, fname, mask, resname="VOX", atname="VOXL", z=None, fraction=1 ):

        indices = np.array(list(np.where(mask))).T
        centers = self.indices_to_centers(indices)

        if ( self.dim == 2 ):
            centers_ = np.zeros((len(centers), 3), np.float)
            centers_[:,:2] = centers
            centers_[:,2] = z
            centers = centers_


        if ( fraction < 1 ):
            mask = np.random.random(len(indices)) < fraction
            # indices = indices[mask]
            centers = centers[mask]

        f = open(fname, "w")

        anum = 1
        rnum = 1

        for ind, xyz in enumerate(centers):

            f.write("%s%5i %4s %3s %s%4i    %8.3f%8.3f%8.3f%6.2f%6.2f %11s\n"%(
                "HETATM",
                anum,
                atname,
                resname,
                "A",
                rnum,
                xyz[0],xyz[1],xyz[2],
                1.0,
                1.0,
                "HB"
                ))

            anum += 1
            rnum += 1
            anum %= 100000
            rnum %= 10000

        f.close()



    def dump_grids_true(self, fname, func, resname="VOX", atname="VOXL", jitter=False, z=None):
        centers = self.all_centers()
        vals = self.arr[tuple(self.floats_to_indices(centers).T)]

        if ( self.dim == 2 ):
            centers_ = np.zeros((len(centers), 3), np.float)
            centers_[:,:2] = centers
            centers_[:,2] = z
            centers = centers_


        f = open(fname, "w")

        anum = 1
        rnum = 1

        for ind, xyz in enumerate(centers):
            if ( jitter ):
                xyz[0] += 0.01*2*(1 - 0.5*random.random())
                xyz[1] += 0.01*2*(1 - 0.5*random.random())
                xyz[2] += 0.01*2*(1 - 0.5*random.random())
            val = vals[ind]
            if (not func(val)):
                continue

            f.write("%s%5i %4s %3s %s%4i    %8.3f%8.3f%8.3f%6.2f%6.2f %11s\n"%(
                "HETATM",
                anum,
                atname,
                resname,
                "A",
                rnum,
                xyz[0],xyz[1],xyz[2],
                1.0,
                1.0,
                "HB"
                ))

            anum += 1
            rnum += 1
            anum %= 100000
            rnum %= 10000

        f.close()



    def clash_check(self, pts, max_clashes):
        assert(self.dim == 3)

        return numba_clash_check(pts, max_clashes, self.arr, self.lb, self.cs)

    def ray_trace(self, start, end, max_clashes, debug=False):
        assert(self.dim == 3)

        return numba_ray_trace(start, end, max_clashes, self.arr, self.lb, self.cs, debug)

    def ray_trace_many(self, starts, ends, max_clashes, debug=False):
        assert(self.dim == 3)

        return numba_ray_trace_many(starts, ends, max_clashes, self.arr, self.lb, self.cs, debug)

    def add_to_clashgrid(self, pts, atom_radius, store_val=True ):
        if ( isinstance( atom_radius, list ) ):
            assert(len(pts) == len(atom_radius))
            numba_make_clashgrid_var_atom_radius(pts, atom_radius, self.arr, self.lb, self.ub, self.cs, self.arr.shape, store_val)
        else:
            numba_make_clashgrid(pts, atom_radius, self.arr, self.lb, self.ub, self.cs, self.arr.shape, store_val)


    def add_to_sum_grid(self, pts, atom_radius, store_val=1 ):
        numba_make_sum_grid(pts, atom_radius, self.arr, self.lb, self.ub, self.cs, self.arr.shape, store_val)


    # fill the voxel array with ipt for all voxels closest to ipt.
    # initialize self to -1 and dist_grid to +100000
    def add_to_near_grid(self, pts, atom_radius, dist_grid, store_vals = None):
        assert((self.lb == dist_grid.lb).all())
        assert((self.ub == dist_grid.ub).all())
        assert((self.cs == dist_grid.cs).all())
        assert(self.arr.shape == dist_grid.arr.shape)
        if ( store_vals is None ):
            store_vals = np.arange(len(pts))
        numba_add_to_near_grid(pts, store_vals, atom_radius, self.arr, dist_grid.arr, self.lb, self.ub, self.cs, self.arr.shape)


    # fill voxels with -1 if below surface, 1 if above
    def do_surface_crawl(self, start, normal, direction, distance):
        return numba_do_surface_crawl(start, normal, direction, distance, self.arr, self.lb, self.ub, self.cs, self.arr.shape)

    def flood_fill(self, fill_val, overwrite_val):
        if ( self.dim == 2 ):
            return numba_flood_fill_2d(fill_val, overwrite_val, self.arr, self.lb, self.ub, self.cs, self.arr.shape )
        if ( self.dim == 3 ):
            return numba_flood_fill_3d(fill_val, overwrite_val, self.arr, self.lb, self.ub, self.cs, self.arr.shape )
        assert(False)

    def flood_fill_from_here(self, fill_val, overwrite_val, start_idx):
        return numba_flood_fill_3d_from_here(fill_val, overwrite_val, start_idx, self.arr, self.lb, self.ub, self.cs, self.arr.shape)


@njit(fastmath=True,cache=True)
def numba_seek_to_surface(pt, normal_step, up_down_steps, fail, arr, lb, ub, cs, shape):

    initial_pt = lookup_vec(pt, arr, lb, cs, shape)
    if ( initial_pt == 0 ):
        fail[0] = True
        return pt

    look_for = 1 if initial_pt == -1 else -1


    up_vec = pt.copy()
    down_vec = pt.copy()
    for i in range(up_down_steps):
        up_vec += normal_step
        if ( lookup_vec(up_vec, arr, lb, cs, shape) == look_for ):
            return up_vec
        down_vec -= normal_step
        if ( lookup_vec(down_vec, arr, lb, cs, shape) == look_for ):
            return down_vec


    fail[0] = True
    return up_vec

@njit(fastmath=True,cache=True)
def distance_two_pts(pt1, pt2):
    x = pt1[0] - pt2[0]
    y = pt1[1] - pt2[1]
    z = pt1[2] - pt2[2]

    return np.sqrt( x*x + y*y + z*z )

# keep, visited locations, current distance
@njit(fastmath=True,cache=True)
def numba_do_surface_crawl(start, normal, direction, distance, arr, lb, ub, cs, shape):

    up_down_steps = 20
    up_down_step = cs[0]*0.3
    normal_step = normal*up_down_step

    forward_step_size = cs[0] 
    forward_step = forward_step_size * direction


    fail = np.array([0], np.bool_)

    traversed = []
    traveled = 0

    prev = start
    current = start
    while ( traveled < distance ):

        surf = numba_seek_to_surface(current, normal_step, up_down_steps, fail, arr, lb, ub, cs, shape)
        if ( fail[0] ):
            return traversed, traveled

        traversed.append(surf)

        # traveled += distance_two_pts( surf, prev )
        traveled = distance_two_pts( surf, start )
        prev = surf
        current = prev + forward_step

    return traversed, traveled


@njit(fastmath=True,cache=True)
def numba_add_to_near_grid(pts, store_vals, atom_radius, near_grid, dist_grid, lb, ub, cs, shape):
    for i in range(len(pts)):
        pt = pts[i]
        store_val = store_vals[i]
        numba_store_near_grid(near_grid, dist_grid, atom_radius*2, pt, store_val, lb, ub, cs, shape)


@njit(fastmath=True,cache=True)
def numba_store_near_grid(near_grid, dist_grid, _x, pt, idx, lb, ub, cs, shape):

    # these should like really be here
    assert(len(pt) == 3)

    low_high = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.float_)
    for i in range(3):
        low_high[0, i] = pt[i] - _x
        low_high[1, i] = pt[i] + _x

    for i in range(3):
        assert( low_high[0, i] > lb[i] + cs[i] )
        assert( low_high[1, i] < ub[i] - cs[i] )

    # transform bounds into upper and lower corners in voxel array indices
    bounds = xform_vectors( low_high, lb, cs, shape )


    # translate voxel array indices back to 3d coords and do distance check
    _x2 = _x*_x
     
    for i in range(bounds[0, 0], bounds[1, 0] + 1):
        x = numba_ind_index_to_center(i, lb[0], cs[0]) - pt[0]
        x2 = x*x
        for j in range(bounds[0, 1], bounds[1, 1] + 1):
            y = numba_ind_index_to_center(j, lb[1], cs[1]) - pt[1]
            y2 = y*y
            for k in range(bounds[0, 2], bounds[1, 2] + 1):
                z = numba_ind_index_to_center(k, lb[2], cs[2]) - pt[2]
                z2 = z*z
                dist2 = x2 + y2 + z2
                if ( dist2 < _x2 ):
                    if ( dist2 < dist_grid[i, j, k] ):
                        near_grid[i, j, k] = idx
                        dist_grid[i, j, k] = dist2


@njit(fastmath=True,cache=True)
def numba_make_sum_grid(pts, atom_radius, arr, lb, ub, cs, shape, store_val):
    for i in range(len(pts)):
        pt = pts[i]
        numba_indices_add_within_x_of(arr, store_val, atom_radius*2, pt, lb, ub, cs, shape)


@njit(fastmath=True,cache=True)
def numba_indices_add_within_x_of(arr, to_store, _x, pt, lb, ub, cs, shape):

    # these should like really be here
    assert(len(pt) == 3)

    low_high = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.float_)
    for i in range(3):
        low_high[0, i] = pt[i] - _x
        low_high[1, i] = pt[i] + _x

    for i in range(3):
        assert( low_high[0, i] > lb[i] + cs[i] )
        assert( low_high[1, i] < ub[i] - cs[i] )

    # transform bounds into upper and lower corners in voxel array indices
    bounds = xform_vectors( low_high, lb, cs, shape )


    # translate voxel array indices back to 3d coords and do distance check
    _x2 = _x*_x
     
    for i in range(bounds[0, 0], bounds[1, 0] + 1):
        x = numba_ind_index_to_center(i, lb[0], cs[0]) - pt[0]
        x2 = x*x
        for j in range(bounds[0, 1], bounds[1, 1] + 1):
            y = numba_ind_index_to_center(j, lb[1], cs[1]) - pt[1]
            y2 = y*y
            for k in range(bounds[0, 2], bounds[1, 2] + 1):
                z = numba_ind_index_to_center(k, lb[2], cs[2]) - pt[2]
                z2 = z*z
                if ( x2 + y2 + z2 < _x2 ):
                    arr[i, j, k] += to_store





@njit(fastmath=True,cache=True)
def numba_make_clashgrid(pts, atom_radius, arr, lb, ub, cs, shape, store_val):
    for i in range(len(pts)):
        pt = pts[i]
        numba_indices_store_within_x_of(arr, store_val, atom_radius*2, pt, lb, ub, cs, shape)

@njit(fastmath=True,cache=True)
def numba_make_clashgrid_var_atom_radius(pts, atom_radius, arr, lb, ub, cs, shape, store_val):
    for i in range(len(pts)):
        pt = pts[i]
        radius = atom_radius[i]
        numba_indices_store_within_x_of(arr, store_val, radius*2, pt, lb, ub, cs, shape)


@njit(fastmath=True,cache=True)
def numba_indices_store_within_x_of(arr, to_store, _x, pt, lb, ub, cs, shape):

    # these should like really be here
    assert(len(pt) == 3)

    low_high = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.float_)
    for i in range(3):
        low_high[0, i] = pt[i] - _x
        low_high[1, i] = pt[i] + _x

    for i in range(3):
        assert( low_high[0, i] > lb[i] + cs[i] )
        assert( low_high[1, i] < ub[i] - cs[i] )

    # transform bounds into upper and lower corners in voxel array indices
    bounds = xform_vectors( low_high, lb, cs, shape )


    # translate voxel array indices back to 3d coords and do distance check
    _x2 = _x*_x
     
    for i in range(bounds[0, 0], bounds[1, 0] + 1):
        x = numba_ind_index_to_center(i, lb[0], cs[0]) - pt[0]
        x2 = x*x
        for j in range(bounds[0, 1], bounds[1, 1] + 1):
            y = numba_ind_index_to_center(j, lb[1], cs[1]) - pt[1]
            y2 = y*y
            for k in range(bounds[0, 2], bounds[1, 2] + 1):
                z = numba_ind_index_to_center(k, lb[2], cs[2]) - pt[2]
                z2 = z*z
                if ( x2 + y2 + z2 < _x2 ):
                    arr[i, j, k] = to_store




        

@njit(fastmath=True,cache=True)
def numba_index_to_center(vec, lb, cs, shape):
    out = np.array([0, 0, 0])
    for i in range(3):
        out = (vec[i] + 0.5) * cs[i] + lb[i]
    return out

@njit(fastmath=True,cache=True)
def numba_ind_index_to_center(i, lb, cs):
    return (i + 0.5) * cs + lb


@njit(fastmath=True,cache=True)
def numba_indices_to_centers(inds, lb, cs):
    out = np.zeros((len(inds), len(lb)), dtype=np.float_)
    for i in range(len(inds)):
        for j in range(len(lb)):
            out[i, j] = (inds[i, j] + 0.5) * cs[j] + lb[j]
    return out


@njit(fastmath=True,cache=True)
def xform_vectors(vecs, lb, cs, shape):
    out = np.zeros((len(vecs), len(lb)), dtype=np.int_)
    return xform_vectors_w_out(vecs, lb, cs, shape, out)

@njit(fastmath=True,cache=True)
def xform_vectors_w_out(vecs, lb, cs, shape, out):
    for i in range(len(vecs)):
        for j in range(len(lb)):
            out[i, j] = xform_1_pt(vecs[i, j], lb[j], cs[j], shape[j])
    return out

@njit(fastmath=True,cache=True)
def xform_vector(vec, lb, cs, shape):
    out = np.array([0, 0, 0], dtype=np.int_)
    for i in range(len(vec)):
        out[i] = xform_1_pt(vec[i], lb[i], cs[i], shape[i])
    return out

@njit(fastmath=True,cache=True)
def xform_1_pt(pt, lb, cs, shape):
    x = np.int( ( pt - lb ) / cs )
    if ( x <= 0 ):
        return np.int(0)
    if ( x >= shape-1 ):
        return shape-1
    return x

@njit(fastmath=True,cache=True)
def lookup_vec(vec, arr, lb, cs, shape):
    return arr[xform_1_pt(vec[0], lb[0], cs[0], shape[0]),
               xform_1_pt(vec[1], lb[1], cs[1], shape[1]),
               xform_1_pt(vec[2], lb[2], cs[2], shape[2])
            ]

@njit(fastmath=True,cache=True)
def numba_clash_check(pts, max_clashes, arr, lb, cs):
    
    clashes = 0

    for i in range(len(pts)):
        pt = pts[i]
        x = xform_1_pt(pt[0], lb[0], cs[0], arr.shape[0])
        y = xform_1_pt(pt[1], lb[1], cs[1], arr.shape[1])
        z = xform_1_pt(pt[2], lb[2], cs[2], arr.shape[2])

        clashes += arr[x, y, z]

        if ( clashes > max_clashes ):
            return clashes

    return clashes


@njit(fastmath=True,cache=True)
def numba_ray_trace_many(starts, ends, max_clashes, arr, lb, cs, debug=False):
    clashes = np.zeros(len(starts), np.int_)
    for i in range(len(starts)):
        clashes[i] = numba_ray_trace(starts[i], ends[i], max_clashes, arr, lb, cs, debug)

    return clashes



@njit(fastmath=True,cache=True)
def numba_ray_trace(start, end, max_clashes, arr, lb, cs, debug=False):

    arr_start = np.zeros((3), np.float_)
    arr_start[0] = xform_1_pt(start[0], lb[0], cs[0], arr.shape[0])
    arr_start[1] = xform_1_pt(start[1], lb[1], cs[1], arr.shape[1])
    arr_start[2] = xform_1_pt(start[2], lb[2], cs[2], arr.shape[2])

    arr_end = np.zeros((3), np.float_)
    arr_end[0] = xform_1_pt(end[0], lb[0], cs[0], arr.shape[0])
    arr_end[1] = xform_1_pt(end[1], lb[1], cs[1], arr.shape[1])
    arr_end[2] = xform_1_pt(end[2], lb[2], cs[2], arr.shape[2])

    slope = arr_end - arr_start
    largest = np.max(np.abs(slope))
    slope /= largest

    max_iter = largest+1

    clashes = 0
    x = arr_start[0]
    y = arr_start[1]
    z = arr_start[2]
    for i in range(max_iter):
        clashes += arr[int(x+0.5), int(y+0.5), int(z+0.5)]
        if ( debug ):
            print(i, largest, slope)
            arr[int(x+0.5), int(y+0.5), int(z+0.5)] = True
        if ( clashes >= max_clashes ):
            return clashes
        x += slope[0]
        y += slope[1]
        z += slope[2]

    return clashes



@njit(fastmath=True,cache=True)
def _lookup_3d(null_val, loc, arr, shape):
    if ( loc[0] == 0 or loc[0] >= shape[0]-1):
        return null_val
    if ( loc[1] == 0 or loc[1] >= shape[1]-1):
        return null_val
    if ( loc[2] == 0 or loc[2] >= shape[2]-1):
        return null_val
    return arr[loc[0], loc[1], loc[2]]

@njit(fastmath=True,cache=True)
def _increase_ptr( ptr, offset, cur_stack, stacks, stack_sizes, generate_stack ):
    ptr += 1
    if ( ptr == stack_sizes[cur_stack] ):
        offset += stack_sizes[cur_stack]
        cur_stack += 1
        if ( generate_stack ):
            stacks[cur_stack] = np.zeros((stack_sizes[cur_stack], 3), np.int_)
    return ptr, offset, cur_stack

# faster flood fill but harder to write
@njit(fastmath=True,cache=True)
def numba_flood_fill_3d_from_here(fill_val, overwrite_val, start_idx, arr, lb, ub, cs, shape ):

    num_points = shape[0]*shape[1]*shape[2]

    stack_size0 = num_points//100 + 2 # plus 2 so the 2nd element can't be in 2
    stack_size1 = num_points//10 + 1
    stack_size2 = num_points

    stack0 = np.zeros((stack_size0, 3), np.int_)
    stack1 = np.zeros((1, 3), np.int_)
    stack2 = np.zeros((1, 3), np.int_)

    stack_sizes = [stack_size0, stack_size1, stack_size2]
    stacks = [stack0, stack1, stack2]


    stack0[0] = start_idx
    arr[start_idx[0], start_idx[1], start_idx[2]] = fill_val
    stack0[0] = start_idx

    process_ptr = 0
    process_offset = 0
    cur_process_stack = 0

    set_offset = 0
    cur_set_stack = 0
    set_ptr = 1

    while ( process_ptr < set_ptr ):
        # print(set_ptr, stack_size0)

        loc = stacks[cur_process_stack][process_ptr - process_offset]

        process_ptr, process_offset, cur_process_stack = _increase_ptr( 
                            process_ptr, process_offset, cur_process_stack, stacks, stack_sizes, False)

        # right
        loc[0] += 1
        if ( _lookup_3d(fill_val, loc, arr, shape) == overwrite_val ):
            arr[loc[0], loc[1], loc[2]] = fill_val
            stacks[cur_set_stack][set_ptr - set_offset] = loc
            set_ptr, set_offset, cur_set_stack = _increase_ptr( 
                            set_ptr, set_offset, cur_set_stack, stacks, stack_sizes, True)
        # left
        loc[0] -= 2
        if ( _lookup_3d(fill_val, loc, arr, shape) == overwrite_val ):
            arr[loc[0], loc[1], loc[2]] = fill_val
            stacks[cur_set_stack][set_ptr - set_offset] = loc
            set_ptr, set_offset, cur_set_stack = _increase_ptr( 
                            set_ptr, set_offset, cur_set_stack, stacks, stack_sizes, True)
        # down
        loc[0] += 1
        loc[1] += 1
        if ( _lookup_3d(fill_val, loc, arr, shape) == overwrite_val ):
            arr[loc[0], loc[1], loc[2]] = fill_val
            stacks[cur_set_stack][set_ptr - set_offset] = loc
            set_ptr, set_offset, cur_set_stack = _increase_ptr( 
                            set_ptr, set_offset, cur_set_stack, stacks, stack_sizes, True)
        # up
        loc[1] -= 2
        if ( _lookup_3d(fill_val, loc, arr, shape) == overwrite_val ):
            arr[loc[0], loc[1], loc[2]] = fill_val
            stacks[cur_set_stack][set_ptr - set_offset] = loc
            set_ptr, set_offset, cur_set_stack = _increase_ptr( 
                            set_ptr, set_offset, cur_set_stack, stacks, stack_sizes, True)
        # forward
        loc[1] += 1
        loc[2] += 1
        if ( _lookup_3d(fill_val, loc, arr, shape) == overwrite_val ):
            arr[loc[0], loc[1], loc[2]] = fill_val
            stacks[cur_set_stack][set_ptr - set_offset] = loc
            set_ptr, set_offset, cur_set_stack = _increase_ptr( 
                            set_ptr, set_offset, cur_set_stack, stacks, stack_sizes, True)
        # backward
        loc[2] -= 2
        if ( _lookup_3d(fill_val, loc, arr, shape) == overwrite_val ):
            arr[loc[0], loc[1], loc[2]] = fill_val
            stacks[cur_set_stack][set_ptr - set_offset] = loc
            set_ptr, set_offset, cur_set_stack = _increase_ptr( 
                            set_ptr, set_offset, cur_set_stack, stacks, stack_sizes, True)
        loc[2] += 1






# this does forward filling going from 0->hi and hi->0
# is this a fast way to do it? no idea
# don't allow diagonal filling for speed
@njit(fastmath=True,cache=True)
def numba_flood_fill_3d(fill_val, overwrite_val, arr, lb, ub, cs, shape ):

    # for cache-coherence, we always iter on z last
    any_changed = True
    while (any_changed):
        any_changed = False

        # forward fill in positive direction
        for x in range(1, shape[0]-2):
            for y in range(1, shape[1]-2):
                for z in range(1, shape[2]-2):
                    if ( arr[x, y, z] != fill_val ):
                        continue
                    if ( arr[x, y, z+1] == overwrite_val ):
                        arr[x, y, z+1] = fill_val
                        any_changed = True
                    if ( arr[x, y+1, z] == overwrite_val ):
                        arr[x, y+1, z] = fill_val
                        any_changed = True
                    if ( arr[x+1, y, z] == overwrite_val ):
                        arr[x+1, y, z] = fill_val
                        any_changed = True

        # forward fill in negative direction
        for x in range(shape[0]-2, 1, -1):
            for y in range(shape[1]-2, 1, -1):
                for z in range(shape[2]-2, 1, -1):
                    if ( arr[x, y, z] != fill_val ):
                        continue
                    if ( arr[x, y, z-1] == overwrite_val ):
                        arr[x, y, z-1] = fill_val
                        any_changed = True
                    if ( arr[x, y-1, z] == overwrite_val ):
                        arr[x, y-1, z] = fill_val
                        any_changed = True
                    if ( arr[x-1, y, z] == overwrite_val ):
                        arr[x-1, y, z] = fill_val
                        any_changed = True




# this does forward filling going from 0->hi and hi->0
# is this a fast way to do it? no idea
# don't allow diagonal filling for speed
@njit(fastmath=True,cache=True)
def numba_flood_fill_2d(fill_val, overwrite_val, arr, lb, ub, cs, shape  ):

    # for cache-coherence, we always iter on z last
    any_changed = True
    while (any_changed):
        any_changed = False

        # forward fill in positive direction
        for x in range(1, shape[0]-2):
            for y in range(1, shape[1]-2):
                if ( arr[x, y] != fill_val ):
                    continue
                if ( arr[x, y+1] == overwrite_val ):
                    arr[x, y+1] = fill_val
                    any_changed = True
                if ( arr[x+1, y] == overwrite_val ):
                    arr[x+1, y] = fill_val
                    any_changed = True

        # forward fill in negative direction
        for x in range(shape[0]-2, 1, -1):
            for y in range(shape[1]-2, 1, -1):
                if ( arr[x, y] != fill_val ):
                    continue
                if ( arr[x, y-1] == overwrite_val ):
                    arr[x, y-1] = fill_val
                    any_changed = True
                if ( arr[x-1, y] == overwrite_val ):
                    arr[x-1, y] = fill_val
                    any_changed = True













