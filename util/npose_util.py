#!/usr/bin/env python

import os
import sys
import math

import voxel_array

import pandas as pd
import numpy as np
import warnings
import gzip
import struct
import itertools

from collections import OrderedDict

try:
    from numba import njit
    from numba import jit
    import numba
except:
    sys.path.append("/home/bcov/sc/random/just_numba")
    from numba import njit

# silent tools is conditionally imported when needed
sys.path.append("/home/bcov/silent_tools")

# Useful numbers
# N [-1.45837285,  0 , 0]
# CA [0., 0., 0.]
# C [0.55221403, 1.41890368, 0.        ]
# CB [ 0.52892494, -0.77445692, -1.19923854]

if ( hasattr(os, 'ATOM_NAMES') ):
    assert( hasattr(os, 'PDB_ORDER') )

    ATOM_NAMES = os.ATOM_NAMES
    PDB_ORDER = os.PDB_ORDER
else:
    ATOM_NAMES=['N', 'CA', 'CB', 'C', 'O']
    PDB_ORDER = ['N', 'CA', 'C', 'O', 'CB']

_byte_atom_names = []
_atom_names = []
for i, atom_name in enumerate(ATOM_NAMES):
    long_name = " " + atom_name + "       "
    _atom_names.append(long_name[:4])
    _byte_atom_names.append(atom_name.encode())

    globals()[atom_name] = i

R = len(ATOM_NAMES)

if ( "N" not in globals() ):
    N = -1
if ( "C" not in globals() ):
    C = -1
if ( "CB" not in globals() ):
    CB = -1


_pdb_order = []
for name in PDB_ORDER:
    _pdb_order.append( ATOM_NAMES.index(name) )


def gzopen(name, mode="rt"):
    if (name.endswith(".gz")):
        return gzip.open(name, mode)
    else:
        return open(name, mode)


_space = " ".encode()[0]
@njit(fastmath=True)
def space_strip(string):
    start = 0
    while(start < len(string) and string[start] == _space):
        start += 1
    end = len(string)
    while( end > 0 and string[end-1] == _space):
        end -= 1
    return string[start:end]

@njit(fastmath=True)
def byte_startswith( haystack, needle ):
    if ( len(haystack) < len(needle) ):
        return False
    for i in range(len(needle)):
        if ( haystack[i] != needle[i] ):
            return False
    return True

@njit(fastmath=True)
def byte_equals( haystack, needle ):
    if ( len(haystack) != len(needle) ):
        return False
    for i in range(len(needle)):
        if ( haystack[i] != needle[i] ):
            return False
    return True

@njit(fastmath=True)
def getline( bytess, start ):
    cur = start
    while (cur < len(bytess) and bytess[cur] != 10 ):
        cur += 1
    cur += 1
    return bytess[start:cur], cur

# ord(" ") == 32
# ord("-") == 45
# ord(".") == 46
# ord("0") == 48
# ord("9") == 57

@njit(fastmath=True)
def stof(string):
    multiplier = 0

    parsed = False
    negative = 1
    start = 0
    end = len(string) - 1
    for i in range(len(string)):
        char = string[i]
        # print(char)
        if ( not parsed ):
            if ( char == 32 ): # " "
                start = i + 1
                continue
            if ( char == 45 ): # "-"
                start = i + 1
                negative = -1
                continue
        if ( char == 32 ): # " "
            break
        if ( char == 46 ): # "."
            multiplier = np.float64(1)
            parsed = True
            continue
        if ( char >= 48 and char <= 57 ): # 0 9
            parsed = True
            multiplier /= np.float64(10)
            end = i
            continue
        print("Float parse error! Unrecognized character: ", char)
        assert(False)

    if ( not parsed ):
        print("Float parse error!")
        assert(False)

    result = np.float64(0)

    if ( multiplier == 0 ):
        multiplier = 1
    for i in range(end, start-1, -1):
        char = string[i]
        if ( char == 46 ): # "."
            continue
        value = np.float64(char - 48) # 0

        result += value * multiplier
        multiplier *= np.float64(10)

    result *= negative

    return np.float32(result)

@njit(fastmath=True, cache=True)
def parse_aa(name3):

    a = name3[0]
    b = name3[1]
    c = name3[2]

    if ( a <= 73 ): # I
        if ( a <= 67 ): # C
            if ( a == 65 ): # A
                if ( b == 83 ):
                    if ( c == 78 ):
                        return 78
                    if ( c == 80 ):
                        return 68
                    return 88
                if ( b == 76 and c == 65 ):
                    return 65
                if ( b == 82 and c == 71 ):
                    return 82
            if ( a == 67 and b == 89 and c == 83 ): # C
                    return 67
        else:
            if ( a == 71 ): # G
                if ( b != 76 ):
                    return 88
                if ( c == 85 ):
                    return 69
                if ( c == 89 ):
                    return 71
                if ( c == 78 ):
                    return 81
            if ( a == 72 and b == 73 and c == 83 ): # H
                    return 72
            if ( a == 73 and b == 76 and c == 69 ): # I
                    return 73
    else: 
        if ( a <= 80 ): # P
            if ( a == 76 ): # L
                if ( b == 69 and c == 85 ):
                    return 76
                if ( b == 89 and c == 83 ):
                    return 75
            if ( a == 77 ): # M
                if ( b == 69 and c == 84 ):
                    return 77
            if ( a == 80 ): # P
                if ( b == 72 and c == 69 ):
                    return 70
                if ( b == 82 and c == 79 ):
                    return 80
        else:
            if ( a == 83 and b == 69 and c == 82 ): # S
                    return 83
            if ( a == 84 ): # T
                if ( c == 82 ):
                    if ( b == 72 ):
                        return 84
                    if ( b == 89 ):
                        return 89
                    return 88
                if ( b == 82 and c == 80 ):
                    return 87
            if ( a == 86 and b == 65 and c == 76 ): # V
                    return 86
    return 88


# _atom = "ATOM".encode()
_null_line = "ATOMCB00000000000000000000000000000000000000000000000000000000"#.encode()
_null_line_size = len(_null_line)

# _CB = "CB".encode()
# _empty_bytes = "".encode()

# Switches to next residue whenever
#  Line doesn't start with atom
#  Line isn't long enough
#  Res/resnum/chain changes
@njit(fastmath=True, cache=True)
def read_npose_from_data( data, null_line_atom_names, NCACCBR, scratch_residues, scratch_chains, scratch_aa):


    _null_line = null_line_atom_names[:_null_line_size]
    array_byte_atom_names = null_line_atom_names[_null_line_size:]

    _CB = _null_line[4:6]
    _atom = _null_line[:4]
    _empty_bytes = _null_line[:0]

    N = NCACCBR[0]
    CA = NCACCBR[1]
    C = NCACCBR[2]
    CB = NCACCBR[3]
    R = NCACCBR[4]

    byte_atom_names = []
    for i in range(len(array_byte_atom_names)//4):
        aname = array_byte_atom_names[i*4:i*4+4]
        byte_atom_names.append(space_strip(aname))

    seqpos = 0
    res_ident = _empty_bytes
    res_has_n_atoms = 0
    next_res = False
    scratch_residues[0].fill(0)

    # lines.append(_nulline)

    cursor = 0
    keep_going = True
    while keep_going:
        line, cursor = getline(data, cursor)
        if ( cursor >= len(data) ):
            keep_going = False
            line = _null_line

        # print(iline)

        if ( not byte_startswith(line, _atom) or len(line) < 54 ):
            next_res = True
            res_ident = _empty_bytes
            continue

        ident = line[17:27]
        if ( not byte_equals( ident, res_ident ) ):
            next_res = True

        if ( next_res ):
            if ( res_has_n_atoms > 0 ):

                res = scratch_residues[seqpos]
                if ( res_has_n_atoms != R ):
                    missing = np.where(res[:,3] == 0)[0]

                    # We only know how to fix missing CB
                    first_missing = byte_atom_names[missing[0]]
                    if ( len(missing) > 1 or not byte_equals(first_missing,  _CB) ):
                        print("Error! missing atoms:")
                        for i in range(len(missing)):
                            print(byte_atom_names[missing[i]])
                        print("in residue:")
                        print(res_ident)
                        assert(False)

                    # Fixing CB
                    xform = get_stub_from_n_ca_c(res[N,:3], res[CA,:3], res[C,:3])
                    res[CB] = get_CB_from_xform( xform )


                seqpos += 1
                #If we run out of scratch, double its size
                if ( seqpos == len(scratch_residues) ):
                    old_size = len(scratch_residues)
                    new_scratch = np.zeros((old_size*2, R, 4), np.float32)
                    for i in range(old_size):
                        new_scratch[i] = scratch_residues[i]
                    scratch_residues = new_scratch

                    new_scratch2 = np.zeros((old_size*2), np.byte)
                    for i in range(old_size):
                        new_scratch2[i] = scratch_chains[i]
                    scratch_chains = new_scratch2

                    new_scratch2 = np.zeros((old_size*2), np.byte)
                    for i in range(old_size):
                        new_scratch2[i] = scratch_aa[i]
                    scratch_aa = new_scratch2


                scratch_residues[seqpos].fill(0)

            res_ident = ident
            res_has_n_atoms = 0
            next_res = False


        # avoid parsing stuff we know we don't need
        if ( res_has_n_atoms == R ):
            continue

        atom_name = space_strip(line[12:16])

        # figure out which atom we have
        atomi = -1
        for i in range( R ):
            if ( byte_equals( atom_name, byte_atom_names[i] ) ):
                atomi = i
                break
        if ( atomi == -1 ):
            continue

        res = scratch_residues[seqpos]
        if ( res[atomi,3] != 0 ):
            print("Error! duplicate atom:")
            print( atom_name )
            print("in residue:" )
            print( res_ident )
            assert(False)

        res_has_n_atoms += 1

        res[atomi,0] = stof(line[30:38])
        res[atomi,1] = stof(line[38:46])
        res[atomi,2] = stof(line[46:54])
        res[atomi,3] = 1

        if ( res_has_n_atoms == 1 ):
            scratch_chains[seqpos] = line[21]
            scratch_aa[seqpos] = parse_aa(line[17:20])

    to_ret = np.zeros((seqpos, R, 4))
    for i in range(seqpos):
        to_ret[i] = scratch_residues[i]
    
    return to_ret.reshape(-1, 4), scratch_residues, scratch_chains, scratch_aa


g_scratch_residues = np.zeros((1000,R,4), np.float32)
g_scratch_chains = np.zeros((1000), np.byte)
g_scratch_aa = np.zeros((1000), np.byte)
# for i in range(1000):
#     g_scratch_residues.append(np.zeros((R,4), np.float32))

_array_byte_atom_names = list(" "*len(_byte_atom_names)*4)
for i in range(len(_byte_atom_names)):
    name = _atom_names[i]
    for j in range(len(name)):
        _array_byte_atom_names[i*4+j] = name[j]
_array_atom_names = "".join(_array_byte_atom_names)

_null_line_atom_names = (_null_line + _array_atom_names).encode()

NCACCBR = np.zeros(5, np.int64)
if ( "N" in locals() ):
    NCACCBR[0] = N
if ( "CA" in locals() ):
    NCACCBR[1] = CA
if ( "C" in locals() ):
    NCACCBR[2] = C
if ( "CB" in locals() ):
    NCACCBR[3] = CB
NCACCBR[4] = R

def npose_from_file(fname, chains=False, aa=False):
    return npose_from_file_fast(fname, chains, aa)

def npose_from_file_fast(fname, chains=False, aa=False):
    with gzopen(fname, "rb") as f:
        data = f.read()
    return npose_from_bytes(data, chains, aa)

def npose_from_bytes(data, chains=False, aa=False):

    global g_scratch_residues
    global g_scratch_chains
    global g_scratch_aa

    npose, scratch, scratch_chains, scratch_aa = read_npose_from_data( data, _null_line_atom_names, NCACCBR, g_scratch_residues, 
                                                                                                g_scratch_chains, g_scratch_aa)
    np.around(npose, 3, npose)  # get rid of random numba noise

    g_scratch_residues = scratch
    g_scratch_chains = scratch_chains
    g_scratch_aa = scratch_aa

    output = [npose]
    if ( chains ):
        output.append(bytes(scratch_chains[:nsize(npose)]).decode("ascii"))
    if ( aa ):
        output.append(bytes(scratch_aa[:nsize(npose)]).decode("ascii"))
    if ( len(output) == 1 ):
        return output[0]
    return output


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

@njit(fastmath=True)
def cross(vec1, vec2):
    result = np.zeros(3)
    a1, a2, a3 = vec1[0], vec1[1], vec1[2]
    b1, b2, b3 = vec2[0], vec2[1], vec2[2]
    result[0] = a2 * b3 - a3 * b2
    result[1] = a3 * b1 - a1 * b3
    result[2] = a1 * b2 - a2 * b1
    return result


def build_CB(tpose):
    CB_pos = np.array([0.52980185, -0.77276349, -1.19909418, 1.0], dtype=np.float)
    return tpose @ CB_pos

def build_H(npose):
    ncac = extract_atoms(npose, [N, CA, C])

    # we have freedom on where to build the first H
    # we access the freedom by deciding where to build the first C
    # it has to be 120 degrees accross from N-CA

    first_CA_to_N = ncac[0,:3] - ncac[1,:3]
    first_CA_to_N /= np.linalg.norm(first_CA_to_N)

    perp = np.cross( first_CA_to_N, np.array([1, 0, 0] ) )
    if ( np.linalg.norm(perp) == 0 ):
        perp = np.cross( first_CA_to_N, np.array([0, 1, 0] ) )
    perp /= np.linalg.norm(perp)

    C_sin_vect = np.cross(first_CA_to_N, perp)
    C_sin_vect /= np.linalg.norm(C_sin_vect)

    #                                    -cos(120)                      sin(120) 
    to_first_c = first_CA_to_N * 0.4999999999999998 + C_sin_vect * 0.8660254037844387

    first_c = ncac[0].copy()
    first_c[:3] += to_first_c

    # assert( np.abs(np.dot( to_first_c, -first_CA_to_N ) - np.cos(np.radians(120))) < 0.01 )

    cnca = np.r_[ first_c[None,:], ncac[:-1] ].reshape(-1, 3, 4)[...,:3]

    to_c_unit = cnca[:,0] - cnca[:,1]
    to_c_unit /= np.linalg.norm(to_c_unit, axis=-1)[...,None]

    to_ca_unit = cnca[:,2] - cnca[:,1]
    to_ca_unit /= np.linalg.norm(to_ca_unit, axis=-1)[...,None]


    to_anti_h = to_c_unit + to_ca_unit
    to_anti_h /= np.linalg.norm(to_anti_h, axis=-1)[...,None]

    # to_anti_h = anti_h - cnca[:,1]
    # to_anti_h /= np.linalg.norm(to_anti_h, axis=-1)[...,None]

    h_bond_length = 1.01

    H_coord = cnca[:,1] - h_bond_length * to_anti_h

    return H_coord


# the CA C O angle is 120.8
def build_O(npose):
    ncac = extract_atoms(npose, [N, CA, C])

    return build_O_ncac(ncac)

def build_O_ncac(ncac):

    # we have freedom on where to build the last O
    # we access the freedom by deciding where to build the last N
    # it has to be 116.2 degrees accross from N-CA

    last_CA_to_C = ncac[-1,:3] - ncac[-2,:3]
    last_CA_to_C /= np.linalg.norm(last_CA_to_C)

    perp = np.cross( last_CA_to_C, np.array([1, 0, 0] ) )
    if ( np.linalg.norm(perp) == 0 ):
        perp = np.cross( last_CA_to_C, np.array([0, 1, 0] ) )
    perp /= np.linalg.norm(perp)

    N_sin_vect = np.cross(last_CA_to_C, perp)
    N_sin_vect /= np.linalg.norm(N_sin_vect)

    #                                    -cos(116.2)                      sin(116.2) 
    to_last_n = last_CA_to_C * 0.44150585279174515 + N_sin_vect * 0.8972583696743285

    last_n = ncac[-1].copy()
    last_n[:3] += to_last_n

    assert( np.abs(np.dot( to_last_n, -last_CA_to_C ) - np.cos(np.radians(116.2))) < 0.01 )

    cacn = np.r_[ ncac[1:], last_n[None,:]].reshape(-1, 3, 4)[...,:3]

    to_ca_unit = cacn[:,0] - cacn[:,1]
    to_ca_unit /= np.linalg.norm(to_ca_unit, axis=-1)[...,None]

    to_n_unit = cacn[:,2] - cacn[:,1]
    to_n_unit /= np.linalg.norm(to_n_unit, axis=-1)[...,None]

    the_cross = np.cross(to_ca_unit, to_n_unit)
    o_sin_unit = np.cross(to_ca_unit, the_cross)
    o_sin_unit /= np.linalg.norm(o_sin_unit, axis=-1)[...,None]

    #                               cos(120.8)                      sin(120.8)
    c_to_o_unit = to_ca_unit * -0.5120428648705714 + o_sin_unit * 0.8589598969306645

    # to_anti_O = to_ca_unit + to_n_unit
    # to_anti_O /= np.linalg.norm(to_anti_O, axis=-1)[...,None]

    # to_anti_h = anti_h - cnca[:,1]
    # to_anti_h /= np.linalg.norm(to_anti_h, axis=-1)[...,None]

    o_bond_length = 1.231015

    O_coord = cacn[:,1] + o_bond_length * c_to_o_unit

    return O_coord


def build_npose_from_tpose(tpose):
    npose = np.ones((len(tpose)*R, 4), np.float64)

    by_res = npose.reshape(-1, R, 4)

    if ( "N" in ATOM_NAMES ):
        by_res[:,N,:] = get_N_from_xform(tpose)
    if ( "CA" in ATOM_NAMES ):
        by_res[:,CA,:3] = tpose[:,:3,3]
    if ( "C" in ATOM_NAMES ):
        by_res[:,C,:] = get_C_from_xform(tpose)
    if ( "CB" in ATOM_NAMES ):
        by_res[:,CB,:] = build_CB(tpose)
    if ( "O" in ATOM_NAMES ):
        by_res[:,O,:3] = build_O(npose)

    return npose

RT_dihedral_180 = np.array( [[ 4.49319115e-01,  8.93371330e-01,  2.34256657e-16,  2.52958446e+00],
                             [ 8.93371330e-01, -4.49319115e-01, -2.21310612e-16,  2.83850891e+00],
                             [-9.24565618e-17,  3.08717270e-16, -1.00000000e+00,  1.11974031e-17],
                             [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]],
                             np.float64 )

RT_rosetta_res1 = np.array( [[1.      , 0.      , 0.      , 1.458001],
                             [0.      , 1.      , 0.      , 0.      ],
                             [0.      , 0.      , 1.      , 0.      ],
                             [0.      , 0.      , 0.      , 1.      ]],
                             np.float64)


def npose_from_length(size):

    tpose = np.zeros((size, 4, 4), np.float64)
    tpose[0] = RT_rosetta_res1

    # the numerical errors are just painful here
    for i in range(size-1):
        tpose[i+1] = tpose[i] @ RT_dihedral_180

    return build_npose_from_tpose(tpose)


def nsize(npose):
    return int(len(npose)/R)

def tsize(tpose):
    return len(tpose)

def itsize(itpose):
    return len(itpose)

def get_res( npose, resnum):
    return npose[R*resnum:R*(resnum+1)]

@njit(fastmath=True, cache=True)
def get_stub_from_n_ca_c(n, ca, c):
    e1 = ca - n
    e1 /= np.linalg.norm(e1)

    e3 = cross( e1, c - n )
    e3 /= np.linalg.norm(e3)

    e2 = cross( e3, e1 )

    stub = np.identity(4)
    stub[:3,0] = e1
    stub[:3,1] = e2
    stub[:3,2] = e3
    stub[:3,3] = ca

    return stub

@njit(fastmath=True, cache=True)   
def get_stubs_from_n_ca_c(n, ca, c):
    out = np.zeros((len(n), 4, 4), np.float_)
    for i in range(len(n)):
        out[i] = get_stub_from_n_ca_c(n[i], ca[i], c[i])
    return out

# def get_stubs_from_n_ca_c(n, ca, c):
#     e1 = ca - n
#     e1 = np.divide( e1, np.linalg.norm(e1, axis=1)[..., None] )

#     e3 = np.cross( e1, c - n, axis=1 )
#     e3 = np.divide( e3, np.linalg.norm(e3, axis=1)[..., None] )

#     e2 = np.cross( e3, e1, axis=1 )

#     stub = np.zeros((len(n), 4, 4))
#     stub[...,:3,0] = e1
#     stub[...,:3,1] = e2
#     stub[...,:3,2] = e3
#     stub[...,:3,3] = ca
#     stub[...,3,3] = 1.0

#     return stub

def get_stub_from_npose(npose, resnum):
    # core::kinematics::Stub( CA, N, C )

    res = get_res(npose, resnum)

    return get_stub_from_n_ca_c(res[N,:3], res[CA,:3], res[C,:3])

def get_stubs_from_npose( npose ):
    ns  = extract_atoms(npose, [N])
    cas = extract_atoms(npose, [CA])
    cs  = extract_atoms(npose, [C])

    return get_stubs_from_n_ca_c(ns[:,:3], cas[:,:3], cs[:,:3])


def write_pdb_info_labels(f, pdb_info_labels):
    for resnum1 in sorted(list(pdb_info_labels)):
        labels = pdb_info_labels[resnum1]
        f.write("REMARK PDBinfo-LABEL:%5i "%resnum1)
        f.write(" ".join(labels))
        f.write("\n")


_atom_record_format = (
    "ATOM  {atomi:5d} {atomn:^4}{idx:^1}{resn:3s} {chain:1}{resi:4d}{insert:1s}   "
    "{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{b:6.2f}{seg:-4d}{elem:2s}\n"
)


def format_atom(
        atomi=0,
        atomn='ATOM',
        idx=' ',
        resn='RES',
        chain='A',
        resi=0,
        insert=' ',
        x=0,
        y=0,
        z=0,
        occ=1,
        b=0,
        seg=1,
        elem=''
):
    return _atom_record_format.format(**locals())


def dump_npdb(npose, fname, atoms_present=list(range(R)), pdb_order=_pdb_order, out_file=None, pdb_info_labels={}, chains=None):
    assert(len(atoms_present) == len(pdb_order))
    local_R = len(atoms_present)
    if ( chains is None ):
        use_chains = "A"*(len(npose)//local_R)
    else:
        use_chains = chains
    out = out_file
    if ( out_file is None ):
        out = open(fname, "w")
    for ri, res in enumerate(npose.reshape(-1, local_R, 4)):
        atom_offset = ri*local_R+1
        for i, atomi in enumerate(pdb_order):
            a = res[atoms_present.index(atomi)]
            out.write( format_atom(
                atomi=(atom_offset+i)%100000,
                resn='ALA',
                resi=(ri+1)%10000,
                chain=use_chains[ri],
                atomn=_atom_names[atomi],
                x=a[0],
                y=a[1],
                z=a[2],
                ))
    write_pdb_info_labels(out, pdb_info_labels)
    if ( out_file is None ):
        out.close()


def dump_pts(pts, name):
    with open(name, "w") as f:
        for ivert, vert in enumerate(pts):
            f.write(format_atom(ivert%100000, resi=ivert%10000, x=vert[0], y=vert[1], z=vert[2]))

def dump_line(start, direction, length, name):
    dump_lines([start], [direction], length, name)

def dump_lines(starts, directions, length, name):

    starts = np.array(starts)
    if ( len(starts.shape) == 1 ):
        starts = np.tile(starts, (len(directions), 1))

    directions = np.array(directions)

    vec = np.linspace(0, length, 80)

    pt_collections = []

    for i in range(len(starts)):
        start = starts[i]
        direction = directions[i]

        pts = start + direction*vec[:,None]
        pt_collections.append(pts)

    pts = np.concatenate(pt_collections)

    dump_pts(pts, name)

def dump_lines_clustered(starts, directions, length, name, cluster_resl):
    centers, _ = slow_cluster_points(starts, cluster_resl)

    dump_lines(starts[centers], directions[centers], length, name)

def get_final_dict(score_dict, string_dict):
    final_dict = OrderedDict()
    keys_score = [] if score_dict is None else list(score_dict)
    keys_string = [] if string_dict is None else list(string_dict)

    all_keys = keys_score + keys_string

    argsort = sorted(range(len(all_keys)), key=lambda x: all_keys[x])

    for idx in argsort:
        key = all_keys[idx]

        if ( idx < len(keys_score) ):
            final_dict[key] = "%8.3f"%(score_dict[key])
        else:
            final_dict[key] = string_dict[key]

    return final_dict


def add_to_score_file(tag, fname, write_header=False, score_dict=None, string_dict=None):
    with open(fname, "a") as f:
        add_to_score_file_open(tag, f, write_header, score_dict, string_dict)

def add_to_score_file_open(tag, f, write_header=False, score_dict=None, string_dict=None):
    final_dict = get_final_dict( score_dict, string_dict )
    if ( write_header ):
        f.write("SCORE:     %s description\n"%(" ".join(final_dict.keys())))
    scores_string = " ".join(final_dict.values())
    f.write("SCORE:     %s        %s\n"%(scores_string, tag))


def add_to_silent(npose, tag, fname, write_header=False, score_dict=None, string_dict=None, pdb_info_labels={}):
    with open(fname, "a") as f:
        add_to_silent_file_open(npose, tag, f, write_header, score_dict, string_dict, pdb_info_labels)

def add_to_silent_file_open(npose, tag, f, write_header=False, score_dict=None, string_dict=None, pdb_info_labels={}):
    final_dict = get_final_dict( score_dict, string_dict )
    if ( write_header ):
        f.write("SEQUENCE: A\n")
        f.write("SCORE:     score %s description\n"%(" ".join(final_dict.keys())))
        f.write("REMARK BINARY SILENTFILE\n")

    scores_string = " ".join(final_dict.values())
    f.write("SCORE:     0.000 %s        %s\n"%(scores_string, tag))
    f.write("ANNOTATED_SEQUENCE: %s %s\n"%("A"*nsize(npose), tag))

    write_pdb_info_labels(f, pdb_info_labels)

    Hs = build_H(npose)

    by_res = npose.reshape(-1, R, 4)

    for i in range(len(by_res)):
        res_data = get_silent_res_data(np.r_[
            by_res[i,N,:3],
            by_res[i,CA,:3],
            by_res[i,C,:3],
            by_res[i,O,:3],
            by_res[i,CB,:3],
            by_res[i,CB,:3],
            Hs[i], 
         ])
        f.write("%s %s\n"%(res_data, tag))


chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"

@njit(fastmath=True, cache=True)
def code_to_6bit(byte):
    return chars[byte]

@njit(fastmath=True, cache=True)
def encode_24_to_32(i0, i1, i2):
    return code_to_6bit( i0 & 63 ) + \
            code_to_6bit( ((i1 << 2) | (i0 >> 6)) & 63 ) + \
            code_to_6bit( ((i1 >> 4) | ((i2 << 4) & 63)) & 63 ) + \
            code_to_6bit( i2 >> 2 )


float_packer = struct.Struct("f")
def get_silent_res_data(coords):

    ba = bytearray()

    for coord in coords:
        ba += float_packer.pack(coord)

    return inner_get_silent_res_data(ba)

@njit(fastmath=True, cache=True)
def inner_get_silent_res_data(ba):
    line = "L"

    iters = int(math.ceil(len(ba) / 3))

    len_ba = len(ba)
    for i in range(iters):
        i0 = 0
        i1 = 0
        i2 = 0
        i0 = ba[i*3+0]
        if ( i*3 + 1 < len_ba ):
            i1 = ba[i*3+1]
        if ( i*3+2 < len_ba ):
            i2 = ba[i*3+2]

        line += encode_24_to_32(i0, i1, i2)

    return line




def xform_to_superimpose_nposes( mobile, mobile_resnum, ref, ref_resnum ):

    mobile_stub = get_stub_from_npose(mobile, mobile_resnum)
    mobile_stub_inv = np.linalg.inv(mobile_stub)

    ref_stub = get_stub_from_npose(ref, ref_resnum)

    xform = ref_stub @ mobile_stub_inv

    return xform

def xform_npose(xform, npose):
    return (xform @ npose[...,None]).reshape(-1, 4)

def extract_atoms(npose, atoms):
    return npose.reshape(-1, R, 4)[...,atoms,:].reshape(-1,4)

def extract_N_CA_C(npose):
    indices = []
    for i in range(nsize(npose)):
        indices.append(i*R+N)
        indices.append(i*R+CA)
        indices.append(i*R+C)
    return npose[indices]

def extract_CA(npose):
    indices = np.arange(CA, nsize(npose)*R, R)
    return npose[indices]

def points_from_tpose(tpose):
    return tpose[:,:,-1]


def calc_rmsd(npose1, npose2):
    assert( len(npose1) == len(npose2))
    return math.sqrt(np.sum(np.square(np.linalg.norm(npose1[:,:3] - npose2[:,:3], axis=-1))) / ( len(npose1) ))

def tpose_from_npose( npose ):
    return get_stubs_from_npose( npose )


def itpose_from_tpose( tpose ):
    return np.linalg.inv(tpose)


def my_rstrip(string, strip):
    if (string.endswith(strip)):
        return string[:-len(strip)]
    return string


def get_tag(fname):
    name = os.path.basename(fname)
    return my_rstrip(my_rstrip(name, ".gz"), ".pdb")


# _resl = 0.5
# _atom_size = 5 - _resl
#Bounds are lb, ub, resl

#num_clashes = clash_grid.arr[tuple(clash_grid.floats_to_indices(xformed_cas).T)].sum()
def ca_clashgrid_from_npose(npose, atom_size, resl, padding=0):
    return clashgrid_from_points( extract_CA(npose), atom_size, resl, padding)

#Bounds are lb, ub, resl
def clashgrid_from_tpose(tpose, atom_size, resl, padding=0):
    return clashgrid_from_points( points_from_tpose(tpose), atom_size, resl, padding)

def clashgrid_from_points(points, atom_size, resl, padding=0, low_high=None):
    points = points[:,:3]
    if ( low_high is None ):
        low = np.min(points, axis=0) - atom_size*2 - resl*2 - padding*2
        high = np.max(points, axis=0) + atom_size*2 + resl*2 + padding*2
    else:
        low, high = low_high

    clashgrid = voxel_array.VoxelArray(low, high, np.array([resl]*3), bool)

    clashgrid.add_to_clashgrid(points, atom_size)
    # for pt in points:
    #     inds = clashgrid.indices_within_x_of(atom_size*2, pt)
    #     clashgrid.arr[tuple(inds.T)] = True

    return clashgrid


def nearest_object_grid(objects, atom_size=3.5, resl=0.25, padding=0, store=None):

    objects = objects[:,:3]

    low = np.min(objects, axis=0) - atom_size*2 - resl*2  - padding
    high = np.max(objects, axis=0) + atom_size*2 + resl*2 + padding
        
    nearest_object = voxel_array.VoxelArray(low, high, np.array([resl]*3), np.int32)
    nearest_dist = voxel_array.VoxelArray(low, high, np.array([resl]*3), np.float32)

    nearest_object.arr.fill(-1)
    nearest_dist.arr.fill(1000)

    nearest_object.add_to_near_grid(objects, atom_size, nearest_dist, store)

    return nearest_object, nearest_dist


def xforms_from_four_points(c, u, v, w):
    c = c[...,:3]
    u = u[...,:3]
    v = v[...,:3]
    w = w[...,:3]

    e1 = u - v
    e1 = e1 / np.linalg.norm(e1, axis=1)[...,None]
    e3 = np.cross( e1, w - v, axis=1)
    e3 = e3 / np.linalg.norm(e3, axis=1)[...,None]
    e2 = np.cross(e3, e1, axis=1)

    xforms = np.zeros((len(c), 4, 4))
    xforms[...,:3,0] = e1
    xforms[...,:3,1] = e2
    xforms[...,:3,2] = e3
    xforms[...,:3,3] = c
    xforms[...,3,3] = 1.0

    return xforms

def npose_to_motif_hash_frames(npose):
    by_res = npose.reshape(-1, R, 4)


    Ns = by_res[:,N]
    CAs = by_res[:,CA]
    Cs = by_res[:,C]

    CEN = np.array([-0.865810,-1.764143,1.524857, 1.0])


    #CEN = Xform().from_four_points( CA, N, CA, C ) * CEN;
    # Vec const DIR1 = C-N;
    # Vec const CEN2 = (C+N)/2;
    # return Xform().from_four_points( CEN, CEN2, CA, CA+DIR1 );

    cen = (xforms_from_four_points(CAs, Ns, CAs, Cs) @ CEN[...,None]).reshape(-1, 4)

    dir1 = Cs - Ns
    cen2 = (Cs+Ns)/2

    return xforms_from_four_points(cen, cen2, CAs, CAs+dir1)

def npose_to_will_hash_frames(npose):
    by_res = npose.reshape(-1, R, 4)


    Ns = by_res[:,N]
    CAs = by_res[:,CA]
    Cs = by_res[:,C]


    frames = xforms_from_four_points(CAs, Ns, CAs, Cs)

    avg_centroid_offset = [-0.80571551, -1.60735769, 1.46276045]

    t = frames[:,:3,:3] @ avg_centroid_offset + CAs[:, :3]

    frames[:,:3,3] = t

    return frames


def npose_to_rif_hash_frames(npose):
    by_res = npose.reshape(-1, R, 4)


    Ns = by_res[:,N,:3]
    CAs = by_res[:,CA,:3]
    Cs = by_res[:,C,:3]

    e1 = (Cs+Ns)/2 - CAs
    e1 = e1 / np.linalg.norm(e1, axis=1)[...,None]
    e3 = np.cross( e1, Cs - CAs )
    e3 = e3 / np.linalg.norm(e3, axis=1)[...,None]
    e2 = np.cross( e3, e1 )
    e2 = e2 / np.linalg.norm(e2, axis=1)[...,None]

    frames = np.zeros((len(Cs), 4, 4))
    frames[:,:3,0] = e1
    frames[:,:3,1] = e2
    frames[:,:3,2] = e3
    frames[:,3,3] = 1.0

    t = frames[:,:3,:3] @ np.array([-1.952799123558066, -0.2200069625712990, 1.524857]) + CAs
    frames[:,:3,3] = t


    return frames

def npose_to_derp_hash_frames(npose):
    by_res = npose.reshape(-1, R, 4)


    Ns = by_res[:,N,:3]
    CAs = by_res[:,CA,:3]
    Cs = by_res[:,C,:3]

    ca2n = Ns - CAs
    ca2c = Cs - CAs

    tgt1 = ca2n
    tgt2 = ca2c

    a = tgt1
    a /= np.linalg.norm(a, axis=-1)[:, None]
    c = np.cross(a, tgt2)
    c /= np.linalg.norm(c, axis=-1)[:, None]
    b = np.cross(c, a)

    stubs = np.zeros((len(by_res), 4, 4))
    stubs[:, :3, 0] = a
    stubs[:, :3, 1] = b
    stubs[:, :3, 2] = c
    stubs[:, :3, 3] = CAs
    stubs[:, 3, 3] = 1

    return stubs

def pair_xform(xform1, xform2):
    return np.linalg.inv(xform1) @ xform2

def sin_cos_range( x, tol=0.001):
    if ( x >= -1 and x <= 1 ):
        return x
    elif ( x <= -1 and x >= -( 1 + tol ) ):
        return -1
    elif ( x >= 1 and x <= 1 + tol ):
        return 1
    else:
        eprint("sin_cos_range ERROR: %.8f"%x )
        return -1 if x < 0 else 1

_xx = (0, 0)
_xy = (0, 1)
_xz = (0, 2)
_yx = (1, 0)
_yy = (1, 1)
_yz = (1, 2)
_zx = (2, 0)
_zy = (2, 1)
_zz = (2, 2)
_x = (0, 3)
_y = (1, 3)
_z = (2, 3)

_float_precision = 0.00001

# def rt6_from_xform(xform, rt6):
#     rt6[1] = xform[0,3]
#     rt6[2] = xform[1,3]
#     rt6[3] = xform[2,3]

#     if ( xform[_zz] >= 1 - _float_precision ):
#         e1 = math.atan2( sin_cos_range( xform[_yx] ), sin_cos_range( xform[_xx] ) )
#         e2 = 0
#         e3 = 0
#     elif ( xform[_zz] <= -1 + _float_precision ):
#         e1 = math.atan2( sin_cos_range( xform[_yx] ), sin_cos_range( xform[_xx] ) )
#         e2 = 0
#         e3 = math.pi
#     else:
#         pos_sin_theta = math.sqrt( 1 - xform[_zz]**2 )
#         e3 = math.asin( pos_sin_theta )
#         if ( xform[_zz] < 0 ):
#             e3 = math.pi - e3
#         e1 = math.atan2( xform[_xz], -xform[_yz])
#         e2 = math.atan2( xform[_zx],  xform[_zy])

#     if ( e1 < 0 ):
#         e1 += math.pi * 2
#     if ( e2 < 0 ):
#         e2 += math.pi * 2

#     rt6[4] = 180/math.pi*min(max(0, e1), math.pi*2-0.0000000000001)
#     rt6[5] = 180/math.pi*min(max(0, e2), math.pi*2-0.0000000000001)
#     rt6[6] = 180/math.pi*min(max(0, e3), math.pi  -0.0000000000001)


#     return rt6

def rt6_from_xform(xform, xyzTransform):

    xyzTransform.R.xx = xform[_xx]
    xyzTransform.R.xy = xform[_xy]
    xyzTransform.R.xz = xform[_xz]
    xyzTransform.R.yx = xform[_yx]
    xyzTransform.R.yy = xform[_yy]
    xyzTransform.R.yz = xform[_yz]
    xyzTransform.R.zx = xform[_zx]
    xyzTransform.R.zy = xform[_zy]
    xyzTransform.R.zz = xform[_zz]
    xyzTransform.t.x = xform[_x]
    xyzTransform.t.y = xform[_y]
    xyzTransform.t.z = xform[_z]

    return xyzTransform.rt6()


def xform_from_axis_angle_deg( axis, angle ):
    return xform_from_axis_angle_rad( axis, angle * math.pi / 180 )

def xform_from_axis_angle_rad( axis, angle ):
    xform = np.zeros((4, 4), np.float64)
    xform[3,3] = 1.0

    cos = np.cos(angle, dtype=np.float64)
    sin = np.sin(angle, dtype=np.float64)
    ux = axis[0]
    uy = axis[1]
    uz = axis[2]

    xform[0, 0] = cos + ux**2*(1-cos)
    xform[0, 1] = ux*uy*(1-cos) - uz*sin
    xform[0, 2] = ux*uz*(1-cos) + uy*sin

    xform[1, 0] = uy*ux*(1-cos) + uz*sin
    xform[1, 1] = cos + uy**2*(1-cos)
    xform[1, 2] = uy*uz*(1-cos) - ux*sin

    xform[2, 0] = uz*ux*(1-cos) - uy*sin
    xform[2, 1] = uz*uy*(1-cos) + ux*sin
    xform[2, 2] = cos + uz**2*(1-cos)

    return xform


def get_N_from_xform(xform):
    N_pos = np.array([-1.45800100, 0.00000000, 0.00000000, 1.0])
    return xform @ N_pos

def get_C_from_xform(xform):
    C_pos = np.array([0.55084745, 1.42016972, 0.00000000, 1.0])
    return xform @ C_pos

@njit(fastmath=True)
def get_CB_from_xform(xform):
    CB_pos = np.array([0.53474492, -0.76147505, -1.21079691, 1.0])
    return xform @ CB_pos

def get_phi_vector(xform):
    N_pos = get_N_from_xform(xform)
    return xform[:3,3] - N_pos[:3]

def get_psi_vector(xform):
    C_pos = get_C_from_xform(xform)
    return C_pos[:3] - xform[:3,3]

def get_phi_rotation_xform(xform, angle_deg, ca):
    vec = get_phi_vector(xform)
    vec /= np.linalg.norm(vec)
    rotate_xform = xform_from_axis_angle_deg(vec, angle_deg)
    trans = ca[:3] + (rotate_xform @ -ca)[:3]
    rotate_xform[:3,3] = trans
    return rotate_xform

def get_psi_rotation_xform(xform, angle_deg, ca):
    vec = get_psi_vector(xform)
    vec /= np.linalg.norm(vec)
    rotate_xform = xform_from_axis_angle_deg(vec, angle_deg)
    trans = ca[:3] + (rotate_xform @ -ca)[:3]
    rotate_xform[:3,3] = trans
    return rotate_xform

def apply_dihedral_to_points(points, xform, start_pos):
    unaffected = points[:start_pos]
    modified = xform_npose(xform, points[start_pos:])
    return np.concatenate([unaffected, modified])

def apply_dihedral_to_xforms(xforms, xform, start_pos):
    unaffected = xforms[:start_pos]
    modified = xform @ xforms[start_pos:]
    return np.concatenate([unaffected, modified])

def rotate_npose_phi(npose, tpose, resno, delta_phi):
    phi_xform = get_phi_rotation_xform(tpose[resno], delta_phi, npose[resno*R+CA])
    return apply_dihedral_to_points(npose, phi_xform, resno*R+CA)

def rotate_tpose_phi(tpose, resno, delta_phi):
    phi_xform = get_phi_rotation_xform(tpose[resno], delta_phi, tpose[resno,:,3])
    return apply_dihedral_to_xforms(tpose, phi_xform, resno)    # This affects resno xform

def rotate_npose_psi(npose, tpose, resno, delta_psi):
    psi_xform = get_psi_rotation_xform(tpose[resno], delta_psi, npose[resno*R+CA])
    return apply_dihedral_to_points(npose, psi_xform, resno*R+C)

def rotate_tpose_psi(tpose, resno, delta_psi):
    psi_xform = get_psi_rotation_xform(tpose[resno], delta_psi, tpose[resno,:,3])
    return apply_dihedral_to_xforms(tpose, psi_xform, resno+1)

def set_npose_phi(npose, tpose, phis, resno, phi):
    return set_phi(npose, tpose, phis, resno, phi, R, CA, CA, 0)

def set_ca_phi(points, tpose, phis, resno, phi):
    return set_phi(points, tpose, phis, resno, phi, 1, 0, 0, 0)    # This affects resno

def set_phi(points, tpose, phis, resno, phi, local_R, R_off, R_CA, t_off):
    delta_phi = phi - phis[resno]
    new_phis = phis.copy()
    new_phis[resno] = phi

    phi_xform = get_phi_rotation_xform(tpose[resno], delta_phi, points[resno*local_R+R_CA])
    new_points = apply_dihedral_to_points(points, phi_xform, resno*local_R+R_off)
    new_tpose = apply_dihedral_to_xforms(tpose, phi_xform, resno+t_off)

    return new_points, new_tpose, new_phis

def set_npose_psi(npose, tpose, psis, resno, psi):
    return set_psi(npose, tpose, psis, resno, psi, R, C, CA, 1)

def set_ca_psi(points, tpose, psis, resno, psi):
    return set_psi(points, tpose, psis, resno, psi, 1, 1, 0, 1)

def set_psi(points, tpose, psis, resno, psi, local_R, R_off, R_CA, t_off):
    delta_psi = psi - psis[resno]
    new_psis = psis.copy()
    new_psis[resno] = psi

    psi_xform = get_psi_rotation_xform(tpose[resno], delta_psi, points[resno*local_R+R_CA])
    new_points = apply_dihedral_to_points(points, psi_xform, resno*local_R+R_off)
    new_tpose = apply_dihedral_to_xforms(tpose, psi_xform, resno+t_off)

    return new_points, new_tpose, new_psis


def get_dihedral(atom1, atom2, atom3, atom4):
    a = atom2 - atom1
    a /= np.linalg.norm(a)
    b = atom3 - atom2
    b /= np.linalg.norm(b)
    c = atom4 - atom3
    c /= np.linalg.norm(c)

    x = -np.dot( a, c ) + ( np.dot( a, b ) * np.dot( b, c) )
    y = np.dot( a, np.cross( b, c ) )

    angle = 0 if ( y == 0 and x == 0 ) else math.atan2( y, x )

    return angle

def get_dihedrals(atom1, atom2, atom3, atom4):

    a = atom2 - atom1
    a /= np.linalg.norm(a, axis=-1)[:,None]
    b = atom3 - atom2
    b /= np.linalg.norm(b, axis=-1)[:,None]
    c = atom4 - atom3
    c /= np.linalg.norm(c, axis=-1)[:,None]

    x = -dot( a, c ) + ( dot( a, b ) * dot( b, c) )
    y = dot( a, np.cross( b, c ) )

    return np.arctan2( y, x )


def get_npose_phis(npose):
    npose_by_res = npose.reshape(-1,R,4)

    phis = np.zeros(len(npose_by_res), np.float64)

    phis[1:] = np.degrees( get_dihedrals( npose_by_res[:-1,C,:3],
                                          npose_by_res[1:,N,:3],
                                          npose_by_res[1:,CA,:3],
                                          npose_by_res[1:,C,:3]
                                          ))
    return phis


def get_npose_psis(npose):
    npose_by_res = npose.reshape(-1,R,4)

    psis = np.zeros(len(npose_by_res), np.float64)

    psis[:-1] = np.degrees( get_dihedrals( npose_by_res[:-1,N,:3],
                                          npose_by_res[:-1,CA,:3],
                                          npose_by_res[:-1,C,:3],
                                          npose_by_res[1:,N,:3]
                                          ))
    return psis


def npose_abego(npose):
    phis = get_npose_phis(npose)
    psis = get_npose_psis(npose)

    abegos = np.zeros(len(phis), 'U1')

    abegos[ ( phis < 0 ) &  (( -75 <= psis ) & ( psis < 50 )) ] = "A"
    abegos[ ( phis < 0 ) & ~(( -75 <= psis ) & ( psis < 50 )) ] = "B"

    abegos[ ( phis >= 0 ) & ~(( -100 <= psis ) & ( psis < 100 )) ] = "E"
    abegos[ ( phis >= 0 ) &  (( -100 <= psis ) & ( psis < 100 )) ] = "G"

    # we don't do O

    return abegos



# If unit vector is specified. Only points with positive dot products are kept
def prepare_context_by_dist_and_limits(context, pt, max_dist, unit_vector=None):
    pt = pt[:3]
    if ( not unit_vector is None ):
        vectors = context - pt
        dots = np.sum( np.multiply(vectors, unit_vector), axis=1)
        context = context[dots > 0]
    if ( len(context) == 0):
        context = np.array([[1000, 1000, 1000]])
    dists = np.linalg.norm( pt - context, axis=1 )
    context_dists = zip(context, dists)
    context_dists = sorted(context_dists, key=lambda x: x[1])
    context_by_dist, dists = zip(*context_dists)
    context_by_dist = np.array(context_by_dist)


    pos = 0
    context_dist_limits = []
    for dist in range(int(max_dist)+1):
        while ( pos < len(context_by_dist) and dists[pos] < dist ):
            pos += 1
        context_dist_limits.append(pos)

    context_dist_limits = np.array(context_dist_limits)

    return context_by_dist, context_dist_limits

# def clash_check_points_context(pts, point_dists, context_by_dist, context_dist_limits, clash_dist, max_clash):
#     if ( context_by_dist is None):
#         return 0
#     return jit_clash_check_points_context(pts, point_dists, context_by_dist, context_dist_limits, clash_dist, max_clash)

@njit(fastmath=True)
def clash_check_points_context(pts, point_dists, context_by_dist, context_dist_limits, clash_dist, max_clash, tol=0):
    clashes = 0
    clash_dist2 = clash_dist * clash_dist
    pts = pts[:,:3]
    for ipt in range(len(pts)):
        pt = pts[ipt]
        lo_limit = context_dist_limits[max(0, int(point_dists[ipt] - clash_dist - 1 - tol))]
        limit = context_dist_limits[int(point_dists[ipt] + clash_dist + tol)]
        context = context_by_dist[lo_limit:limit]
        clashes += np.sum( np.sum( np.square( pt - context ), axis=1 ) < clash_dist2 )

        if ( clashes >= max_clash ):
            return clashes
    return clashes


def xform_magnitude_sq_fast( trans_err2, traces, lever2 ):

    # trans_part = rts[...,:3,3]
    # err_trans2 = np.sum(np.square(trans_part), axis=-1)

    # rot_part = rts[...,:3,:3]
    # traces = np.trace(rot_part,axis1=-1,axis2=-2)
    cos_theta = ( traces - 1 ) / 2

    # We clip to 0 here so that negative cos_theta gets lever as error
    clipped_cos = np.clip( cos_theta, 0, 1)

    err_rot2 = ( 1 - np.square(clipped_cos) ) * lever2

    # err = np.sqrt( err_trans2 + err_rot2 )
    err =  trans_err2 + err_rot2 

    return err


def xform_magnitude_sq( rts, lever2 ):

    trans_part = rts[...,:3,3]
    err_trans2 = np.sum(np.square(trans_part), axis=-1)

    rot_part = rts[...,:3,:3]
    traces = np.trace(rot_part,axis1=-1,axis2=-2)
    cos_theta = ( traces - 1 ) / 2

    # We clip to 0 here so that negative cos_theta gets lever as error
    clipped_cos = np.clip( cos_theta, 0, 1)

    err_rot2 = ( 1 - np.square(clipped_cos) ) * lever2

    # err = np.sqrt( err_trans2 + err_rot2 )
    err =  err_trans2 + err_rot2 

    return err

def xform_magnitude( rts, lever2 ):

    return np.sqrt( xform_magnitude_sq( rts, lever2 ) )

#a00 a01 a02 a03
#a10 a11 a12 a13
#a20 a21 a22 a23
#a30 a31 a32 a33

#b00 b01 b02 b03
#b10 b11 b12 b13
#b20 b21 b22 b23
#b30 b31 b32 b33

# c = a @ b
#
# c00 = b00 a00 * b10 a01 * b20 a02 * b30 a03

# c11 = b01 a10 * b11 a11 * b21 a12 * b31 a13

# c22 = b02 a20 * b12 a21 * b22 a22 * b32 a23

# c03 = b03 a00 * b13 a01 * b23 a02 * b33 a03
# c13 = b03 a10 * b13 a11 * b23 a12 * b33 a13
# c23 = b03 a20 * b13 a21 * b23 a22 * b33 a23

@njit(fastmath=True)
def mm2(inv_xform, xforms, traces, trans_err2):

    a = inv_xform
    b = xforms

    # leaving out the 4th term because we know it's 0
    traces[:] = np.sum( a[0,:3] * b[:,:3,0], axis=-1 )
    traces += np.sum( a[1,:3] * b[:,:3,1], axis=-1 )
    traces += np.sum( a[2,:3] * b[:,:3,2], axis=-1 )

    # we know the 4th term here has a 1 in b
    trans_err2[:] = np.square(np.sum( a[0,:3] * b[:,:3,3], axis=-1) + a[0,3])
    trans_err2 += np.square(np.sum( a[1,:3] * b[:,:3,3], axis=-1) + a[1,3])
    trans_err2 += np.square(np.sum( a[2,:3] * b[:,:3,3], axis=-1) + a[2,3])


def mm1(inverse_xforms, cur_index, xforms, out):
    np.matmul(inverse_xforms[cur_index], xforms, out=out)

# This would be better if it found the center of each cluster
# This requires nxn of each cluster though
def cluster_xforms( close_thresh, lever, xforms, inverse_xforms = None, info_every=None ):

    if ( xforms.dtype != np.float32 ):
        xforms = xforms.astype(np.float32)

    if ( inverse_xforms is None ):
        inverse_xforms = np.linalg.inv(xforms)
    else:
        if ( inverse_xforms.dtype != inverse_xforms ):
            inverse_xforms = inverse_xforms.astype(np.float32)

    size = len(xforms)
    min_distances = np.zeros(size, float)
    min_distances.fill(9e9)
    assignments = np.zeros(size, int)
    center_indices = []

    lever2 = lever*lever

    cur_index = 0

    traces = np.zeros(len(xforms), dtype=np.float32)
    trans_err2 = np.zeros(len(xforms), dtype=np.float32)

    out = np.zeros((len(xforms), 4, 4), dtype=np.float32)

    while ( np.sqrt(min_distances.max()) > close_thresh ):

        # mm1(inverse_xforms, cur_index, xforms, out)
        # distances1 = xform_magnitude_sq( out, lever2 )
        # distances = xform_magnitude_sq( inverse_xforms[cur_index] @ xforms, lever2 )

        mm2(inverse_xforms[cur_index], xforms, traces, trans_err2)
        distances = xform_magnitude_sq_fast( trans_err2, traces, lever2 )

        # assert((np.abs(distances1 - distances) < 0.01).all())

        changes = distances < min_distances
        assignments[changes] = len( center_indices )
        min_distances[changes] = distances[changes]

        center_indices.append( cur_index )
        cur_index = min_distances.argmax()

        if ( not info_every is None ):
            if ( len(center_indices) % info_every == 0 ):
                print("Cluster round %i: max_dist: %6.3f  "%(len(center_indices), np.sqrt(min_distances[cur_index])))

    return center_indices, assignments

def slow_cluster_points(points, distance, info_every=None):

    as_xforms = np.tile(np.identity(4), (len(points), 1, 1))
    as_xforms[:,:3,3] = points[:,:3]

    return cluster_xforms( distance, 3, as_xforms, info_every )


# This would be better if it found the center of each cluster
# This requires nxn of each cluster though
@njit(fastmath=True, cache=True)
def cluster_points( points, close_thresh, find_centers=False ):

    size = len(points)
    min_distances = np.zeros(size, np.float_)
    min_distances.fill(9e9)
    assignments = np.zeros(size, np.int_)
    center_indices = []

    cur_index = 0

    # trans_err2 = np.zeros(len(points), dtype=np.float32)
    
    ct = 0

    while ( min_distances.max() > close_thresh**2 ):
        
        ct += 1

        distances = np.sum( np.square( points - points[cur_index] ), axis=-1 )

        changes = distances < min_distances
        assignments[changes] = len( center_indices )
        min_distances[changes] = distances[changes]

        center_indices.append( cur_index )
        cur_index = min_distances.argmax()
        
        if ct > 1000:
            break

        # if ( not info_every is None ):
        #     if ( len(center_indices) % info_every == 0 ):
        #         print("Cluster round %i: max_dist: %6.3f  "%(len(center_indices), np.sqrt(min_distances[cur_index])))

    center_indices = np.array(center_indices, np.int_)
    
    if len(center_indices) > 0:

        if ( find_centers ):
            for icluster in range(len(center_indices)):
                mask = assignments == icluster
                who = np.where(mask)[0]
                members = points[mask]

                min_dist = 9e9
                imin_dist = 0
                for i in range(len(members)):
                    dist_sum = np.sum( np.square( members[i] - members) )
                    if ( dist_sum < min_dist ):
                        min_dist = dist_sum
                        imin_dist = i

                center_indices[icluster] = who[imin_dist]


    return center_indices, assignments


def get_clusters(assignments, num_clusters):
    clusters = []
    for i in range(num_clusters):
        clusters.append([])
    
    for i in range(len(assignments)):
        clusters[assignments[i]].append(i)

    return clusters


def center_of_mass( coords ):
    com = np.sum( coords, axis=-2 ) / coords.shape[-2]
    return com

def radius_of_gyration( coords, com=None):
    if (com is None):
        com = center_of_mass(coords)

    # The extra 1s will cancel here
    dist_from_com2 = np.square(np.sum( coords - com, axis=-1))

    return np.sqrt( np.sum(dist_from_com2) / coords.shape[-2] )

def xform_from_flat( twelve ):
    xform = np.identity(4)
    xform[:3,:3].flat = twelve[:9]
    xform[:3,3].flat = twelve[9:]
    return xform

def flat_from_xform( xform ):
    return list(xform[:3,:3].flat) + list(xform[:3,3].flat)


# assumes flat12 format
def load_xforms(file):
    xforms = []
    with open(file) as f:
        for line in f:
            line = line.strip()
            sp = line.split()
            flat = [float(x) for x in sp]
            xform = xform_from_flat(flat)
            xforms.append(xform)
    return np.array(xforms)

def save_xforms(file, xforms):
    with open(file, "w") as f:
        for xform in xforms:
            f.write(" ".join("%12.8f"%x for x in flat_from_xform(xform)))
            f.write("\n")

def skew(rots):
    return 1/2 * ( rots - np.transpose( rots, axes=[0, 2, 1] ) )

def cay(rots):
    idents = np.tile(np.identity(3), (len(rots), 1, 1))

    return np.linalg.inv( idents + rots ) @ ( idents - rots )

def get_normed_rotations(rots):

    return cay(skew(cay(rots)))

    # return cay( ( np.transpose( rots, axes=[0, 2, 1] ) - rots ) / ( 1 + np.trace( rots, axis1=-1, axis2=-2 ) )[:,None,None])

def _F(width, max_decimals, x):
    try:
        whole_size = int(np.log10(x)) + 1
        if ( whole_size + 2 > width ):
            fmt = "%i"
        else:
            decimals = width - whole_size - 1
            fmt = "%%.%if"%(decimals)
        return fmt%x
    except:
        return str(x) # nan and stuff


def KMGT(x, w=3, d=1):
    if( x < 1e3  ): return _F( w, d, x/1e0  );
    if( x < 1e6  ): return _F( w, d, x/1e3  )+"K";
    if( x < 1e9  ): return _F( w, d, x/1e6  )+"M";
    if( x < 1e12 ): return _F( w, d, x/1e9  )+"G";
    if( x < 1e15 ): return _F( w, d, x/1e12 )+"T";
    if( x < 1e18 ): return _F( w, d, x/1e15 )+"P";
    if( x < 1e21 ): return _F( w, d, x/1e18 )+"E";
    if( x < 1e24 ): return _F( w, d, x/1e21 )+"Z";
    else:           return _F( w, d, x/1e24 )+"Y";


def linear_regression( x, y ):

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    dx = x - x_mean
    dy = y - y_mean

    slope = np.sum( dx*dy ) / np.sum( np.square(dx) )
    intercept = y_mean - slope * x_mean

    return slope, intercept

@njit(fastmath=True, cache=True)
def _fill_in_gaps(array):
    for i in range(1, len(array)-1):
        if ( array[i-1] == array[i+1] ):
            if ( array[i] != array[i-1] ):
                array[i] = array[i-1]
        if ( i < len(array) - 3 ):
            if ( array[i-1] == array[i+2] ):
                if ( array[i] != array[i-1] ):
                    array[i] = array[i-1]

# currently only supports loops and helices
def npose_dssp_helix(npose, hbond_cutoff=-0.1):

    Hs = build_H( npose )[:,:3]
    Ns = extract_atoms( npose, [N])[:,:3]
    Cs = extract_atoms( npose, [C])[:,:3]
    Os = extract_atoms( npose, [O])[:,:3]

    don_rays = Hs - Ns
    don_rays /= np.linalg.norm( don_rays, axis=-1 )[:,None]

    acc_rays = Os - Cs
    acc_rays /= np.linalg.norm( acc_rays, axis=-1 )[:,None]

    hbond_i_ip4 = fast_hbond( Hs[4:], don_rays[4:], Os[:-4], acc_rays[:-4], 0.5 ) < hbond_cutoff

    # is_hbond = hbond_i_ip4 < -0.2

    abegos = npose_abego(npose)


    is_forward = np.zeros(nsize(npose), np.bool)
    is_backward = np.zeros(nsize(npose), np.bool)

    temp = np.zeros(nsize(npose), np.bool)

    is_helix = np.zeros(nsize(npose), np.bool)

    is_forward[:-4] = hbond_i_ip4 & ( abegos[:-4] == 'A' )
    is_backward[4:] = hbond_i_ip4 & ( abegos[4:] == 'A' )

    _fill_in_gaps( is_forward )
    _fill_in_gaps( is_backward )

    state = -1
    for i in range(1, len(is_helix)):
        # before helix
        if ( state == -1 ):
            if ( is_forward[i] ):
                state = 0
                is_helix[i] = True
        elif ( state == 0 ):
            if ( not is_forward[i] ):
                state = 1
            else:
                is_helix[i] = True
            if ( is_backward[i] ):
                is_helix[i] = True
        else:
            if ( is_backward[i] and not is_forward[i] ):
                is_helix[i] = True
            else:
                state = -1


    n_consensus = is_helix[3:6].mean() > 0.5
    c_consensus = is_helix[-6:-3].mean() > 0.5

    is_helix[:6] = n_consensus
    is_helix[-6:] = c_consensus

    return is_helix


def dot(a, b):
    return np.einsum('ij,ij->i', a, b)

def fast_hbond(donor_hs, donor_rays, acceptors, acceptor_rays, extra_length=0):

    h_to_a = acceptors - donor_hs

    # dump_lines(donor_hs, h_to_a, 1, "test.pdb")

    h_to_a_len = np.linalg.norm( h_to_a, axis=-1 )
    h_to_a /= h_to_a_len[:,None]

    h_dirscore = dot( donor_rays, h_to_a ).clip(0, 1)

    a_dirscore = (dot( acceptor_rays, h_to_a ) * -1).clip( 0, 1 )

    diff = h_to_a_len - 2.00

    # if diff < 0:
    #     diff *= 1.5
    diff += ( diff * 0.5 ).clip(None, 0)

    # if diff > 0:
    #     diff = ( diff - extra_length ).clip(0, None)

    temp = (diff - extra_length).clip(0, None)
    diff = np.minimum( diff, temp )

    max_diff = 0.8

    score = np.square( 1 - np.square( diff / max_diff ).clip(None, 1) ) * -1

    dirscore = h_dirscore * h_dirscore * a_dirscore

    return score * dirscore


def npose_helix_elements(is_helix):
    ss_elements = []

    offset = 0
    ilabel = -1
    for label, group in itertools.groupby(is_helix):
        ilabel += 1
        this_len = sum(1 for _ in group)
        next_offset = offset + this_len

        ss_elements.append( (label, offset, next_offset-1))

        offset = next_offset
    return ss_elements


# This is written sort of funky so that it only reads the file once
#  and is compatible with silentdd
def nposes_from_silent(fname, chains=False, aa=False):
    import silent_tools

    _, f = silent_tools.assert_is_silent_and_get_scoreline(fname, return_f=True)

    first = True
    line = ""
    while ( not line is None and not line.startswith("SCORE") ):
        try:
            line = next(f)
        except:
            line = None


    nposes = []
    sequences = []
    the_chains = []
    tags = []


    while ( not line is None ):

        while ( not line is None and not line.startswith("SCORE") ):
            try:
                line = next(f)
            except:
                line = None

        if ( line is None ):
            break

        structure, line = silent_tools.rip_structure_by_lines(f, line )

        tag = structure[0].split()[-1]
        if ( tag == "description" ):
            continue

        tags.append( tag )

        if ( first ):
            silent_type = silent_tools.detect_silent_type(structure)

            is_binary = silent_type == "BINARY"
            is_protein = silent_type == "PROTEIN"

            assert( is_binary ^ is_protein )
            first = False


        if ( aa ):
            sequences.append("".join(silent_tools.get_sequence_chunks( structure, tag=tags[-1] )))

        if ( is_binary ):
            ncaco = silent_tools.sketch_get_atoms(structure, [0, 1, 2, 3]).reshape(-1, 4, 3)

        if ( is_protein ):
            ncac = silent_tools.sketch_get_ncac_protein_struct(structure).reshape(-1, 3, 3)
            # print(ncac[0])
            ncac_for_o = np.ones((len(ncac)*3, 4), np.float)
            ncac_for_o[:,:3] = ncac.reshape(-1, 3)
            ncaco = np.zeros((len(ncac), 4, 3), np.float)
            ncaco[:,:3,:] = ncac
            ncaco[:,3,:] = build_O_ncac(ncac_for_o)


        npose_by_res = np.ones((len(ncaco), R, 4), np.float)

        for atom in ATOM_NAMES:
            if ( atom == "CB" ):
                tpose = get_stubs_from_n_ca_c(ncaco[:,0], ncaco[:,1], ncaco[:,2])
                cbs = build_CB(tpose)
                npose_by_res[:,CB,:3] = cbs[:,:3]
            elif ( atom == "N" ):
                npose_by_res[:,N,:3] = ncaco[:,0]
            elif ( atom == "CA" ):
                npose_by_res[:,CA,:3] = ncaco[:,1]
            elif ( atom == "C" ):
                npose_by_res[:,C,:3] = ncaco[:,2]
            elif ( atom == "O" ):
                npose_by_res[:,O,:3] = ncaco[:,3]

        # ok, so protein silent files aren't really supported, we can only get CA right now
        # if ( is_protein ):


        npose = npose_by_res.reshape(-1, 4)

        nposes.append(npose)

        if ( chains ):
            the_chains.append(silent_tools.get_chain_ids(structure, tag))

    f.close()

    to_ret = [nposes, tags]
    if ( chains ):
        to_ret.append(the_chains)
    if ( aa ):
        to_ret.append(sequences)
    return to_ret









