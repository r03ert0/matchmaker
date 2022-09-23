import subprocess as sp
import sys
import igl
import trimesh
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.append("bin/matchmesh/")
import matchmesh2 as mm

# Iterative closest point algorithm
def icp_step(_v_mov, _v_ref, _f_ref):
    _, _, v_out = igl.point_mesh_squared_distance(_v_mov, _v_ref, _f_ref)

    C = np.zeros((3,3))
    oout=np.mean(v_out, axis=0)
    omov=np.mean(_v_mov, axis=0)
    for i in range(len(_v_mov)):
        C = C + np.outer(_v_mov[i,:]-omov, v_out[i,:]-oout)
    U,_,VT = np.linalg.svd(C)
    R = U.dot(VT)
    # R = V.T.dot(U.T)
    t = omov - oout.dot(R)
    _v_mov = _v_mov.dot(R)
    _v_mov -= t

    distance = np.arccos((np.trace(R)-1)/2.0)
    distance += np.sum(t**2)
    return _v_mov, distance

def icp(_v_ref, _f_ref, _v_mov, maxiter=100):
    tol = 1e-6
    distance0 = -1
    for i in range(maxiter):
        _v_mov, distance = icp_step(_v_mov, _v_ref, _f_ref)
        # print(distance)
        diff = distance if distance0<0 else np.abs(distance-distance0)
        if diff<tol:
          break
        distance0 = distance
    # print("final distance: %g (after %i iterations)"%(distance, i))

    return _v_mov, distance

from sklearn.cluster import spectral_clustering
def remove_olfactory_bulbs(V, F):
    A = igl.adjacency_matrix(F)
    labels = spectral_clustering(A)
    loop_lengths = []
    for label in np.unique(labels):
        fpatch = np.array([f for f in F if (
            labels[f[0]]!=label
            and labels[f[1]]!=label
            and labels[f[2]]!=label
        )])
        loop = igl.boundary_loop(fpatch)
        loop_lengths.append((len(loop), label))
    sorted_labels = np.sort(np.array(loop_lengths, dtype=[('length', int), ('index', int)]), axis=0, order='length')
    l1 = sorted_labels[0][1]
    l2 = sorted_labels[1][1]
    fpatch = np.array([f for f in F if (
        labels[f[0]]!=l1
        and labels[f[1]]!=l1
        and labels[f[2]]!=l1
        and labels[f[0]]!=l2
        and labels[f[1]]!=l2
        and labels[f[2]]!=l2
    )])
    v0, f0, _, _ = igl.remove_unreferenced(V, fpatch)
    loops = igl.all_boundary_loop(f0)
    new_verts = []
    for l in loops:
        mean = np.mean(v0[l], axis=0)
        new_verts.append(mean)
    f1 = igl.topological_hole_fill(f0, np.array([0]), loops)
    v1 = np.row_stack([v0, new_verts])

    return v1, f1

def process_mesh(input_mesh_path):
    tmp1 = input_mesh_path.replace("ply", "tmp1.ply")
    tmp2 = input_mesh_path.replace("ply", "tmp2.ply")
    output_mesh = input_mesh_path.replace("ply", "sphere.ply")

    print("convert to sphere")
    out = sp.run([
        "bin/meshparam/meshparam_mac",
        "-i", input_mesh_path,
        "-o", tmp1
    ], capture_output=True)
    print(out.stdout.decode("utf-8") )

    print("improve the sphere")
    out = sp.run([
        "bin/meshgeometry/meshgeometry_mac",
        "-i", tmp1,
        "-normalise",
        "-scale", "0.01",
        "-sphereLaplaceSmooth", "1", "5000",
        "-o", tmp2
    ], capture_output=True)
    print(out.stdout.decode("utf-8") )

    out = sp.run([
        "node", "bin/homogeneous/homogeneous.js",
        "-i", tmp2,
        "--alpha=0.3", "--tau=0.5", "--aspectRatio=25", "--niter=5000",
        "-o", output_mesh
    ], capture_output=True)
    print(out) #.stdout.decode("utf-8") )

def energy(sph):
    smoothE = mm.smooth_energy(sph) # smoothE is a scalar    
    defE = mm.deformation_smooth_energy(sph) # a scalar    
    projectE = mm.project_energy(sph)
    E = 1*smoothE + 10*defE + 1*projectE

    return E

# remove olfactory bulbs from mov
Vmov0, Fmov0 = igl.read_triangle_mesh("test_data/F21_P16_seg.ply")
Vmov, Fmov = remove_olfactory_bulbs(Vmov0, Fmov0)
igl.write_triangle_mesh("test_data/no-ob.ply", Vmov, Fmov, force_ascii=False)

# remesh mov without olfactory bulbs
out = sp.run([
    "bin/GraphiteThree/build/Darwin-clang-dynamic-Release/bin/vorpalite",
    "profile=scan",
    "sys:ascii=true",
    f"pts={len(Vmov)}",
    "test_data/no-ob.ply",
    "test_data/no-ob.remeshed.ply"
], capture_output=True)
print(out)

process_mesh("test_data/no-ob.remeshed.ply")
Vmov, Fmov = igl.read_triangle_mesh("test_data/no-ob.remeshed.ply")

process_mesh("test_data/F115_P16_Nissl_x20_mesh.ply")
Vref, Fref = igl.read_triangle_mesh("test_data/F115_P16_Nissl_x20_mesh.ply")

# scale mov to match ref
vol_r = trimesh.Trimesh(vertices=Vref, faces=Fref).volume
vol_m = trimesh.Trimesh(vertices=Vmov, faces=Fmov).volume
scale = (vol_m/vol_r)**(1/3)
Vmov *= 1/scale
print("scale:", 1/scale)

# center
ref_mean = np.mean(Vref, axis=0)
mov_mean = np.mean(Vmov, axis=0)
Vref = Vref - ref_mean
Vmov = Vmov - mov_mean
print("ref_mean:", ref_mean)
print("mov_mean:", mov_mean)

# try all 24 axis-aligned rotation matrices to find the best one
rotations = []
for xaxis in [0, 1, 2]:
    arr = [0, 1, 2]
    arr.remove(xaxis)
    for yaxis in arr:
        for xsign in [-1, 1]:
            for ysign in [-1, 1]:
                x = np.zeros(3)
                x[xaxis] = xsign
                y = np.zeros(3)
                y[yaxis] = ysign
                z = np.cross(x, y)
                r = np.column_stack([x, y, z])
                rotations.append(r)
min_dist = float("inf")
for rot in rotations:
    _, dist = icp(Vref, Fref, Vmov@rot, maxiter=5)
    print(dist, *rot)
    if dist<min_dist:
        min_dist=dist
        min_rot=rot
Vmov = Vmov@min_rot
print("mov rotation:", min_rot)

igl.write_triangle_mesh("test_data/scaled_centered_ref.ply", Vref, Fref, force_ascii=False)
igl.write_triangle_mesh("test_data/scaled_centered_mov.ply", Vmov, Fmov, force_ascii=False)

Sref, Fref = igl.read_triangle_mesh("test_data/F115_P16_Nissl_x20_mesh.sphere.ply")
Smov, Fmov = igl.read_triangle_mesh("test_data/no-ob.remeshed.sphere.ply")

# orient Smov as Sref
mm.preparation((Vref, Fref, Vmov, Fmov,
    Sref, Fref, Smov, Fmov,
    Smov, Fmov))
min_rot = None
min_energy = energy(Smov)
print("initial", min_energy)
for _ in range(1000):
    rot = R.random().as_matrix()
    e = energy(Smov@rot)
    if e < min_energy:
        min_energy = e
        min_rot = rot
print("final", min_energy)
print(rot, np.linalg.det(min_rot))
Smov = Smov@min_rot
# igl.write_triangle_mesh("test_data/rot.sphere.ply", Smov, Fmov, force_ascii=False)

mm.preparation((Vref, Fref, Vmov, Fmov,
    Sref, Fref, Smov, Fmov,
    Smov, Fmov))

x = mm.matchmesh((Vref, Fref, Vmov, Fmov,
    Sref, Fref, Smov, Fmov,
    Smov, Fmov))
sph = mm.sphere(x)

igl.write_triangle_mesh("test_data/result.mov.ply", Vmov, Fmov, force_ascii=False)
igl.write_triangle_mesh("test_data/result.mov-morph.sphere.ply", sph, Fmov, force_ascii=False)
igl.write_triangle_mesh("test_data/result.mov-morph.ply", mm.project(sph), Fmov, force_ascii=False)
