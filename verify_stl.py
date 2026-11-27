"""Re-import the exported STL files and sanity-check them for printing.

    blender --background --factory-startup --python verify_stl.py
"""
import bpy, bmesh, os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
FILES = ["aardwark_scoop_body.stl", "aardwark_scoop_inlay_logo.stl", "aardwark_scoop_inlay_text.stl"]


def load(fname):
    bpy.ops.wm.stl_import(filepath=os.path.join(OUT, fname))
    ob = bpy.context.selected_objects[0]
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-4)   # STL has no shared vertices
    bm.to_mesh(ob.data)
    bm.free()
    return ob


def stats(ob):
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    nonman = sum(1 for e in bm.edges if not e.is_manifold)
    vol = bm.calc_volume(signed=True)
    zs = [v.co.z for v in bm.verts]
    xs = [v.co.x for v in bm.verts]
    ys = [v.co.y for v in bm.verts]
    tris = sum(len(f.verts) - 2 for f in bm.faces)
    bm.free()
    return nonman, vol, (min(xs), max(xs)), (min(ys), max(ys)), (min(zs), max(zs)), tris


def intersect_volume(a, b):
    tmp = bpy.data.objects.new("tmp", a.data.copy())
    bpy.context.scene.collection.objects.link(tmp)
    mod = tmp.modifiers.new("i", "BOOLEAN")
    mod.operation, mod.object, mod.solver = "INTERSECT", b, "MANIFOLD"
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(tmp.evaluated_get(dg))
    bm = bmesh.new()
    bm.from_mesh(me)
    v = bm.calc_volume(signed=True) if len(bm.faces) else 0.0
    bm.free()
    bpy.data.objects.remove(tmp)
    return v


bpy.ops.wm.read_factory_settings(use_empty=True)
objs = {}
for f in FILES:
    ob = load(f)
    objs[f] = ob
    nm, vol, xr, yr, zr, tris = stats(ob)
    print(f"[verify] {f}: triangles={tris} nonmanifold_edges={nm} volume={vol / 1000:.3f} cm3 "
          f"size={xr[1] - xr[0]:.2f} x {yr[1] - yr[0]:.2f} x {zr[1] - zr[0]:.2f} mm  z={zr[0]:.2f}..{zr[1]:.2f}")
body = objs[FILES[0]]
for f in FILES[1:]:
    ov = intersect_volume(body, objs[f])
    print(f"[verify] overlap body/{f}: {ov:.4f} mm3 (should be ~0 -> inlay sits exactly in its recess)")
