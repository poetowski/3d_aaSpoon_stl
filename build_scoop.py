"""
Aardwark coffee scoop - a JURA-style measuring scoop for pre-ground coffee
(mleta kava) with the aardwark logo on the handle and a maker's mark underneath.

Parametric Blender build script (tested with Blender 5.2).

    blender --background --factory-startup --python build_scoop.py -- [--render]

Everything is modelled in millimetres (1 Blender unit = 1 mm).
Results are written to ./output/
"""
import bpy, bmesh, math, os, re, sys
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
ARGV = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
OUT = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)
DO_RENDER = "--render" in ARGV


def log(*a):
    print("[scoop]", *a, flush=True)


# ----------------------------------------------------------------- parameters
# Bowl - the rim plane is z = 0, the bowl hangs below it (use orientation).
RIM_R = 21.5        # inner radius at the rim (outer rim diameter = 2*(RIM_R+WALL))
WALL = 2.0          # wall / floor thickness
ARC_R = 18.0        # upper wall arc radius (wall is vertical at the rim)
WALL_ANG = 35.0     # deg from vertical where the arc becomes a straight cone
FILLET_R = 6.0      # inner floor fillet radius
TARGET_ML = 19.5    # level capacity in ml (~7 g ground coffee) -> depth is solved
RIM_CHAMF = 0.5     # small chamfer on the outer rim edge (bed side)
SEGS = 192          # segments around the bowl

# Handle
TOTAL_LEN = 185.0   # same overall length as the JURA 65606 scoop
HANDLE_T = 5.5      # handle thickness
NECK_HW = 8.5       # half width where the handle leaves the bowl
END_HW = 10.0       # half width / radius of the rounded end
FILLET_TOP = 6.0    # top-view fillet radius where the handle meets the rim
SKIRT = 2.5         # ...it only spans the top SKIRT mm, then tapers into the handle
TOP_CHAMF = 0.5     # 45 deg chamfer on the top edges (bed side when printing)
BOT_ROUND = 2.0     # rounding of the underside edges
GUSSET_R = 15.0     # side-view fillet between handle underside and bowl
HOLE_D = 6.5        # hanging hole

# Branding
LOGO_MAIN = os.path.join(HERE, "assets", "aardwark_logo_main.svg")      # animal
LOGO_WORD = os.path.join(HERE, "assets", "aardwark_logo_wordmark.svg")  # letters
WORD_OFFSET = (61.82, 2.05)  # wordmark position inside the main logo (SVG units)
LOGO_H = 13.0       # logo height on the handle
LOGO_DEPTH = 0.8    # deboss depth (also inlay thickness)
TEXT = "Manufactured by CJ 2026"
FONT = r"C:\Windows\Fonts\Rubik-Bold.ttf"
TEXT_MAX_W = 92.0
TEXT_MAX_H = 9.5
TEXT_DEPTH = 0.6

# Derived
RIM_RO = RIM_R + WALL
X_NECK = RIM_RO
XE = TOTAL_LEN - RIM_RO - END_HW    # centre of the rounded end
CX = RIM_R - ARC_R                  # centre of the wall arc (r coordinate)


# ------------------------------------------------------------------ helpers
def link(ob, coll=None):
    (coll or bpy.context.scene.collection).objects.link(ob)
    return ob


def clean_mesh(me, merge=1e-5):
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=merge)
    bmesh.ops.dissolve_degenerate(bm, edges=bm.edges, dist=1e-6)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    me.update()


def mesh_object(name, verts, faces, coll=None):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
    me.validate()
    clean_mesh(me)
    return link(bpy.data.objects.new(name, me), coll)


def mesh_stats(ob):
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    nonman = sum(1 for e in bm.edges if not e.is_manifold)
    vol = bm.calc_volume(signed=True)
    # connected components
    seen, comps = set(), 0
    for v in bm.verts:
        if v.index in seen:
            continue
        comps += 1
        stack = [v]
        seen.add(v.index)
        while stack:
            w = stack.pop()
            for e in w.link_edges:
                o = e.other_vert(w)
                if o.index not in seen:
                    seen.add(o.index)
                    stack.append(o)
    nf = len(bm.faces)
    bm.free()
    return dict(nonmanifold=nonman, volume=vol, components=comps, faces=nf)


def bbox(ob):
    vs = [ob.matrix_world @ v.co for v in ob.data.vertices]
    mn = Vector((min(v.x for v in vs), min(v.y for v in vs), min(v.z for v in vs)))
    mx = Vector((max(v.x for v in vs), max(v.y for v in vs), max(v.z for v in vs)))
    return mn, mx


def apply_boolean(target, cutter, op, solver="MANIFOLD"):
    before = mesh_stats(target)["volume"]
    mod = target.modifiers.new("bool", "BOOLEAN")
    mod.operation = op
    mod.object = cutter
    mod.solver = solver
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(target.evaluated_get(dg))
    target.modifiers.remove(mod)
    old = target.data
    target.data = me
    bpy.data.meshes.remove(old)
    st = mesh_stats(target)
    log(f"  {op:10s} {cutter.name:14s} [{solver}] vol {before:9.1f} -> {st['volume']:9.1f}"
        f"  nonmanifold={st['nonmanifold']} parts={st['components']}")
    return st


def revolve(name, profile, segs=SEGS, coll=None):
    """Solid of revolution around Z. profile = [(r,z), ...] starting and ending on
    the axis (r == 0); interior points must have r > 0."""
    assert profile[0][0] == 0 and profile[-1][0] == 0
    mid = profile[1:-1]
    m = len(mid)
    verts = [(0.0, 0.0, profile[0][1])]
    for j in range(segs):
        a = 2 * math.pi * j / segs
        c, s = math.cos(a), math.sin(a)
        verts += [(r * c, r * s, z) for r, z in mid]
    verts.append((0.0, 0.0, profile[-1][1]))
    ib = len(verts) - 1

    def ix(j, k):
        return 1 + (j % segs) * m + k

    faces = []
    for j in range(segs):
        faces.append((0, ix(j + 1, 0), ix(j, 0)))
        for k in range(m - 1):
            faces.append((ix(j, k), ix(j + 1, k), ix(j + 1, k + 1), ix(j, k + 1)))
        faces.append((ix(j, m - 1), ix(j + 1, m - 1), ib))
    return mesh_object(name, verts, faces, coll)


# ------------------------------------------------------------------ the bowl
def inner_profile(h):
    """Inner bowl surface from the rim (RIM_R,0) down to the floor centre (0,-h).
    Returns [(r, z, theta)], theta = wall angle from vertical (for offsetting)."""
    th1 = math.radians(WALL_ANG)
    pts = []
    n_arc = 48
    for i in range(n_arc + 1):
        th = th1 * i / n_arc
        pts.append((CX + ARC_R * math.cos(th), -ARC_R * math.sin(th), th))
    e_r, e_z = pts[-1][0], pts[-1][1]
    s = (h - ARC_R * math.sin(th1) - FILLET_R * (1 - math.sin(th1))) / math.cos(th1)
    if s < 0:
        raise ValueError("bowl depth too small for arc + fillet")
    f_r, f_z = e_r - s * math.sin(th1), e_z - s * math.cos(th1)
    if s > 1e-6:
        pts.append((f_r, f_z, th1))
    rfc, zfc = f_r - FILLET_R * math.cos(th1), -h + FILLET_R
    if rfc < 1.0:
        raise ValueError("bowl floor collapsed - make the rim wider")
    n_fil = 32
    for i in range(1, n_fil + 1):
        th = th1 + (math.pi / 2 - th1) * i / n_fil
        pts.append((rfc + FILLET_R * math.cos(th), zfc - FILLET_R * math.sin(th), th))
    n_floor = max(2, int(rfc / 1.5))
    for i in range(1, n_floor):
        pts.append((rfc * (1 - i / n_floor), -h, math.pi / 2))
    pts.append((0.0, -h, math.pi / 2))
    return pts


def profile_volume_ml(pts):
    v = 0.0
    for (r0, z0, _), (r1, z1, _) in zip(pts, pts[1:]):
        v += math.pi * (z0 - z1) * (r0 * r0 + r0 * r1 + r1 * r1) / 3.0
    return v / 1000.0


def solve_depth():
    th1 = math.radians(WALL_ANG)
    lo = ARC_R * math.sin(th1) + FILLET_R * (1 - math.sin(th1)) + 1e-4
    hi = 60.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        try:
            v = profile_volume_ml(inner_profile(mid))
        except ValueError:
            hi = mid
            continue
        if v < TARGET_ML:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ----------------------------------------------------------------- the handle
def hw_base(x):
    """Half width of the handle body (slight taper towards the end)."""
    t = min(max((x - X_NECK) / (XE - X_NECK), 0.0), 1.0)
    return NECK_HW + (END_HW - NECK_HW) * t


# top-view fillet circle: tangent to the handle side (y = NECK_HW) and to the rim circle
PF_YC = NECK_HW + FILLET_TOP
PF_XC = math.sqrt((RIM_RO + FILLET_TOP) ** 2 - PF_YC ** 2)
PF_XT = PF_XC * RIM_RO / (RIM_RO + FILLET_TOP)     # tangent point on the rim circle
PF_YT = PF_YC * RIM_RO / (RIM_RO + FILLET_TOP)


def hw_top(x):
    """Half width at the top face, including the fillet into the rim."""
    base = hw_base(x)
    if x >= PF_XC:
        return base
    if x <= PF_XT:
        return max(base, PF_YT)
    return max(base, PF_YC - math.sqrt(max(FILLET_TOP ** 2 - (x - PF_XC) ** 2, 0.0)))


# side-view fillet circle: tangent to the underside line z=-HANDLE_T and
# externally tangent to the outer wall arc (centre (CX,0), radius ARC_R+WALL)
G_ZC = -HANDLE_T - GUSSET_R
G_XC = CX + math.sqrt((ARC_R + WALL + GUSSET_R) ** 2 - G_ZC ** 2)
_d = Vector((G_XC - CX, G_ZC)).normalized()
G_XT, G_ZT = CX + (ARC_R + WALL) * _d.x, (ARC_R + WALL) * _d.y   # tangent point on bowl


def underside_z(x):
    if x >= G_XC:
        return -HANDLE_T
    if x <= G_XT:
        return G_ZT
    return G_ZC + math.sqrt(max(GUSSET_R ** 2 - (G_XC - x) ** 2, 0.0))


N_ARC_SEC = 8


def half_section(x, zb):
    """Right half of the handle cross-section (y >= 0), from the top edge down
    to the bottom centre (exclusive of the centre points)."""
    ht, hb = hw_top(x), hw_base(x)
    t = -zb
    c = TOP_CHAMF
    rb = min(BOT_ROUND + 0.9 * (t - HANDLE_T), hb - 0.3, t - c - SKIRT - 0.2)
    pts = [(ht - c, 0.0), (ht, -c), (hb, -c - SKIRT), (hb, zb + rb)]
    for k in range(1, N_ARC_SEC + 1):
        phi = -0.5 * math.pi * k / N_ARC_SEC
        pts.append((hb - rb + rb * math.cos(phi), zb + rb + rb * math.sin(phi)))
    return pts


def build_handle(coll):
    x0 = 8.0
    xs = []
    x = x0
    while x < 48.0:
        xs.append(x)
        x += 0.5
    while x < XE - 1e-6:
        xs.append(x)
        x += 2.0
    xs.append(XE)
    K = len(half_section(XE, -HANDLE_T))
    N = 2 * K + 2
    verts, faces, rings = [], [], []
    for x in xs:
        zb = underside_z(x)
        right = half_section(x, zb)
        ring = [(x, 0.0, 0.0)] + [(x, y, z) for y, z in right] + [(x, 0.0, zb)] + \
               [(x, -y, z) for y, z in reversed(right)]
        base = len(verts)
        verts += ring
        rings.append(list(range(base, base + N)))
    for ra, rb_ in zip(rings, rings[1:]):
        for j in range(N):
            faces.append((ra[j], ra[(j + 1) % N], rb_[(j + 1) % N], rb_[j]))
    faces.append(tuple(reversed(rings[0])))                 # buried start cap
    # rounded end: revolve the right half-section around the vertical axis at XE
    last = rings[-1]
    tc, bc = last[0], last[K + 1]
    right_ids = last[1:K + 1]
    left_ids = [last[2 * K + 1 - k] for k in range(K)]      # mirror of right[k]
    right_pts = [(verts[i][1], verts[i][2]) for i in right_ids]
    M = 48
    rows = [right_ids]
    for m in range(1, M):
        phi = math.pi * m / M
        base = len(verts)
        verts += [(XE + y * math.sin(phi), y * math.cos(phi), z) for y, z in right_pts]
        rows.append(list(range(base, base + K)))
    rows.append(left_ids)
    for a, b in zip(rows, rows[1:]):
        faces.append((tc, a[0], b[0]))
        for k in range(K - 1):
            faces.append((a[k], a[k + 1], b[k + 1], b[k]))
        faces.append((a[K - 1], bc, b[K - 1]))
    return mesh_object("Handle", verts, faces, coll)


# ----------------------------------------------------------------- SVG / logo
_TOK = re.compile(r"[MmLlHhVvCcSsQqTtZzAa]|[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")


def svg_paths(path):
    txt = open(path, encoding="utf-8").read()
    out = []
    for m in re.finditer(r"<path\b([^>]*)>", txt, re.S):
        d = re.search(r'\sd="([^"]+)"', m.group(1))
        if d:
            out.append(d.group(1))
    return out


def _cubic(p0, p1, p2, p3, out):
    ln = (p1 - p0).length + (p2 - p1).length + (p3 - p2).length
    n = max(6, int(ln / 0.35))
    for i in range(1, n + 1):
        t = i / n
        u = 1 - t
        out.append(p0 * u ** 3 + p1 * 3 * u * u * t + p2 * 3 * u * t * t + p3 * t ** 3)


def path_polylines(d):
    """Flatten an SVG path 'd' (M L H V C S Q T Z, abs+rel) into closed polylines."""
    toks = _TOK.findall(d)
    i, cmd = 0, None
    cur = start = Vector((0.0, 0.0))
    last_c = None
    polys, poly = [], []

    def num():
        nonlocal i
        v = float(toks[i])
        i += 1
        return v

    def pt(rel):
        x, y = num(), num()
        return cur + Vector((x, y)) if rel else Vector((x, y))

    while i < len(toks):
        if toks[i].isalpha():
            cmd = toks[i]
            i += 1
        elif cmd in ("M", "m"):
            cmd = "L" if cmd == "M" else "l"
        rel, C = cmd.islower(), cmd.upper()
        if C == "Z":
            if len(poly) > 2:
                polys.append(poly)
            poly, cur, last_c = [], start, None
            continue
        if C == "M":
            if len(poly) > 2:
                polys.append(poly)
            cur = pt(rel)
            start = cur
            poly = [cur.copy()]
            last_c = None
        elif C == "L":
            cur = pt(rel)
            poly.append(cur.copy())
            last_c = None
        elif C == "H":
            x = num()
            cur = Vector((cur.x + x if rel else x, cur.y))
            poly.append(cur.copy())
            last_c = None
        elif C == "V":
            y = num()
            cur = Vector((cur.x, cur.y + y if rel else y))
            poly.append(cur.copy())
            last_c = None
        elif C in ("C", "S"):
            if C == "C":
                c1 = pt(rel)
            else:
                c1 = (2 * cur - last_c) if last_c is not None else cur.copy()
            c2, p3 = pt(rel), pt(rel)
            _cubic(cur, c1, c2, p3, poly)
            last_c, cur = c2, p3
        elif C in ("Q", "T"):
            q = pt(rel) if C == "Q" else ((2 * cur - last_c) if last_c is not None else cur.copy())
            p3 = pt(rel)
            _cubic(cur, cur + (q - cur) * 2 / 3, p3 + (q - p3) * 2 / 3, p3, poly)
            last_c, cur = q, p3
        else:
            raise ValueError("unsupported SVG command " + cmd)
    if len(poly) > 2:
        polys.append(poly)
    # drop closing duplicates
    clean = []
    for p in polys:
        q = [p[0]]
        for v in p[1:]:
            if (v - q[-1]).length > 1e-4:
                q.append(v)
        if (q[-1] - q[0]).length < 1e-4:
            q.pop()
        clean.append(q)
    return clean


def logo_polylines():
    """aardwark logo = aardvark silhouette from the main logo + single-colour
    wordmark (the main logo splits letters into colour pieces). The (R) mark is
    left out - far too small to print."""
    animal = None
    for d in svg_paths(LOGO_MAIN):
        polys = path_polylines(d)
        if min(v.x for p in polys for v in p) < 30.0:
            animal = polys
    assert animal, "aardvark path not found"
    polys = list(animal)
    off = Vector(WORD_OFFSET)
    for d in svg_paths(LOGO_WORD):
        polys += [[v + off for v in p] for p in path_polylines(d)]
    return polys


def prism_from_polylines(name, polylines, z0, z1, coll):
    """Even-odd filled 2D outline extruded between z0 and z1 (via a 2D curve)."""
    cu = bpy.data.curves.new(name + "_cu", "CURVE")
    cu.dimensions = "2D"
    cu.fill_mode = "BOTH"
    cu.extrude = (z1 - z0) / 2.0
    for p in polylines:
        sp = cu.splines.new("POLY")
        sp.points.add(len(p) - 1)
        for pnt, v in zip(sp.points, p):
            pnt.co = (v.x, v.y, 0.0, 1.0)
        sp.use_cyclic_u = True
    ob = link(bpy.data.objects.new(name + "_tmp", cu), coll)
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
    bpy.data.objects.remove(ob)
    bpy.data.curves.remove(cu)
    me.transform(Matrix.Translation((0, 0, (z0 + z1) / 2.0)))
    clean_mesh(me)
    me.name = name
    return link(bpy.data.objects.new(name, me), coll)


def text_mesh(name, body, font_path, coll):
    cu = bpy.data.curves.new(name + "_cu", "FONT")
    cu.body = body
    cu.font = bpy.data.fonts.load(font_path, check_existing=True)
    cu.size = 10.0
    cu.align_x = "CENTER"
    cu.align_y = "CENTER"
    cu.resolution_u = 20
    cu.extrude = 0.5        # 1 mm thick, centred on z = 0
    ob = link(bpy.data.objects.new(name + "_tmp", cu), coll)
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
    bpy.data.objects.remove(ob)
    bpy.data.curves.remove(cu)
    clean_mesh(me)
    me.name = name
    return link(bpy.data.objects.new(name, me), coll)


# ---------------------------------------------------------------------- build
def build():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scn = bpy.context.scene
    scn.unit_settings.system = "METRIC"
    scn.unit_settings.scale_length = 0.001
    scn.unit_settings.length_unit = "MILLIMETERS"
    build_coll = bpy.data.collections.new("build_parts")
    scn.collection.children.link(build_coll)

    # --- bowl
    h = solve_depth()
    inner = inner_profile(h)
    cap_ml = profile_volume_ml(inner)
    log(f"bowl: rim d={2 * RIM_R:.1f} (outer {2 * RIM_RO:.1f}) mm, inner depth={h:.2f} mm,"
        f" level capacity={cap_ml:.2f} ml")
    outer = [(r + WALL * math.cos(t), z - WALL * math.sin(t)) for r, z, t in inner]
    thc = math.asin(RIM_CHAMF / (ARC_R + WALL))
    chamf_pt = (CX + (ARC_R + WALL) * math.cos(thc), -RIM_CHAMF)
    outer_prof = [(0.0, 0.0), (RIM_RO - RIM_CHAMF, 0.0), chamf_pt] + \
                 [p for p in outer if p[1] < -RIM_CHAMF - 1e-6]
    outer_prof[-1] = (0.0, outer_prof[-1][1])
    bowl = revolve("Bowl_outer", outer_prof, coll=build_coll)
    cavity_prof = [(0.0, 1.0), (RIM_R, 1.0)] + [(r, z) for r, z, _ in inner]
    cavity = revolve("Bowl_cavity", cavity_prof, coll=build_coll)

    # --- handle
    handle = build_handle(build_coll)
    log(f"handle: neck/end width {2 * NECK_HW:.1f}/{2 * END_HW:.1f} mm, thickness {HANDLE_T} mm,"
        f" gusset fillet x {G_XT:.1f}..{G_XC:.1f}")
    for ob in (bowl, cavity, handle):
        log(f"  part {ob.name:12s} {mesh_stats(ob)}")

    # --- hanging hole
    n = 64
    hv = []
    for zz in (-HANDLE_T - 2.0, 2.0):
        hv += [(XE + HOLE_D / 2 * math.cos(2 * math.pi * k / n),
                HOLE_D / 2 * math.sin(2 * math.pi * k / n), zz) for k in range(n)]
    hf = [tuple(range(n - 1, -1, -1)), tuple(range(n, 2 * n))] + \
         [(k, (k + 1) % n, n + (k + 1) % n, n + k) for k in range(n)]
    hole = mesh_object("Hole", hv, hf, build_coll)

    # --- logo (top face, reads left->right from the bowl)
    polys = logo_polylines()
    umin = min(v.x for p in polys for v in p)
    umax = max(v.x for p in polys for v in p)
    vmin = min(v.y for p in polys for v in p)
    vmax = max(v.y for p in polys for v in p)
    sc = LOGO_H / (vmax - vmin)
    logo_len = (umax - umin) * sc
    top_lo = X_NECK + 5.0
    top_hi = XE - HOLE_D / 2 - 3.0
    lx0 = 0.5 * (top_lo + top_hi) - 0.5 * logo_len
    vc = 0.5 * (vmin + vmax)
    lp = [[Vector((lx0 + (v.x - umin) * sc, (vc - v.y) * sc)) for v in p] for p in polys]
    log(f"logo: {logo_len:.1f} x {LOGO_H:.1f} mm at x={lx0:.1f}..{lx0 + logo_len:.1f}"
        f" (handle width there {2 * hw_top(lx0):.1f}..{2 * hw_top(lx0 + logo_len):.1f})")
    scn["logo_x0"] = lx0
    logo_cut = prism_from_polylines("Logo_cutter", lp, -LOGO_DEPTH, 1.0, build_coll)
    logo_inlay = prism_from_polylines("Inlay_logo", lp, -LOGO_DEPTH, 0.0, scn.collection)

    # --- maker's mark (underside, reads correctly after rolling the scoop over)
    txt = text_mesh("Text_raw", TEXT, FONT, build_coll)
    mn, mx = bbox(txt)
    w, hgt = mx.x - mn.x, mx.y - mn.y
    s = min(TEXT_MAX_W / w, TEXT_MAX_H / hgt)
    bot_lo = G_XC + 4.0
    bot_hi = top_hi
    tcx = 0.5 * (bot_lo + bot_hi)
    tcy = 0.5 * (mn.y + mx.y)
    # centre, scale, roll 180 deg about X, place on the underside
    m_center = Matrix.Translation((-0.5 * (mn.x + mx.x), -tcy, 0.0))
    m_scale = Matrix.Diagonal((s, s, 1.0, 1.0))
    m_roll = Matrix.Rotation(math.pi, 4, "X")
    base = m_roll @ m_scale @ m_center
    tw, th = w * s, hgt * s
    flat_hw = hw_base(tcx - tw / 2) - BOT_ROUND
    log(f"text: '{TEXT}' {tw:.1f} x {th:.1f} mm at x={tcx - tw / 2:.1f}..{tcx + tw / 2:.1f}"
        f" (flat underside width there {2 * flat_hw:.1f})")
    assert th / 2 < flat_hw - 0.8, "text too tall for the flat underside"

    def text_prism(name, z_lo, z_hi, coll):
        me = txt.data.copy()
        # raw text spans z in [-0.5, 0.5]; map to [z_lo, z_hi] after the roll
        me.transform(base)                               # z now in [-0.5, 0.5] (flipped)
        me.transform(Matrix.Translation((tcx, 0, 0)) @
                     Matrix.Translation((0, 0, (z_lo + z_hi) / 2)) @
                     Matrix.Diagonal((1, 1, (z_hi - z_lo), 1)))
        me.name = name
        clean_mesh(me)
        return link(bpy.data.objects.new(name, me), coll)

    text_cut = text_prism("Text_cutter", -HANDLE_T - 1.0, -HANDLE_T + TEXT_DEPTH, build_coll)
    text_inlay = text_prism("Inlay_text", -HANDLE_T, -HANDLE_T + TEXT_DEPTH, scn.collection)

    for ob in (hole, logo_cut, logo_inlay, text_cut, text_inlay):
        log(f"  part {ob.name:12s} {mesh_stats(ob)}")

    # --- booleans
    body = bowl
    body.name = "Scoop_body"
    body.data.name = "Scoop_body"
    apply_boolean(body, handle, "UNION")
    apply_boolean(body, cavity, "DIFFERENCE")
    apply_boolean(body, hole, "DIFFERENCE")
    apply_boolean(body, logo_cut, "DIFFERENCE")
    st = apply_boolean(body, text_cut, "DIFFERENCE")
    build_coll.objects.unlink(body)
    scn.collection.objects.link(body)
    build_coll.hide_viewport = True
    build_coll.hide_render = True

    mn, mx = bbox(body)
    log(f"BODY: size {mx.x - mn.x:.2f} x {mx.y - mn.y:.2f} x {mx.z - mn.z:.2f} mm,"
        f" volume {st['volume'] / 1000:.2f} cm3, nonmanifold={st['nonmanifold']},"
        f" parts={st['components']}, faces={st['faces']}")
    for ob in (logo_inlay, text_inlay):
        s2 = mesh_stats(ob)
        log(f"INLAY {ob.name}: volume {s2['volume']:.1f} mm3, nonmanifold={s2['nonmanifold']},"
            f" parts={s2['components']}")
    return body, logo_inlay, text_inlay, dict(depth=h, capacity=cap_ml)


# --------------------------------------------------------------------- export
PRINT_MATRIX = Matrix.Rotation(math.pi, 4, "X")   # top face (rim + logo) onto the bed


def export(body, inlays):
    mn, mx = bbox(body)
    # after the 180 deg roll the rim sits at z=0; centre the part on the bed
    pm = Matrix.Translation((-(mn.x + mx.x) / 2, 0, 0))
    pm = PRINT_MATRIX @ pm
    files = []
    for ob, fname in [(body, "aardwark_scoop_body.stl")] + inlays:
        tmp = bpy.data.objects.new("exp_" + ob.name, ob.data.copy())
        bpy.context.scene.collection.objects.link(tmp)
        tmp.data.transform(pm)
        for o in bpy.context.scene.objects:
            o.select_set(False)
        tmp.select_set(True)
        bpy.context.view_layer.objects.active = tmp
        path = os.path.join(OUT, fname)
        bpy.ops.wm.stl_export(filepath=path, export_selected_objects=True,
                              apply_modifiers=True, ascii_format=False, global_scale=1.0)
        tmin, tmax = bbox(tmp)
        log(f"exported {fname}: z {tmin.z:.2f}..{tmax.z:.2f}")
        me = tmp.data
        bpy.data.objects.remove(tmp)
        bpy.data.meshes.remove(me)
        files.append(path)
    return files


# --------------------------------------------------------------------- render
def srgb(hexstr):
    c = [int(hexstr[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return tuple(((x + 0.055) / 1.055) ** 2.4 if x > 0.04045 else x / 12.92 for x in c) + (1.0,)


def material(name, hexcol, rough=0.45):
    mat = bpy.data.materials.new(name)
    if hasattr(mat, "use_nodes"):
        try:
            mat.use_nodes = True
        except Exception:
            pass
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = srgb(hexcol)
    bsdf.inputs["Roughness"].default_value = rough
    return mat


def set_material(ob, mat):
    ob.data.materials.clear()
    ob.data.materials.append(mat)


def look_at(cam_ob, target, direction, dist):
    loc = Vector(target) + Vector(direction).normalized() * dist
    cam_ob.location = loc
    cam_ob.rotation_euler = (Vector(target) - loc).to_track_quat("-Z", "Y").to_euler()


def render_previews(body, logo_inlay, text_inlay):
    scn = bpy.context.scene
    scn.render.engine = "CYCLES"
    scn.cycles.device = "CPU"
    scn.cycles.samples = 48
    scn.cycles.use_denoising = True
    scn.render.resolution_x, scn.render.resolution_y = 1500, 950
    scn.render.image_settings.file_format = "PNG"
    world = bpy.data.worlds.new("World")
    scn.world = world
    try:
        world.use_nodes = True
    except Exception:
        pass
    bg = world.node_tree.nodes.get("Background")
    bg.inputs[0].default_value = (0.86, 0.87, 0.9, 1.0)
    bg.inputs[1].default_value = 0.55

    def sun(name, rot, energy, angle=8):
        ld = bpy.data.lights.new(name, "SUN")
        ld.energy = energy
        ld.angle = math.radians(angle)
        ob = link(bpy.data.objects.new(name, ld))
        ob.rotation_euler = [math.radians(a) for a in rot]
        return ob

    sun("Key", (40, 10, 35), 3.2)
    sun("Rim", (-55, 0, 160), 1.6, 15)

    m_black = material("Black PETG", "1d1d21", 0.42)
    m_grey = material("Grey PLA", "c9c9cc", 0.55)
    m_purple = material("aardwark purple", "7350AA", 0.4)
    m_lilac = material("aardwark lilac", "AA78C8", 0.4)
    m_floor = material("Floor", "f2f2f4", 0.8)

    floor_me = bpy.data.meshes.new("Floor")
    S = 2000
    floor_me.from_pydata([(-S, -S, 0), (S, -S, 0), (S, S, 0), (-S, S, 0)], [], [(0, 1, 2, 3)])
    floor = link(bpy.data.objects.new("Floor", floor_me))
    set_material(floor, m_floor)

    cam_d = bpy.data.cameras.new("Cam")
    cam_d.clip_start, cam_d.clip_end = 1.0, 10000.0
    cam = link(bpy.data.objects.new("Cam", cam_d))
    scn.camera = cam

    parts = (body, logo_inlay, text_inlay)
    for ob in parts:   # display only - STL export is unaffected
        ob.data.shade_smooth()
        ob.data.set_sharp_from_angle(angle=math.radians(35))
    bmin, bmax = bbox(body)
    zbot = bmin.z

    def pose(matrix):
        for ob in parts:
            ob.matrix_world = matrix

    def shot(fname, target, direction, dist, lens=60, ortho=None, floor_z=None, rot=None):
        if ortho:
            cam_d.type, cam_d.ortho_scale = "ORTHO", ortho
        else:
            cam_d.type, cam_d.lens = "PERSP", lens
        floor.hide_render = floor_z is None
        if floor_z is not None:
            floor.location.z = floor_z
        if rot is None:
            look_at(cam, target, direction, dist)
        else:   # axis-aligned views: explicit orientation so the image is not spun
            cam.location = Vector(target) + Vector(direction).normalized() * dist
            cam.rotation_euler = [math.radians(a) for a in rot]
        scn.render.filepath = os.path.join(OUT, fname)
        bpy.ops.render.render(write_still=True)
        log("rendered", fname)

    center = Vector(((bmin.x + bmax.x) / 2, 0, (bmin.z + bmax.z) / 2))
    # 1) two-colour, use orientation
    set_material(body, m_black)
    set_material(logo_inlay, m_purple)
    set_material(text_inlay, m_lilac)
    pose(Matrix.Identity(4))
    shot("preview_1_two_colour.png", center, (0.35, -0.8, 0.75), 330, lens=55, floor_z=zbot - 0.05)
    # 2) single-colour (inlays hidden) - what a plain print looks like
    set_material(body, m_grey)
    logo_inlay.hide_render = text_inlay.hide_render = True
    shot("preview_2_single_colour.png", center, (0.35, -0.8, 0.75), 330, lens=55, floor_z=zbot - 0.05)
    logo_inlay.hide_render = text_inlay.hide_render = False
    # 3) top view (logo check)
    set_material(body, m_black)
    shot("preview_3_top.png", Vector((center.x, 0, 0)), (0, 0, 1), 400, ortho=200,
         floor_z=zbot - 0.05, rot=(0, 0, 0))
    # 4) print orientation: rim + logo on the bed, underside text on top
    pm = Matrix.Rotation(math.pi, 4, "X")
    pose(pm)
    shot("preview_4_print_orientation.png", Vector((center.x, 0, 10)), (0.35, -0.8, 0.75), 330,
         lens=55, floor_z=-0.05)
    shot("preview_5_underside_text.png", Vector((center.x, 0, 0)), (0, 0, 1), 400, ortho=200,
         floor_z=-0.05, rot=(0, 0, 0))
    # detail: handle/bowl junction as it sits on the print bed
    shot("detail_junction.png", Vector((24, 0, 6)), (0.55, -0.75, 0.55), 120, lens=70, floor_z=-0.05)
    # detail: logo close-up (top face, use orientation)
    pose(Matrix.Identity(4))
    lx = bpy.context.scene.get("logo_x0", 46.0)
    shot("detail_logo.png", Vector((lx + 21, 0, 0)), (0, 0, 1), 300, ortho=46,
         floor_z=zbot - 0.05, rot=(0, 0, 0))
    # 6) side profile
    pose(Matrix.Identity(4))
    shot("preview_6_side.png", Vector((center.x, 0, center.z)), (0, -1, 0), 400, ortho=200,
         floor_z=None, rot=(90, 0, 0))
    pose(Matrix.Identity(4))


# ----------------------------------------------------------------------- main
if __name__ == "__main__":
    body, logo_inlay, text_inlay, info = build()
    export(body, [(logo_inlay, "aardwark_scoop_inlay_logo.stl"),
                  (text_inlay, "aardwark_scoop_inlay_text.stl")])
    if DO_RENDER:
        render_previews(body, logo_inlay, text_inlay)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "aardwark_scoop.blend"))
    log("saved blend; done")
