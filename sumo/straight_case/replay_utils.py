"""SUMO 部分的公共工具。

* 路径与数据读取：data/route.csv、data/speed.csv、data/tasks.csv
* 可选的私有编号映射：network/id_map.csv
* 车道中心线几何：生成路线折线，把里程定位到坐标
* FCD 回放文件输出，供 SUMO 的 FCD 回放设备读取

本文件不包含任何地图数据。路网 network/map.net.xml 由使用者自行提供，不随仓库发布。
"""
import csv
import math
from collections import namedtuple
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
NET_DIR = ROOT / "network"
OUT_DIR = ROOT / "output"
NET_FILE = NET_DIR / "map.net.xml"
ID_MAP_FILE = NET_DIR / "id_map.csv"

DT = 0.2                  # 速度曲线的时间步长，单位 s
MAX_YAW_PER_STEP = 30.0   # 每步允许的最大航向变化，单位度

Task = namedtuple(
    "Task",
    "name trigger ot service stop_edge stop_start stop_end trigger_s t_ref")


# --------------------------------------------------------------------------- 数据
def load_routes():
    return pd.read_csv(DATA_DIR / "route.csv")


def load_speeds():
    return pd.read_csv(DATA_DIR / "speed.csv")


def vehicle_ids(speed):
    """返回车辆列表，VUT 在前，其余 OT 按列顺序排列。"""
    cols = [c for c in speed.columns if c != "time"]
    return ["VUT"] + [c for c in cols if c != "VUT"]


def vehicle_types(vids):
    return {v: ("vutType" if v == "VUT" else "serviceType") for v in vids}


def load_tasks():
    """读取任务表，返回 Task 列表。

    触发节点相同的行属于同一个协同组，
    它们的触发里程和 VUT 参考到达时刻必须一致，否则报错。
    """
    df = pd.read_csv(DATA_DIR / "tasks.csv")
    tasks = [Task(r.task, r.trigger, r.ot, float(r.service_s), r.stop_edge,
                  float(r.stop_start), float(r.stop_end),
                  float(r.trigger_s), float(r.t_ref))
             for r in df.itertuples(index=False)]
    seen = {}
    for t in tasks:
        key = (t.trigger_s, t.t_ref)
        if seen.setdefault(t.trigger, key) != key:
            raise ValueError("tasks sharing trigger %s disagree on trigger_s / t_ref"
                             % t.trigger)
    return tasks


def family(task_name):
    """'T1#1' 变为 'T1'，同一任务的两个子任务属于同一族。"""
    return task_name.split("#")[0]


# ------------------------------------------------------------------------ 编号映射
class IdMap:
    """可选的私有编号映射，把仓库中的中性编号翻译成真实路网中的编号。

    network/id_map.csv 的列为 kind、public_id、real_id，kind 取 edge 或 node。
    文件不存在时编号原样使用。
    """

    def __init__(self, path=ID_MAP_FILE):
        self._m = {"edge": {}, "node": {}}
        if Path(path).is_file():
            with open(path, newline="", encoding="utf-8-sig") as f:
                for r in csv.DictReader(f):
                    self._m[r["kind"]][r["public_id"]] = r["real_id"]

    def edge(self, public_id):
        return self._m["edge"].get(public_id, public_id)

    def node(self, public_id):
        return self._m["node"].get(public_id, public_id)


def load_network():
    """读取使用者自备的路网 network/map.net.xml 和可选的编号映射。"""
    if not NET_FILE.is_file():
        raise SystemExit(
            "network/map.net.xml not found.\n"
            "The road network is not distributed with this repository. Put your own\n"
            "SUMO network there (see the README, section 'SUMO simulation').")
    import sumolib
    return sumolib.net.readNet(str(NET_FILE), withInternal=True), IdMap()


# ---------------------------------------------------------------------- 数值计算
def integrate_speed(values, dt=DT):
    """梯形积分得到沿路线里程，车辆不在场景中的时刻取 None。"""
    s, prev, out = 0.0, None, []
    for val in values:
        if val != val:                      # NaN 表示车辆不在场景中
            out.append(None)
            prev = None
            continue
        if prev is not None:
            s += 0.5 * (prev + val) * dt
        out.append(s)
        prev = val
    return out


# ---------------------------------------------------------------------- 几何
def lane_segments(lane):
    """把车道按形状拆成折线段，每段为 x0 y0 x1 y1 车道 起点位置 终点位置 航向。"""
    sh = [tuple(map(float, p)) for p in lane.getShape()]
    out, cum = [], 0.0
    for i in range(len(sh) - 1):
        dx, dy = sh[i + 1][0] - sh[i][0], sh[i + 1][1] - sh[i][1]
        seg = math.hypot(dx, dy)
        out.append((sh[i][0], sh[i][1], sh[i + 1][0], sh[i + 1][1], lane.getID(),
                    cum, cum + seg, math.degrees(math.atan2(dy, dx)) % 360.0))
        cum += seg
    return out


def _add(segs, lane):
    """追加一条车道。若与上一段末端不衔接，先补一段直线桥接，保证折线连续。"""
    pts = [tuple(map(float, p)) for p in lane.getShape()]
    if segs:
        a = segs[-1]
        if math.hypot(pts[0][0] - a[2], pts[0][1] - a[3]) > 1e-6:
            segs.append((a[2], a[3], pts[0][0], pts[0][1], lane.getID(),
                         0.0, 0.0, a[7]))
    segs.extend(lane_segments(lane))


def build_path(net, idmap, edges):
    """由边列表生成路线的车道中心线折线。

    位置始终取车道中心线，连接有路口内部车道时一并使用，
    因此坐标和车道位置与 SUMO 自身的口径一致。
    返回折线段列表和累计长度列表。
    """
    segs, prev = [], None
    for e in edges:
        try:
            edge = net.getEdge(idmap.edge(e))
        except Exception:
            raise KeyError("edge '%s' (network id '%s') is not in network/map.net.xml"
                           % (e, idmap.edge(e)))
        if prev is not None:
            via, p0, best = None, edge.getLane(0).getShape()[0], None
            try:
                # 多个连接时取内部车道终点离下一条车道起点最近的一个
                for c in prev.getConnections(toEdge=edge):
                    vl = c.getViaLaneID()
                    if not vl:
                        continue
                    cand = net.getLane(vl)
                    end = cand.getShape()[-1]
                    d = math.hypot(end[0] - p0[0], end[1] - p0[1])
                    if best is None or d < best[0]:
                        best = (d, cand)
                via = best[1] if best else None
            except Exception:
                via = None
            if via is not None:
                _add(segs, via)
            elif segs:
                # 没有内部车道时用直线桥接
                a = segs[-1]
                segs.append((a[2], a[3], p0[0], p0[1], edge.getLane(0).getID(),
                             0.0, 0.0, a[7]))
        _add(segs, edge.getLane(0))
        prev = edge
    cuml = [0.0]
    for s in segs:
        cuml.append(cuml[-1] + math.hypot(s[2] - s[0], s[3] - s[1]))
    return segs, cuml


def locate(segs, cuml, s):
    """把沿路线里程 s 定位为 x y 车道 车道位置 航向。"""
    s = max(0.0, min(cuml[-1], s))
    lo, hi = 0, len(cuml) - 1
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if cuml[mid] <= s:
            lo = mid
        else:
            hi = mid
    seg = segs[lo]
    L = cuml[lo + 1] - cuml[lo]
    f = 0.0 if L < 1e-9 else (s - cuml[lo]) / L
    return (seg[0] + f * (seg[2] - seg[0]), seg[1] + f * (seg[3] - seg[1]),
            seg[4], seg[5] + f * (seg[6] - seg[5]), seg[7])


def smooth_headings(frames, vids, max_yaw=MAX_YAW_PER_STEP):
    """航向取所在车道的前进方向，做三点圆滑并限制转向速率，避免路口处突变。"""
    for v in vids:
        idx = [i for i in range(len(frames)) if v in frames[i]]
        tgt = [frames[i][v][5] for i in idx]
        sm = []
        for k in range(len(tgt)):
            vals = tgt[max(0, k - 1):min(len(tgt), k + 2)]
            sx = sum(math.cos(math.radians(x)) for x in vals)
            sy = sum(math.sin(math.radians(x)) for x in vals)
            sm.append(math.degrees(math.atan2(sy, sx)) % 360.0)
        prev = None
        for k, i in enumerate(idx):
            if prev is None:
                prev = sm[k]
            else:
                d = (sm[k] - prev + 180.0) % 360.0 - 180.0
                prev = (prev + max(-max_yaw, min(max_yaw, d))) % 360.0
            frames[i][v][5] = prev


# ------------------------------------------------------------------------ 输出
def write_fcd(frames, times, vids, comment, path=None):
    """写出 FCD 回放文件，供 scenario_replay.sumocfg 读取。

    frames[i][车辆] = [x, y, 车道, 车道位置, 速度, 航向]
    """
    path = Path(path) if path else OUT_DIR / "replay.fcd.xml"
    path.parent.mkdir(parents=True, exist_ok=True)
    types = vehicle_types(vids)
    lines = ['<?xml version="1.0" encoding="utf-8"?>',
             "<!-- %s -->" % comment, "<fcd-export>"]
    for i, t in enumerate(times):
        if not frames[i]:
            continue
        lines.append('<timestep time="%.2f">' % t)
        for v in vids:
            rec = frames[i].get(v)
            if rec is None:
                continue
            x, y, lane, pos, sp, ang = rec
            lines.append(
                '<vehicle id="%s" x="%.2f" y="%.2f" angle="%.2f" type="%s" '
                'speed="%.2f" lane="%s" pos="%.2f" slope="0.00"/>'
                % (v, x, y, (90.0 - ang) % 360.0, types[v], sp, lane, pos))
        lines.append("</timestep>")
    lines.append("</fcd-export>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def write_metrics(frames, times, vids, speed_name="speed.csv"):
    """输出各车速度 speed.csv，以及 VUT 与各 OT 的相对距离 relative_distance.csv。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ots = [v for v in vids if v != "VUT"]
    rows, rel = [], []
    for i, t in enumerate(times):
        row = {"time": t}
        for v in vids:
            rec = frames[i].get(v)
            row[v] = round(rec[4], 1) if rec else float("nan")
        rows.append(row)
        r = {"time": f"{t:.1f}"}
        a = frames[i].get("VUT")
        for v in ots:
            b = frames[i].get(v)
            r[v] = math.hypot(a[0] - b[0], a[1] - b[1]) if a and b else float("nan")
        rel.append(r)
    pd.DataFrame(rows).to_csv(OUT_DIR / speed_name, index=False, float_format="%.1f")
    pd.DataFrame(rel).to_csv(OUT_DIR / "relative_distance.csv", index=False,
                             float_format="%.6f")
