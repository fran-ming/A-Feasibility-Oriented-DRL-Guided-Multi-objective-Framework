"""把整个运行重新锚定到 VUT 通过各触发节点的时刻，输出 output/replay.fcd.xml。

在记录的调度基础上显式保证协同结构：

1. VUT 轨迹连续。VUT 以 6 m/s 巡航，接近触发点时降到 2 m/s 爬行，
   使它在参考时刻 t_ref 到达每个触发节点。每一段都是前向可达的，
   里程 s(t) 连续，不会出现位置跳变。
2. OT 停驻重锚定。OT 在任务点的停驻时长改为直到 VUT 通过该任务的触发节点为止。
   VUT 一到，OT 立即释放并驶到它前方，而不是等 VUT 走完才动。
3. 协同组共用一个时钟。通过时刻按触发节点只计算一次，该节点上的所有任务读取同一个值，
   因此同组的所有 OT，例如 T3#0 和 T3#1，在同一个仿真步被释放。
   同时性来自共享的时钟，而不是各车自己的计时器。

输入：data/route.csv、data/speed.csv、data/tasks.csv
输出：output/replay.fcd.xml、output/speed.csv、output/relative_distance.csv
replay.fcd.xml 会被覆盖，所以 retime.py 与 build_fcd.py 二选一运行。

    python retime.py
"""
from replay_utils import (DT, build_path, load_network, load_routes, load_speeds,
                          load_tasks, locate, smooth_headings, vehicle_ids,
                          write_fcd, write_metrics)
from cooperative_schedule import group_by_trigger

# VUT 运动学参数
V, A, VLO = 6.0, 4.5, 2.0               # 巡航速度 加减速度 爬行速度，单位 m/s 与 m/s^2
TR = (V - VLO) / A                      # 爬行与巡航之间切换所需时间，单位 s
DR = (V * V - VLO * VLO) / (2 * A)      # 切换过程所用距离，单位 m


def vut_timeline(triggers, t0, s_tot):
    """生成从入场到路线终点的分段，每段为 起始时刻 结束时刻 起始里程 结束里程 类型。

    triggers 为按里程排序的 触发里程 与 参考命中时刻 列表。
    类型含义：const 匀速，slow 全程爬行，fast 全程巡航，
    crawl 先巡航，到触发点前减速到爬行速度。
    """
    timeline, t_cur, s_cur = [], t0, 0.0
    for sk, tk in triggers:
        D, dt = sk - s_cur, tk - t_cur
        if D <= 1e-6:
            continue
        if D <= 2 * DR:
            dur, kind = (dt if D / dt <= V else D / V), "const"
        elif dt >= D / VLO:
            dur, kind = D / VLO, "slow"
        elif dt <= D / V:
            dur, kind = D / V, "fast"
        else:
            Dl = (dt - 2 * TR - (D - 2 * DR) / V) / (1 / VLO - 1 / V)
            if 0.0 <= Dl <= D - 2 * DR:
                dur, kind = dt, "crawl"
            else:
                dur, kind = dt, "const"
        timeline.append((t_cur, t_cur + dur, s_cur, s_cur + D, kind))
        t_cur, s_cur = t_cur + dur, s_cur + D
    timeline.append((t_cur, t_cur + (s_tot - s_cur) / V, s_cur, s_tot, "fast"))
    return timeline


def make_vut_state(timeline, t0, s_tot):
    def vut_state(t):
        if t < t0 - 1e-9:
            return None, None
        for ta, tb, sa, sb, kind in timeline:
            if t > tb + 1e-9:
                continue
            dt, D = t - ta, sb - sa
            if kind == "const":
                v = D / max(1e-9, tb - ta)
                return sa + v * dt, v
            if kind == "fast":
                return sa + V * dt, V
            if kind == "slow":
                return sa + VLO * dt, VLO
            Dl = (tb - ta - 2 * TR - (D - 2 * DR) / V) / (1 / VLO - 1 / V)
            t_cruise = (D - 2 * DR - Dl) / V
            if dt <= TR:
                return sa + VLO * dt + 0.5 * A * dt * dt, min(V, VLO + A * dt)
            if dt <= TR + t_cruise:
                return sa + DR + V * (dt - TR), V
            if dt <= 2 * TR + t_cruise:
                tau = dt - TR - t_cruise
                return (sa + DR + V * t_cruise + V * tau - 0.5 * A * tau * tau,
                        max(VLO, V - A * tau))
            return sb - Dl + VLO * (dt - 2 * TR - t_cruise), VLO
        return s_tot, V
    return vut_state


def ot_blocks(values):
    """把 OT 速度曲线切成交替出现的行驶块和停驻块。"""
    S, VV, s, prev = [], [], 0.0, None
    for x in values:
        if x != x:
            S.append(None)
            VV.append(None)
            prev = None
            continue
        if prev is not None:
            s += 0.5 * (prev + x) * DT
        S.append(s)
        VV.append(float(x))
        prev = x
    blocks, i = [], 0
    while i < len(values):
        if VV[i] is None:
            i += 1
            continue
        dwell = VV[i] < 0.05
        j = i
        while j + 1 < len(values) and VV[j + 1] is not None and (VV[j + 1] < 0.05) == dwell:
            j += 1
        blocks.append({"dwell": dwell, "i0": i, "i1": j, "s": S[i],
                       "dur": (j - i) * DT + DT})
        i = j + 1
    return blocks, S, VV


def main():
    net, idmap = load_network()
    route, speed, tasks = load_routes(), load_speeds(), load_tasks()
    vids = vehicle_ids(speed)
    ots = [v for v in vids if v != "VUT"]
    times = [round(float(t), 2) for t in speed["time"]]

    # ---- 由 data/tasks.csv 得到协同结构 ---------------------------
    groups = group_by_trigger(tasks)                      # 触发节点到成员的映射
    triggers = sorted((m[0].trigger_s, m[0].t_ref) for m in groups.values())
    # 每台 OT 的任务，按 VUT 到达其触发点的先后排序
    order = {ot: [t.name for t in sorted(tasks, key=lambda t: t.trigger_s)
                  if t.ot == ot] for ot in ots}

    paths = {v: build_path(net, idmap,
                           route[route.vehicle == v].sort_values("seq").edge.tolist())
             for v in vids}
    segs_v, cuml_v = paths["VUT"]
    s_tot = cuml_v[-1]

    # ---- 1) VUT 时间线 -----------------------------------------------------
    t0 = triggers[0][1] - triggers[0][0] / V
    timeline = vut_timeline(triggers, t0, s_tot)
    vut_state = make_vut_state(timeline, t0, s_tot)
    print("VUT enters at %.2f s" % t0)
    for a in timeline:
        print("  %6.1f -> %6.1f s   s %6.1f -> %6.1f m   %s" % a)

    sv_map = {}
    for t in times:
        s, _ = vut_state(t)
        if s is not None:
            sv_map[t] = s
    sv_times = sorted(sv_map)

    # ---- 2) 每个触发节点一个释放时钟 -------------------------------
    hit_node = {node: next((t for t in sv_times if sv_map[t] >= m[0].trigger_s), None)
                for node, m in groups.items()}
    trigger_at = {t.name: hit_node[t.trigger] for t in tasks}   # 同组成员共用同一个值
    print("\nactual trigger times:", {k: round(v, 1) for k, v in trigger_at.items()})

    # ---- 3) OT 停驻重锚定 --------------------------------------------
    frames = [dict() for _ in times]
    report = {}
    for ot in ots:
        blocks, S, VV = ot_blocks(speed[ot].to_numpy(float))
        targets = [trigger_at[n] for n in order[ot]]
        segs, cuml = paths[ot]
        new_s, new_v = [None] * len(times), [None] * len(times)
        t_cur, di, log = times[0], 0, []
        for b in blocks:
            if b["dwell"]:
                dur = b["dur"]
                if di < len(targets) and b["dur"] > 1.0:
                    tgt = targets[di]
                    # 停驻到 VUT 通过本任务的触发节点为止
                    dur = max(0.0, tgt - t_cur) if tgt else b["dur"]
                    log.append((order[ot][di], t_cur, t_cur + dur, tgt))
                    di += 1
                for k in range(int(round(dur / DT)) + 1):
                    idx = int(round((t_cur + k * DT - times[0]) / DT))
                    if 0 <= idx < len(times):
                        new_s[idx], new_v[idx] = b["s"], 0.0
                t_cur += dur
            else:
                for k in range(b["i0"], b["i1"] + 1):
                    idx = int(round((t_cur + (k - b["i0"]) * DT - times[0]) / DT))
                    if 0 <= idx < len(times):
                        new_s[idx], new_v[idx] = S[k], VV[k]
                t_cur += b["dur"]
        # 收尾：按最后速度继续行驶到路线终点
        last_s = next((new_s[i] for i in range(len(times) - 1, -1, -1)
                       if new_s[i] is not None), None)
        last_v = next((new_v[i] for i in range(len(times) - 1, -1, -1)
                       if new_v[i] is not None), 0.0)
        if last_s is not None and last_v and last_v > 0.1:
            idx0, s2, k = int(round((t_cur - times[0]) / DT)), last_s, 0
            while s2 < cuml[-1] - 1e-6:
                idx = idx0 + k
                if idx >= len(times):
                    break
                s2 = min(cuml[-1], s2 + last_v * DT)
                if new_s[idx] is None:
                    new_s[idx], new_v[idx] = s2, last_v
                k += 1
        report[ot] = log
        for i in range(len(times)):
            if new_s[i] is not None:
                x, y, lane, pos, ang = locate(segs, cuml, new_s[i])
                frames[i][ot] = [x, y, lane, pos, new_v[i], ang]

    for i, t in enumerate(times):
        s, v = vut_state(t)
        if s is None or s >= s_tot - 1e-6:
            continue
        x, y, lane, pos, ang = locate(segs_v, cuml_v, s)
        frames[i]["VUT"] = [x, y, lane, pos, round(v, 2), ang]

    print("\nOT dwell re-anchoring (OT, task, dwell start -> end, VUT trigger time):")
    for ot, log in report.items():
        for name, a, b, tgt in log:
            late = (a - tgt) if tgt is not None and a > tgt + 1e-6 else 0.0
            print("  %-4s %-6s %6.1f -> %6.1f   trigger %.1f%s"
                  % (ot, name, a, b, tgt or -1,
                     "   !! OT arrives %.1f s after the VUT crosses the trigger" % late
                     if late else ""))

    smooth_headings(frames, vids)
    write_fcd(frames, times, vids,
              "VUT continuous, OT dwell re-anchored to the VUT trigger crossings")
    write_metrics(frames, times, vids)
    print("\nwrote output/replay.fcd.xml, speed.csv, relative_distance.csv")


if __name__ == "__main__":
    main()
