"""协同任务处理：从一个触发节点到一个共同的释放时钟。

本文件是 SUMO 场景处理协同任务的核心逻辑，回答三个问题：

  1. 哪些任务是协同任务？  触发节点相同的任务属于同一个协同组，
                           对应 src/core/config.py 中的 Task.vut_trigger
  2. 它们何时释放？        VUT 通过该触发节点的时刻
  3. 为什么同组任务同时发生？
        VUT 只会通过一个节点一次，锚定在该节点上的所有任务共享同一个命中时刻 t_hit，
        因此共享同一个释放时刻 t_release = t_hit + min(service, OT_RELEASE_DWELL)
        和同一个时间窗 [t_hit - pre_slack, t_hit + service + post_slack]。

调度层必须满足两点：

  * 同一台 OT 不能服务同一组中的两个成员，因为它不能同时停在两个任务点；
  * 任务的理论延迟等于 VUT 从本任务第一个触发点行驶到自己触发点的时间。
    单触发任务和同触发点的任务对为 0，两个子任务由不同节点触发时为两节点之间的行驶时间。

分组、窗口、释放和延迟函数是纯 Python 实现。
main 还需要 sumolib 和自备的 network/map.net.xml，用来测量触发节点在 VUT 路线上的位置：

    python cooperative_schedule.py
"""
from collections import OrderedDict

from replay_utils import (DT, build_path, family, integrate_speed, load_network,
                          load_routes, load_speeds, load_tasks)

OT_RELEASE_DWELL = 3.0   # 服务时间对 VUT 的占用上限，单位 s
PRE_SLACK = 0.0          # 时间窗在触发命中时刻打开
POST_SLACK = 0.0         # 时间窗在命中后 service 秒关闭


# --------------------------------------------------------------- 四条规则
def group_by_trigger(tasks):
    """规则 1：由同一个触发节点触发的任务构成一个协同组。"""
    groups = OrderedDict()
    for t in tasks:
        groups.setdefault(t.trigger, []).append(t)
    return groups


def group_window(t_hit, members, pre_slack=PRE_SLACK, post_slack=POST_SLACK):
    """规则 2：每组一个时间窗，所有成员共享。"""
    service = max(m.service for m in members)
    return max(0.0, t_hit - pre_slack), t_hit + service + post_slack


def group_release(t_hit, members, dwell=OT_RELEASE_DWELL):
    """规则 3：每个节点一个释放时钟，所有成员在这一时刻释放。"""
    return t_hit + min(max(m.service for m in members), dwell)


def group_is_feasible(members):
    """规则 4：组内每个成员一台 OT，同一台车不能同时位于两个任务点。"""
    ots = [m.ot for m in members]
    return len(set(ots)) == len(ots)


def theoretical_delays(tasks, hit_times):
    """每个任务相对其任务族第一个触发点的延迟，hit_times 为节点到命中时刻的映射。"""
    first = {}
    for t in tasks:
        h = hit_times.get(t.trigger)
        if h is not None and (family(t.name) not in first or h < first[family(t.name)]):
            first[family(t.name)] = h
    return {t.name: (hit_times[t.trigger] - first[family(t.name)]
                     if hit_times.get(t.trigger) is not None else None)
            for t in tasks}


# ------------------------------------------------------------ 依赖路网的部分
def trigger_distances(net, idmap, vut_edges):
    """VUT 到达各节点时的沿路线里程，单位 m。

    取进入该节点的那条边末端之前所有车道长度之和，
    data/tasks.csv 中的 trigger_s 即为该数值。
    """
    ends, acc = {}, 0.0
    for e in vut_edges:
        lane = net.getEdge(idmap.edge(e)).getLane(0)
        acc += lane.getLength()
        ends[net.getEdge(idmap.edge(e)).getToNode().getID()] = acc
    return ends, acc


def _fmt(x, spec="%.1f"):
    return spec % x if x is not None and x == x else "   n/a"


def main():
    net, idmap = load_network()
    tasks = load_tasks()
    route = load_routes()
    vut_edges = route[route.vehicle == "VUT"].sort_values("seq").edge.tolist()
    node_s, _ = trigger_distances(net, idmap, vut_edges)
    total = build_path(net, idmap, vut_edges)[1][-1]   # 含路口内部车道的折线长度

    # 由记录的速度曲线积分得到 VUT 沿路线的位置，时间步长 0.2 s
    speed = load_speeds()
    t = speed["time"].to_numpy(float)
    s_t = integrate_speed(speed["VUT"].to_numpy(float), DT)

    def hit_time(node):
        sk = node_s.get(idmap.node(node))
        if sk is None:
            return None
        for i in range(1, len(s_t)):
            if s_t[i] is not None and s_t[i] >= sk:
                return float(t[i])
        return None

    groups = group_by_trigger(tasks)
    hits = {node: hit_time(node) for node in groups}

    print("VUT route %.1f m, %d edges" % (total, len(vut_edges)))
    print("\n%-8s %-8s %9s %10s" % ("task", "trigger", "s (m)", "t_hit (s)"))
    for task in tasks:
        sk = node_s.get(idmap.node(task.trigger))
        print("%-8s %-8s %9s %10s" % (task.name, task.trigger, _fmt(sk),
                                      _fmt(hits[task.trigger])))

    print("\n=== cooperative groups (one VUT crossing per node) ===")
    for node, members in groups.items():
        tk = hits[node]
        if tk is None:
            print("\n  trigger %s   not reached by the VUT" % node)
            continue
        a, b = group_window(tk, members)
        rel = group_release(tk, members)
        kind = "COOPERATIVE (simultaneous)" if len(members) > 1 else "single"
        print("\n  trigger %s   t_hit=%.1f s   window=[%.1f, %.1f]   release=%.1f s   %s"
              % (node, tk, a, b, rel, kind))
        for m in members:
            print("      %-6s %-4s service %4.1f s" % (m.name, m.ot, m.service))
        if len(members) > 1:
            print("      -> released at the SAME instant %.1f s" % rel)
            if group_is_feasible(members):
                print("      -> distinct OTs, each parked at its own point: OK")
            else:
                print("      !! the same OT would serve two members: impossible, "
                      "one of them must be moved to another OT")

    print("\n=== theoretical delay (VUT travel from the family's first trigger) ===")
    delays = theoretical_delays(tasks, hits)
    for task in tasks:
        d = delays[task.name]
        print("  %-6s %-8s theo = %s s" % (task.name, task.trigger, _fmt(d, "%5.2f")))


if __name__ == "__main__":
    main()
