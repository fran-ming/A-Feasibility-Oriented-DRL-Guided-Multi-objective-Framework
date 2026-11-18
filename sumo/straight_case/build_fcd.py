"""把记录的调度结果重建为 FCD 回放文件 output/replay.fcd.xml。

输入：data/route.csv  每辆车逐条边的路线，包含 VUT 和各 OT
      data/speed.csv  每辆车按 0.2 s 步长的速度，NaN 表示车辆不在场景中
输出：output/replay.fcd.xml、output/speed.csv、output/relative_distance.csv

每辆车的沿路线里程由记录速度积分得到，再映射到其路线的车道中心线上。
航向取所在车道的前进方向，而不是相邻两点的连线方向，
并做三点圆滑和每步 30 度的限制，保证路口转向连续。

本脚本按记录原样复现调度结果。
若需要把 VUT 速度曲线和 OT 停驻时间重新锚定到触发时刻，请使用 retime.py。

    python build_fcd.py
"""
from replay_utils import (DT, build_path, integrate_speed, load_network, load_routes,
                          load_speeds, locate, smooth_headings, vehicle_ids,
                          write_fcd, write_metrics)


def main():
    net, idmap = load_network()
    route, speed = load_routes(), load_speeds()
    vids = vehicle_ids(speed)
    times = [round(float(t), 2) for t in speed["time"]]

    paths = {v: build_path(net, idmap,
                           route[route.vehicle == v].sort_values("seq").edge.tolist())
             for v in vids}

    frames = [dict() for _ in times]
    for v in vids:
        segs, cuml = paths[v]
        vals = speed[v].to_numpy(float)
        for i, s in enumerate(integrate_speed(vals, DT)):
            if s is None:
                continue
            x, y, lane, pos, ang = locate(segs, cuml, s)
            frames[i][v] = [x, y, lane, pos, float(vals[i]), ang]

    smooth_headings(frames, vids)
    out = write_fcd(frames, times, vids,
                    "FCD replay rebuilt from data/speed.csv and data/route.csv")
    write_metrics(frames, times, vids)
    print("fcd timesteps %d, end %.1f s -> %s" % (len(times), times[-1], out))


if __name__ == "__main__":
    main()
