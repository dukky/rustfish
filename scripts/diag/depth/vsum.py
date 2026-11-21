import json, sys
DB, K = 16, 16
m = json.load(open(sys.argv[1]))
g = lambda k, i: (m.get(k, []) + [0] * 100000)[i]
NT = ["PV", "CUT", "ALL"]
GR = [("5", [5]), ("6-7", [6, 7]), ("8-9", [8, 9]), ("10-11", [10, 11]), ("12+", [12, 13, 14, 15])]
print("Root LMR verify (reduced root move failed low; full depth > alpha?)  [all | ties at alpha | below alpha]")
for d in range(5, 16):
    t = [g("vroot_try", (0 * DB + d) * K + k) for k in (0, 1)]
    e = [g("vroot_err", (0 * DB + d) * K + k) for k in (0, 1)]
    if sum(t):
        print("  d%-2d  %5d err %5.1f%% | ties %4d err %5.1f%% | below %4d err %5.1f%%" % (
            d, sum(t), 100 * sum(e) / sum(t), t[1], 100 * e[1] / max(t[1], 1), t[0], 100 * e[0] / max(t[0], 1)))
print("Interior LMR verify (reduced quiet failed low; depth-1 search > alpha?), by node type and depth")
for nt in range(3):
    for name, ds in GR:
        t = sum(g("vlmr_try", (nt * DB + d) * K + k) for d in ds for k in range(16))
        e = sum(g("vlmr_err", (nt * DB + d) * K + k) for d in ds for k in range(16))
        tt = sum(g("vlmr_try", (nt * DB + d) * K + k) for d in ds for k in range(8, 16))
        et = sum(g("vlmr_err", (nt * DB + d) * K + k) for d in ds for k in range(8, 16))
        byr = []
        for r in range(1, 8):
            tr = sum(g("vlmr_try", (nt * DB + d) * K + r) + g("vlmr_try", (nt * DB + d) * K + r + 8) for d in ds)
            er = sum(g("vlmr_err", (nt * DB + d) * K + r) + g("vlmr_err", (nt * DB + d) * K + r + 8) for d in ds)
            if tr >= 50:
                byr.append("r%d %.1f%%(%d)" % (r, 100 * er / tr, tr))
        if t:
            print("  %-3s d%-5s n=%7d err %5.2f%%  ties %5.1f%% of fails, err %5.2f%%   %s" % (
                NT[nt], name, t, 100 * e / t, 100 * tt / t, 100 * et / max(tt, 1), " ".join(byr)))
for key, lab in (("vneg", "negative ext (-1) vs depth-1"), ("vsing", "singular ext (+1) vs depth-1")):
    print(lab, ": outcome changed (beta side | alpha side)")
    for nt in range(3):
        for name, ds in GR + [("16", [])]:
            t = sum(g(key + "_try", nt * DB + d) for d in ds)
            e1 = sum(g(key, (nt * DB + d) * K + 1) for d in ds)
            e2 = sum(g(key, (nt * DB + d) * K + 2) for d in ds)
            if t:
                print("  %-3s d%-5s n=%6d  beta %5.2f%%  alpha %5.2f%%" % (NT[nt], name, t, 100 * e1 / t, 100 * e2 / t))
