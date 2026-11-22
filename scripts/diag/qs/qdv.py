"""Deficit tables: prune 'alpha - stand pat >= D' per (equal?, victim, gives check)."""
import sys
sys.argv, files = sys.argv[:1], sys.argv[1:]
exec(open(__file__.replace("qdv.py", "qan.py")).read().split("def main():")[0])
m, ds = load(files)
qn = sum(m["q_nodes"])
VK = ["P", "N", "B", "R", "Q", "promo"]
groups = [("all qp", [0, 1, 2]), ("qp0", [0]), ("qp1+", [1, 2])]
for gname, gs in groups:
    print("\n=== %s" % gname)
    for eq in range(2):
        for gc in range(2):
            for vk in range(6):
                rows = []
                for b in range(SB):
                    n = d = sub = 0
                    for gq in gs:
                        di = (((gq * 2 + eq) * 6 + vk) * 2 + gc) * SB + b
                        n += g(m, "q_dv", di * 2); d += g(m, "q_dv", di * 2 + 1); sub += g(m, "q_dvsub", di)
                    rows.append((n, d, sub))
                tot = sum(r[0] for r in rows)
                if tot < 200: continue
                out = []
                # cumulative from the top (largest deficit) down
                cn = cd = cs = 0
                cum = {}
                for b in range(SB - 1, -1, -1):
                    cn += rows[b][0]; cd += rows[b][1]; cs += rows[b][2]
                    cum[slack(b)] = (cn, cd, cs)
                for D in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
                    cn, cd, cs = cum.get(D, (0, 0, 0))
                    if cn >= 1:
                        out.append("D%d:%.1f%%/%.2f%%/%.2f" % (D, 100 * cn / tot, 100 * cd / cn, 100 * cs / qn))
                print("%s %s chk=%d n=%.0f det=%.1f%%  [D: share pruned / det rate / qs-subtree%%] %s" % (
                    "EQ " if eq else "WIN", VK[vk], gc, tot, 100 * sum(r[1] for r in rows) / tot, " ".join(out)))
