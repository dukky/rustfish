import subprocess, sys, statistics
fens=[l.strip() for l in open("/home/andreas/agent-runs/qs/pos962.txt") if l.strip()]
cmds=[]
for f in fens: cmds += ["position "+f, "pscale"]
cmds.append("quit")
out=subprocess.run(["./st"],input="\n".join(cmds)+"\n",capture_output=True,text=True).stdout
d={}
cur=[]
def flush():
    if not cur: return
    npc=sum(1 for k,_,_ in cur if 1<=k<=4)
    ph="all"
    for ph in ("all", "npc>=10" if npc>=10 else "npc6-9" if npc>=6 else "npc<=5"):
        for k,e,q in cur:
            d.setdefault((ph,k),([],[]))
            d[(ph,k)][0].append(e); d[(ph,k)][1].append(q)
for l in out.splitlines():
    t=l.split()
    if t and t[0]=="ps": cur.append((int(t[1]),int(t[2]),int(t[3])))
    elif t and t[0]=="psdone": flush(); cur=[]
for (ph,k),(e,q) in sorted(d.items()):
    e.sort(); q.sort()
    print(ph, "PNBRQ"[k], len(e), "static mean %.0f median %d iqr %d..%d" % (statistics.mean(e), e[len(e)//2], e[len(e)//4], e[3*len(e)//4]), " qs mean %.0f median %d iqr %d..%d" % (statistics.mean(q), q[len(q)//2], q[len(q)//4], q[3*len(q)//4]))
