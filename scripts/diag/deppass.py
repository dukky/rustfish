import json, sys, subprocess, threading
ENG, IN, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
rs = [json.loads(l) for l in open(IN)]
sig = [r for r in rs if "a14" in r]
out = open(OUT, "w")
lock = threading.Lock()


def work(w, n):
    p = subprocess.Popen([ENG], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    def send(s):
        p.stdin.write(s + "\n"); p.stdin.flush()
    send("setoption name Hash value 64")
    for i in range(w, len(sig), n):
        r = sig[i]
        send("ucinewgame"); send(r["pos"]); send("go depth 18")
        per = {}
        while True:
            l = p.stdout.readline()
            t = l.split()
            if l.startswith("info depth") and "pv" in t:
                per[int(t[2])] = [t[t.index("pv") + 1], int(t[t.index("nodes") + 1])]
            if l.startswith("bestmove"):
                break
        with lock:
            out.write(json.dumps({"pos": r["pos"], "a": r["a"], "b": r["b"], "root": r["b14"] > r["a14"], "per": per}) + "\n"); out.flush()
    send("quit")


ts = [threading.Thread(target=work, args=(w, 4)) for w in range(4)]
[t.start() for t in ts]; [t.join() for t in ts]
