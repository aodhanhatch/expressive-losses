# Summarise verify.py results: python ~/projects/summarize_ver.py <ver_folder> [n_images]
import sys, os, glob, pickle
d = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else None
files = {int(os.path.basename(f)[:-2]): f for f in glob.glob(os.path.join(d, "*.p"))}
idx = sorted(i for i in files if n is None or i < n)
top = n if n else (max(idx) + 1 if idx else 0)
missing = [i for i in range(top) if i not in files]
rows = [pickle.load(open(files[i], "rb")) for i in idx]
N = len(rows)
def pct(k):
    return 100.0 * sum(float(r.get(k, 0)) for r in rows) / N
print(d)
print("images with results: %d (indices %d..%d)" % (N, idx[0], idx[-1]))
if missing:
    print("MISSING indices below %d: %d  e.g. %s" % (top, len(missing), missing[:10]))
print("  standard accuracy        (ok):       %6.2f%%" % pct("ok"))
print("  adversarial accuracy     (pgd_ok):   %6.2f%%" % pct("pgd_ok"))
print("  verified, IBP only       (ibp_ok):   %6.2f%%" % pct("ibp_ok"))
print("  verified, CROWN only     (crown_ok): %6.2f%%" % pct("crown_ok"))
print("  verified, full BaB       (ver_ok):   %6.2f%%" % pct("ver_ok"))
