import cProfile, pstats, sys, os, io
sys.path.insert(0, "/scratch/work/zhangx29/Thesis_Project/4dvarnet_enkf")
sys.argv = ["run_enkf_baseline.py", "--dir", "check_outputs/_k1rate", "--frames", "20"]
import runpy
pr = cProfile.Profile(); pr.enable()
try:
    runpy.run_path("checks/run_enkf_baseline.py", run_name="__main__")
except SystemExit:
    pass
pr.disable()
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(18)
print("\n".join(l for l in s.getvalue().split("\n") if l.strip())[:3000])
