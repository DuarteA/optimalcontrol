"""Produce every number and figure quoted on the Unit 9 closing slides."""
import json, time, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import vehicle as V

OUT = {}
t_start = time.time()


def log(*a):
    print("[%7.1fs]" % (time.time() - t_start), *a, flush=True)


# --------------------------------------------------------------- problem A
log("baseline")
base = V.eco_constant_speed()
OUT["baseline_MJ"] = base["energy"] / 1e6
OUT["baseline_peak_F"] = base["peak_force"]
log("  baseline %.6f MJ, peak |F| %.0f N" % (base["energy"] / 1e6, base["peak_force"]))

log("audit dt sensitivity on the baseline itself")
for dt in (1e-2, 5e-3, 1e-3):
    a = V.eco_audit(base, dt=dt)
    log("  dt=%.0e  E=%.6f MJ  miss_s=%.3f m" % (dt, a["energy"] / 1e6, a["miss_s"]))

eco = {}
rows_A = []
for sch in ("trapezoidal", "hermite-simpson"):
    for N in (20, 40, 80):
        t0 = time.time()
        r = V.eco_solve(N=N, scheme=sch, maxiter=300)
        a = V.eco_audit(r, dt=5e-3)
        eco[(sch, N)] = (r, a)
        rows_A.append(dict(scheme=sch, N=N,
                           E_solver=r["energy"] / 1e6,
                           E_audit=a["energy"] / 1e6,
                           saving=100 * (base["energy"] - a["energy"]) / base["energy"],
                           miss_s=a["miss_s"], miss_v=a["miss_v"],
                           status=r["status"], nit=r["nit"],
                           defect=r["defect_norm"]))
        log("  A %-16s N=%3d  E_solver=%.6f  E_audit=%.6f  save=%.2f%%  "
            "miss_s=%+.3f m  miss_v=%+.4f m/s  [%.1fs]"
            % (sch, N, r["energy"] / 1e6, a["energy"] / 1e6,
               rows_A[-1]["saving"], a["miss_s"], a["miss_v"], time.time() - t0))
OUT["A_rows"] = rows_A

best = eco[("hermite-simpson", 80)][0]
OUT["A_profile_points"] = {
    lbl: float(np.interp(s_at, best["s"], best["v"]))
    for lbl, s_at in (("start", 0.0), ("foot_climb", V.HILL_START),
                      ("crest", V.HILL_START + V.HILL_LENGTH / 2),
                      ("foot_descent", V.HILL_START + V.HILL_LENGTH),
                      ("end", V.ROUTE_LENGTH))}
OUT["A_v_range"] = [float(best["v"].min()), float(best["v"].max())]
log("  profile: %s" % OUT["A_profile_points"])

# --------------------------------------------------------------- problem B
log("time-optimal reference")
t0 = time.time()
T_ref, t_sw, ref = V.time_optimal_reference(dt=5e-3)
OUT["B_T_ref"] = T_ref
OUT["B_t_switch"] = t_sw
OUT["B_peak_v"] = float(ref["v"].max())
log("  T*=%.6f s  switch=%.6f s  peak v=%.3f m/s  [%.1fs]"
    % (T_ref, t_sw, ref["v"].max(), time.time() - t0))

rows_B = []
spr = {}
for sch in ("trapezoidal", "hermite-simpson"):
    for N in (10, 20, 40, 80):
        t0 = time.time()
        r = V.sprint_collocation(N=N, scheme=sch, maxiter=300)
        spr[(sch, N)] = r
        rows_B.append(dict(scheme=sch, N=N, T=r["T"], err=r["T"] - T_ref,
                           status=r["status"], nit=r["nit"],
                           defect=r["defect_norm"]))
        log("  B %-16s N=%3d  T=%.6f  err=%+.6f  [%.1fs]"
            % (sch, N, r["T"], r["T"] - T_ref, time.time() - t0))
t0 = time.time()
ms = V.sprint_multiple_shooting(N=40, S=8, maxiter=300)
rows_B.append(dict(scheme="multiple shooting (S=8)", N=40, T=ms["T"],
                   err=ms["T"] - T_ref, status=ms["status"], nit=ms["nit"],
                   defect=ms["con_norm"]))
log("  B multiple shooting S=8 N=40  T=%.6f  err=%+.6f  [%.1fs]"
    % (ms["T"], ms["T"] - T_ref, time.time() - t0))
OUT["B_rows"] = rows_B

# ------------------------------------------------------------------ figures
log("figures")
FIGDIR = "figures"
import os
os.makedirs(FIGDIR, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "figure.dpi": 150})

# Figure 1: the eco-driving solution against the route
fig, ax = plt.subplots(3, 1, figsize=(8.2, 7.4), sharex=True,
                       gridspec_kw={"height_ratios": [1.0, 1.6, 1.3]})
ss = np.linspace(0, V.ROUTE_LENGTH, 1000)
ax[0].fill_between(ss, 0, V.elevation(ss), color="0.75", lw=0)
ax[0].set_ylabel("elevation [m]")
ax[0].set_ylim(0, V.HILL_HEIGHT * 1.35)
r = eco[("hermite-simpson", 80)][0]
ax[1].plot(base["s"], base["v"] * 3.6, "k--", lw=1.4, label="constant speed (baseline)")
ax[1].plot(r["s"], r["v"] * 3.6, "C0-", lw=2.0, label="energy-optimal")
ax[1].set_ylabel("speed [km/h]")
ax[1].legend(loc="lower left", fontsize=9.5)
ax[2].axhline(0, color="0.6", lw=0.8)
ax[2].plot(base["s"], base["F"] / 1e3, "k--", lw=1.4)
ax[2].plot(r["s"], r["F"] / 1e3, "C0-", lw=2.0)
ax[2].set_ylabel("force [kN]")
ax[2].set_xlabel("distance along the route [m]")
for a_ in ax:
    a_.axvspan(V.HILL_START, V.HILL_START + V.HILL_LENGTH, color="C1", alpha=0.07, lw=0)
fig.tight_layout()
fig.savefig(f"{FIGDIR}/ev_eco_profile.png", bbox_inches="tight")
plt.close(fig)
log("  ev_eco_profile.png")

# Figure 2: the time-optimal solution and its grid-refinement error
fig, ax = plt.subplots(1, 3, figsize=(12.6, 3.9))
ax[0].plot(ref["t"], ref["v"] * 3.6, "k-", lw=2.2, label="reference (bang-bang)")
for N, c in ((10, "C0"), (40, "C2")):
    rr = spr[("trapezoidal", N)]
    ax[0].plot(rr["t"], rr["v"] * 3.6, c + "o-", ms=3.2, lw=1.2,
               label=f"trapezoidal, N={N}")
ax[0].set_xlabel("time [s]"); ax[0].set_ylabel("speed [km/h]")
ax[0].legend(fontsize=8.5, loc="lower center")

ax[1].step(np.append(ref["t"][:-1], ref["t"][-1]),
           np.append(ref["F"], ref["F"][-1]) / 1e3, "k-", where="post", lw=2.2)
for N, c in ((10, "C0"), (40, "C2")):
    rr = spr[("trapezoidal", N)]
    ax[1].plot(rr["t"], rr["F"] / 1e3, c + "o-", ms=3.2, lw=1.2)
ax[1].axvline(t_sw, color="C3", ls=":", lw=1.6)
ax[1].set_xlabel("time [s]"); ax[1].set_ylabel("force [kN]")
ax[1].set_title("the switch is what the grid cannot see", fontsize=10)

Ns = np.array([10, 20, 40, 80])
for sch, mk, c in (("trapezoidal", "o-", "C0"), ("hermite-simpson", "s-", "C2")):
    e = np.array([abs(spr[(sch, N)]["T"] - T_ref) for N in Ns])
    order = -np.polyfit(np.log(Ns), np.log(e), 1)[0]
    OUT.setdefault("B_fitted_order", {})[sch] = float(order)
    ax[2].loglog(Ns, e, mk, color=c, lw=1.7, ms=6,
                 label=f"{sch}  (order {order:.2f})")
e0 = abs(spr[("trapezoidal", 10)]["T"] - T_ref)
ax[2].loglog(Ns, e0 * (Ns / 10.0) ** -1.0, "k:", lw=1.2, label="first order")
ax[2].loglog(Ns, e0 * (Ns / 10.0) ** -2.0, "k--", lw=1.0, label="second order")
ax[2].set_xticks(Ns)
ax[2].set_xticklabels([str(N) for N in Ns])
ax[2].minorticks_off()
ax[2].set_xlabel("number of intervals $N$")
ax[2].set_ylabel(r"$|T_N - T^\star|$  [s]")
ax[2].set_title("both rates are the same", fontsize=10)
ax[2].legend(fontsize=8.5)
fig.tight_layout()
fig.savefig(f"{FIGDIR}/ev_time_optimal.png", bbox_inches="tight")
plt.close(fig)
log("  ev_time_optimal.png")

with open("results.json", "w") as f:
    json.dump(OUT, f, indent=2, default=float)
log("done")
