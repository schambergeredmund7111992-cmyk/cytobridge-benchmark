"""Rebuild the three composite figures in a strict Nature-subjournal style.
Wong (2011) colorblind-safe palette + cividis sequential maps. Reads ONLY the
verified data in analysis/data, analysis/data2, and the frozen E6E7 vectors;
no number is recomputed here, only the visual layer is redrawn."""
import os
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

MD = "/Users/cgxmac/Desktop/CytoBridge/manuscript"
NB = f"{MD}/notebooks"
os.makedirs(NB, exist_ok=True)

PARAMS = '''# === PARAMETERS (edit, then Run All) ===
D1 = "%s/analysis/data"; D2 = "%s/analysis/data2"; OUT = "%s/figs"
import numpy as np, pandas as pd, json, glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import rcParams
# --- Wong (2011) colorblind-safe palette (Nature Methods, "Points of View") ---
BLUE, GREEN, VERM, ORANGE, SKY, PURPLE = "#0072B2", "#009E73", "#D55E00", "#E69F00", "#56B4E9", "#CC79A7"
GRAY, DGRAY, INK, GRID = "#BDBDBD", "#6E6E6E", "#222222", "#EAEAEA"
from matplotlib.colors import LinearSegmentedColormap
SEQ = LinearSegmentedColormap.from_list("wong_seq", ["#F2F7FB", "#9CC3DE", "#3E8FC1", "#0072B2", "#023E66"])  # Wong-blue sequential, no yellow
DPI = 400
rcParams.update({
  "font.size":7, "font.family":"sans-serif", "font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
  "axes.spines.top":False, "axes.spines.right":False, "axes.linewidth":0.6, "axes.edgecolor":INK,
  "text.color":INK, "axes.labelcolor":INK, "xtick.color":INK, "ytick.color":INK,
  "axes.labelsize":7.5, "xtick.labelsize":6.5, "ytick.labelsize":6.5,
  "xtick.major.width":0.6, "ytick.major.width":0.6, "xtick.major.size":2.4, "ytick.major.size":2.4,
  "legend.fontsize":6, "pdf.fonttype":42, "ps.fonttype":42, "savefig.dpi":DPI, "figure.dpi":150,
})
def panel(ax, L, sub=""):
    ax.text(-0.15, 1.13, L, transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="top", ha="right")
    if sub: ax.set_title(sub, fontsize=6.6, loc="center", pad=3, color="#555555")
def cbar(fig, im, ax, label=""):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=5.5, width=0.5, length=2)
    cb.outline.set_linewidth(0.5)
    if label: cb.set_label(label, fontsize=6)
    return cb
''' % (MD, MD, MD)

# ---------- Figure 3: collapse overview, 8 panels ----------
FIG3 = '''fig = plt.figure(figsize=(7.2, 4.9)); gs = gridspec.GridSpec(2,4,figure=fig,hspace=0.64,wspace=0.66)
# (a) per-pair Spearman@50: no configuration beats Mean
ax=fig.add_subplot(gs[0,0]); t1=pd.read_csv(f"{D1}/table1_regenerated.csv")
cb=t1[~t1.predictor.str.contains("Ridge|chemCPA|Mean",case=False,na=False)].sort_values("spearman50_ondiag")
labs=list(cb.predictor)+["Ridge","chemCPA"]; vals=list(cb.spearman50_ondiag)+[0.331,0.086]
colors=[BLUE]*len(cb)+[DGRAY,GRAY]
ax.barh(range(len(labs)),vals,color=colors,height=0.72,edgecolor="white",lw=0.4,zorder=3)
ax.axvline(0.402,color=GREEN,lw=1.3,zorder=4)
ax.text(0.412,len(labs)/2.0,"Mean = 0.402",ha="left",va="center",rotation=90,fontsize=6,color=GREEN,fontweight="bold")
ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs,fontsize=5.8)
ax.set_xlabel("Spearman@50 (on-diagonal)"); ax.set_xlim(0,0.45)
ax.xaxis.grid(True,color=GRID,lw=0.5); ax.set_axisbelow(True)
ax.legend(handles=[Patch(color=BLUE,label="CytoBridge"),Patch(color=DGRAY,label="Ridge"),Patch(color=GRAY,label="chemCPA")],
          fontsize=5.2,frameon=False,loc="lower right",handlelength=1.0,handleheight=1.0,labelspacing=0.25,borderpad=0.2)
panel(ax,"a")
# (b-d) inter-drug r heatmaps per cell line, shared cividis colorbar on (d)
ims=[]
for k,(cpos,cell) in enumerate([(gs[0,1],"A549"),(gs[0,2],"K562"),(gs[0,3],"MCF7")]):
    ax=fig.add_subplot(cpos); Mc=np.load(f"{D2}/interdrug_{cell}.npy")
    im=ax.imshow(Mc,vmin=0.9,vmax=1.0,cmap=SEQ); ims.append((im,ax))
    ax.set_xticks([]); ax.set_yticks([]); panel(ax,"bcd"[k],cell)
cbar(fig, ims[-1][0], ims[-1][1], "inter-drug $r$")
# (e) per-cell mean inter-drug r
ax=fig.add_subplot(gs[1,0]); pc=pd.read_csv(f"{D2}/per_cellline.csv").groupby("cell").inter_drug_r.mean()
ax.bar(range(len(pc)),pc.values,color=BLUE,width=0.62,edgecolor="white",lw=0.5,zorder=3)
ax.set_xticks(range(len(pc))); ax.set_xticklabels(pc.index,fontsize=6.5); ax.set_ylim(0.9,1.0)
ax.set_ylabel("mean inter-drug $r$"); ax.yaxis.grid(True,color=GRID,lw=0.5); ax.set_axisbelow(True); panel(ax,"e")
# (f) per-gene s.d. across drugs, predicted vs true
ax=fig.add_subplot(gs[1,1]); gv=np.load(f"{D2}/gene_variance.npz")
ax.hist(gv["true_std"],bins=40,color=GREEN,alpha=0.65,label="true",log=True,edgecolor="none")
ax.hist(gv["pred_std"],bins=40,color=VERM,alpha=0.6,label="predicted",log=True,edgecolor="none")
ax.set_xlabel("per-gene s.d. across drugs"); ax.set_ylabel("genes")
ax.legend(fontsize=5.8,frameon=False,loc="upper right"); panel(ax,"f")
# (g) matched vs mismatched correlation distribution
ax=fig.add_subplot(gs[1,2]); oo=pd.read_csv(f"{D1}/onoff_loss-only.csv")
p=ax.violinplot([oo[oo.kind=="on"].score,oo[oo.kind=="off"].score],showmeans=True,widths=0.82)
for b,col in zip(p["bodies"],[BLUE,GRAY]): b.set_facecolor(col); b.set_alpha(0.65); b.set_edgecolor(col); b.set_linewidth(0.6)
for key in ["cmeans","cmaxes","cmins","cbars"]:
    if key in p: p[key].set_color(INK); p[key].set_linewidth(0.7)
ax.set_xticks([1,2]); ax.set_xticklabels(["matched","mismatched"],fontsize=6); ax.set_ylabel("top-50 correlation"); panel(ax,"g")
# (h) drug-identity confusion
ax=fig.add_subplot(gs[1,3]); A=np.load(f"{D2}/confusion_A549.npy"); im=ax.imshow(A,cmap=SEQ,vmin=0,vmax=0.4)
ax.set_xlabel("assigned drug"); ax.set_ylabel("true drug"); ax.set_xticks([]); ax.set_yticks([])
cbar(fig, im, ax, "P(assign)"); panel(ax,"h")
fig.savefig(f"{OUT}/fig_collapse_overview.pdf",bbox_inches="tight",dpi=DPI); print("fig3 done")'''

# ---------- Figure 4: control validation, 6 panels ----------
FIG4 = '''fig=plt.figure(figsize=(7.2,4.05)); gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.60,wspace=0.52)
# (a) calibration: AUC vs injected drug-signal fraction
ax=fig.add_subplot(gs[0,0]); cal=pd.read_csv(f"{D2}/calibration.csv"); meta=json.load(open(f"{D2}/calibration_meta.json"))
ax.plot(cal.alpha,cal.auc,"-o",color=BLUE,ms=3,lw=1.4,zorder=3,mec="white",mew=0.4)
ea=meta["effective_alpha"]
ax.axhline(0.583,color=VERM,ls="--",lw=0.9,zorder=2); ax.axvline(ea,color=VERM,ls="--",lw=0.9,zorder=2)
ax.scatter([ea],[0.583],color=VERM,s=28,zorder=5,ec="white",lw=0.5)
ax.annotate(f"CytoBridge\\nα\\u2248{ea:.2f}",xy=(ea,0.583),xytext=(ea+0.20,0.585),fontsize=6,color=VERM,va="center")
ax.set_xlabel("injected drug-signal fraction α"); ax.set_ylabel("discrimination AUC"); ax.set_ylim(0.45,1.03); ax.set_xlim(-0.02,1.02)
ax.grid(True,color=GRID,lw=0.5); ax.set_axisbelow(True); panel(ax,"a")
# (b) AUC ladder across predictors — rotated labels, no overlap
ax=fig.add_subplot(gs[0,1]); v=pd.read_csv(f"{D1}/control_validation.csv").set_index("predictor")
order=["Random (50 perm)","Mean (collapsed)","Ridge (clean)","chemCPA (collapsed)","biolord (collapsed)","CytoBridge (best)","Oracle (truth)"]; v=v.loc[order]
bar_c=[GRAY,GRAY,DGRAY,GRAY,GRAY,BLUE,GREEN]
ax.bar(range(7),v.auc,color=bar_c,width=0.70,edgecolor="white",lw=0.5,zorder=3)
ax.axhline(0.5,color=INK,ls=":",lw=0.8,zorder=2); ax.axhline(0.7,color=DGRAY,ls="--",lw=0.7,zorder=2)
ax.text(-0.45,0.715,"0.70",fontsize=5.4,color=DGRAY,va="bottom",ha="left")
ax.set_xticks(range(7)); ax.set_xticklabels(["Random","Mean","Ridge","chemCPA","biolord","CytoBridge","Oracle"],rotation=38,ha="right",fontsize=6)
ax.set_ylim(0,1.13); ax.set_ylabel("discrimination AUC")
for i,a in enumerate(v.auc): ax.text(i,a+0.025,f"{a:.2f}",ha="center",fontsize=5.8,fontweight="bold",color=INK)
ax.yaxis.grid(True,color=GRID,lw=0.5); ax.set_axisbelow(True); panel(ax,"b")
# (c) per-config discrimination AUC
ax=fig.add_subplot(gs[0,2]); cm=pd.read_csv(f"{D1}/config_metrics.csv")
ax.scatter(range(len(cm)),cm.auc,color=BLUE,s=30,zorder=3,ec="white",lw=0.5)
ax.axhline(0.5,color=INK,ls=":",lw=0.8); ax.axhline(0.7,color=DGRAY,ls="--",lw=0.7); ax.set_ylim(0.45,0.75)
ax.set_xticks([]); ax.set_xlabel("seven loss configs"); ax.set_ylabel("discrimination AUC")
ax.yaxis.grid(True,color=GRID,lw=0.5); ax.set_axisbelow(True); panel(ax,"c")
# (d) per-pair Spearman vs discrimination AUC (uncorrelated)
ax=fig.add_subplot(gs[1,0]); ax.scatter(cm.rho50,cm.auc,color=BLUE,s=30,ec="white",lw=0.5,zorder=3)
ax.axhline(0.5,color=INK,ls=":",lw=0.8); ax.set_xlabel("Spearman@50"); ax.set_ylabel("discrimination AUC")
ax.grid(True,color=GRID,lw=0.5); ax.set_axisbelow(True); panel(ax,"d")
# (e) bootstrap AUC — CI text in empty upper-left
ax=fig.add_subplot(gs[1,1]); b=np.load(f"{D2}/bootstrap_auc.npy"); bm=json.load(open(f"{D2}/bootstrap_meta.json"))
ax.hist(b,bins=30,color=GRAY,alpha=0.9,edgecolor="white",lw=0.2)
ax.axvspan(bm["auc_lo"],bm["auc_hi"],color=VERM,alpha=0.13,zorder=1)
ax.axvline(0.5,color=INK,ls=":",lw=1,zorder=4)
ax.set_xlabel("bootstrap AUC"); ax.set_ylabel("count")
ax.text(0.03,0.96,f"95% CI\\n[{bm['auc_lo']:.2f}, {bm['auc_hi']:.2f}]\\nincludes 0.5",transform=ax.transAxes,fontsize=5.8,va="top",ha="left",color=VERM)
panel(ax,"e")
# (f) on-minus-off gap per config
ax=fig.add_subplot(gs[1,2]); ax.bar(range(len(cm)),cm["gap"],color=BLUE,width=0.62,edgecolor="white",lw=0.5,zorder=3)
ax.set_xticks([]); ax.set_xlabel("seven loss configs"); ax.set_ylabel("on \\u2212 off gap"); ax.set_ylim(0,0.05)
ax.yaxis.grid(True,color=GRID,lw=0.5); ax.set_axisbelow(True); panel(ax,"f")
fig.savefig(f"{OUT}/fig_control_validation.pdf",bbox_inches="tight",dpi=DPI); print("fig4 done")'''

# ---------- Figure 5: mechanism + four cases, 8 panels ----------
FIG5 = '''fig=plt.figure(figsize=(7.2,4.15)); gs=gridspec.GridSpec(2,4,figure=fig,hspace=0.64,wspace=0.60)
# (a) loss components at convergence (log)
lc=pd.read_csv(f"{D1}/loss_components.csv"); names=["L_recon","L_kl","L_contrast","L_pathway","L_direction","L_drugspec","L_delta"]
vals=[float(lc[n].iloc[0]) for n in names]
ax=fig.add_subplot(gs[0,0]); ax.bar(range(len(names)),vals,color=[VERM]+[GRAY]*6,width=0.72,edgecolor="white",lw=0.4,zorder=3,log=True)
ax.set_xticks(range(len(names))); ax.set_xticklabels([n[2:] for n in names],rotation=40,fontsize=5.6,ha="right"); ax.set_ylabel("loss value (log)")
ax.text(0,vals[0]*1.6,f"{vals[0]:.0f}",ha="center",fontsize=6,fontweight="bold",color=VERM); ax.set_ylim(top=vals[0]*5); panel(ax,"a")
# (b) training trajectory: terms stay orders apart
ax=fig.add_subplot(gs[0,1])
mc=sorted(glob.glob("/Users/cgxmac/Desktop/CytoBridge/student_progress_E6E7/E6E7/logs/t7_sub_drugspec1/version_*/metrics.csv"))
md=pd.read_csv(mc[-1])
for col,lab,cc in [("train/L_recon_step","recon",VERM),("train/L_kl_step","kl",ORANGE),("train/L_contrast_step","contrast",GREEN),("train/L_pathway_step","pathway",BLUE)]:
    if col in md:
        s=md[["step",col]].dropna() if "step" in md else md[[col]].dropna().reset_index().rename(columns={"index":"step"})
        ax.plot(range(len(s)),s[col].values,"-",lw=1.1,color=cc,label=lab)
ax.set_yscale("log"); ax.set_xlabel("logged step"); ax.set_ylabel("loss value (log)")
ax.legend(fontsize=5.2,frameon=False,ncol=1,loc="center right",handlelength=1.3,labelspacing=0.3); panel(ax,"b")
# (c-f) four held-out drug pairs on A549
cs=json.load(open(f"{D2}/casestudies.json"))
pred0=np.load("/Users/cgxmac/Desktop/CytoBridge/student_progress_E6E7/E6E7/results/logfc_pred_t7_sub_loss_only.npy")
true0=np.load("/Users/cgxmac/Desktop/CytoBridge/student_progress_E6E7/E6E7/results/logfc_true_t7_sub_loss_only.npy")
meta=pd.read_csv("/Users/cgxmac/Desktop/CytoBridge/student_progress_E6E7/E6E7/results/logfc_meta_t7_sub_loss_only.csv")
idx=list(meta.index[meta.cell_line=="A549"]); pairs=[(0,1),(2,3),(4,5),(0,4)]; cells_pos=[gs[0,2],gs[0,3],gs[1,0],gs[1,1]]
for k,((i,j),gp) in enumerate(zip(pairs,cells_pos)):
    d=json.loads(cs[f"pair{k}"]); ax=fig.add_subplot(gp); a,bb=idx[i],idx[j]; L=4
    ax.plot([-L,L],[-L,L],ls="--",color=DGRAY,lw=0.6,zorder=1)
    ax.scatter(true0[a],true0[bb],s=3.5,color=GREEN,alpha=0.40,zorder=2,label="true",ec="none")
    ax.scatter(pred0[a],pred0[bb],s=3.5,color=VERM,alpha=0.40,zorder=3,label="predicted",ec="none")
    ax.set_xlim(-L,L); ax.set_ylim(-L,L); ax.set_aspect("equal")
    ax.text(0.5,1.04,f"pred {d['pred_r']:.2f} · true {d['true_r']:.2f}",transform=ax.transAxes,fontsize=5.7,ha="center",va="bottom",color="#555555")
    ax.text(-0.24,1.18,"cdef"[k],transform=ax.transAxes,fontsize=8.5,fontweight="bold",va="top",ha="right")
    ax.set_xlabel(d["A"][:11],fontsize=5.8); ax.set_ylabel(d["B"][:11],fontsize=5.8); ax.tick_params(labelsize=5)
    if k==0: ax.legend(fontsize=5,frameon=False,loc="lower right",handletextpad=0.2,borderpad=0.2)
# (g) pathway illusion: matched vs mismatched pathway r
ax=fig.add_subplot(gs[1,2]); ax.bar([0,1],[0.9484,0.9483],width=0.55,color=[BLUE,GRAY],edgecolor="white",lw=0.6,zorder=3)
ax.set_xticks([0,1]); ax.set_xticklabels(["matched","mismatched"],fontsize=6); ax.set_ylabel("pathway $r$"); ax.set_ylim(0,1.20)
ax.annotate("gap\\n0.00006",xy=(0.5,0.96),xytext=(0.5,1.12),fontsize=6,color=VERM,fontweight="bold",ha="center",va="top")
ax.yaxis.grid(True,color=GRID,lw=0.5); ax.set_axisbelow(True); panel(ax,"g")
# (h) effective drug-signal fraction recovered
ax=fig.add_subplot(gs[1,3]); ax.bar([0,1],[1.0,0.03],width=0.55,color=[GREEN,VERM],edgecolor="white",lw=0.6,zorder=3)
ax.set_xticks([0,1]); ax.set_xticklabels(["true\\nsignal","CytoBridge"],fontsize=6); ax.set_ylabel("effective drug-signal α"); ax.set_ylim(0,1.15)
ax.text(1,0.09,"0.03",ha="center",fontsize=6.5,fontweight="bold",color=VERM); ax.yaxis.grid(True,color=GRID,lw=0.5); ax.set_axisbelow(True); panel(ax,"h")
fig.savefig(f"{OUT}/fig_mechanism.pdf",bbox_inches="tight",dpi=DPI); print("fig5 done")'''

# ---------- build + execute the three notebooks ----------
NBS = {"fig1_collapse_overview.ipynb":("Figure: collapse overview (8 panels)",FIG3),
       "fig2_control_validation.ipynb":("Figure: control validation (6 panels)",FIG4),
       "fig3_mechanism_casestudy.ipynb":("Figure: mechanism and cases (8 panels)",FIG5)}
ep=ExecutePreprocessor(timeout=240,kernel_name="python3")
for fn,(title,code) in NBS.items():
    nb=nbf.v4.new_notebook()
    nb.cells=[nbf.v4.new_markdown_cell(f"# {title}\nNature-style rebuild. Edit PARAMS, Run All. Verified data in analysis/data, analysis/data2."),
              nbf.v4.new_code_cell(PARAMS), nbf.v4.new_code_cell(code)]
    nbf.write(nb,f"{NB}/{fn}"); ep.preprocess(nb,{"metadata":{"path":MD}}); nbf.write(nb,f"{NB}/{fn}")
    print("built",fn)
print("[done]")
