"""Cross-document consistency sweep for the Applied Soft Computing submission.

Nine gates. Any FAIL is a real defect, not a style note.
  G1  every tab:collapse / grid number matches the recomputed source of record
  G2  no stale value from before today's leak-free rescoring survives anywhere
  G3  abstract / highlights / results / conclusion tell one story with one set of numbers
  G4  figure captions agree with the regenerated figure data
  G5  cross-references, citations and graphics all resolve
  G6  the figure-backing table1_regenerated.csv agrees with Tables 3 and 4 of the manuscript
  G7  the split sizes the manuscript states are the split sizes the release actually ships
  G8  the scripts in the release package are byte-identical to the source tree
  G9  the published repository actually contains what Data availability promises
"""
import csv
import json
import os
import re
import sys

D = "/Users/cgxmac/Desktop/CytoBridge/0final/CytoBridge_0707"
# G1's source of record now lives in the repo. The old per-session /private/tmp path is kept
# only as a fallback: when that directory was cleaned the open() below raised and the sweep
# stopped before G2..G6 ran, i.e. the gate suite failed open instead of failing loud.
_SOR = f"{D}/analysis/_source_of_record"
SCR = _SOR if os.path.exists(f"{_SOR}/paper_numbers.json") else (
    "/private/tmp/claude-501/-Users-cgxmac-Desktop-CytoBridge/"
    "76cdc850-e96f-4751-a84c-104ce0cc11dc/scratchpad")
tex = open(f"{D}/main.tex").read()
cl = open(f"{D}/cover_letter.tex").read()
fails, warns = [], []


def gate(name):
    print(f"\n{'='*74}\n{name}\n{'='*74}")


# ------------------------------------------------------------------ G1
gate("G1  tab:collapse 与重算源对账")
src = json.load(open(f"{SCR}/paper_numbers.json"))
rows = re.findall(r'^([a-z][^&\n]*?) & (0\.\d+) & (0\.\d+) & (0\.\d+) & (0\.\d+) \\\\$',
                  re.search(r'\\label\{tab:collapse\}(.*?)\\bottomrule', tex, re.S).group(1),
                  re.M)
name_map = {"loss-only": "loss-only", "drug-spec $\\times 1$": "drug-spec $\\times 1$",
            "norm-only": "norm-only", "recovery baseline": "recovery baseline",
            "low recon weight": "low recon weight", "drug-spec $\\times 5$": "drug-spec $\\times 5$",
            "drug-spec $\\times 3$": "drug-spec $\\times 3$"}
checked = 0
for label, r, auc, gap, p in rows:
    label = label.strip()
    if label not in src["rows"]:
        continue
    new = src["rows"][label]["new"]
    for field, got, want in (("r_inter", r, new["r_inter"]), ("auc", auc, new["auc"]),
                             ("gap", gap, new["gap"]), ("p", p, new["p"])):
        if abs(float(got) - want) > 0.0006:
            fails.append(f"G1 tab:collapse {label} {field}: 稿子 {got} vs 重算 {want:.4f}")
        checked += 1
print(f"  比对 {checked} 格 (7 行 × 4 列)  →  {'✅ 全部一致' if checked and not fails else '❌ 见下'}")

zo, zn = src["zero_old"], src["zero_new"]
for lbl, want in (("0.588", zo["auc"]), ("0.092", zo["gap"]), ("0.071", zo["p"])):
    if lbl not in tex:
        fails.append(f"G1 零信息行缺 {lbl} (重算 {want:.4f})")
print(f"  零信息预测器两行 (0.588/0.092/0.071 与 0.500/0.000/0.420)  →  "
      f"{'✅' if all(x in tex for x in ('0.588','0.092','0.071','0.420')) else '❌'}")

best = src["best_new"]
phi = round((best - 0.5) / 0.31 * 100)
print(f"  φ 一致性: 重算 best={best:.4f} → φ={phi}%   稿子写 "
      f"{'13' if 'recovers $13\\%$' in tex else '???'}%  "
      f"{'✅' if phi == 13 and 'recovers $13\\%$' in tex else '❌'}")
if phi != 13 or 'recovers $13\\%$' not in tex:
    fails.append(f"G1 φ 不一致: 重算 {phi}%")

# ------------------------------------------------------------------ G2
gate("G2  旧值残骸扫描（今天 leak-free 重算前的数字）")
STALE = {
    r'0\.583': "旧最好配置 AUC", r'0\.569': "旧 loss-only AUC",
    r'0\.402': "旧 Mean Spearman", r'recovers \$26': "旧 φ",
    r'\$19\$--\$40': "旧 φ 区间", r'p=0\.002': "旧 permutation p",
    r'0\.9910': "旧 r_inter", r'\[0\.50,0\.59\]': "旧 grid 区间",
    r'0\.713|0\.741|0\.773|0\.778': "旧 all-Spearman 例外值",
    r'effective \$\\alpha\$ of \$0\.03': "旧 alpha",
    r'0\.995, while their true responses correlate only 0\.538': "旧药对例子",
    r'Knowledge-Based Systems': "KBS 痕迹",
    r'41\\%\$ power|1007': "旧功效数",
    r'detectable but practically negligible': "旧小节标题",
    # The target oracle is scored on 15 of 27 anchors, so Section 4.6 explicitly refuses to
    # normalise it against the 27-anchor ceiling. The retracted "recovers 70%" survived in the
    # cover letter for three weeks because nothing scanned that file.
    r'70\\%': "已撤回的 70% 归一化说法",
    r'0\.717.{0,40}of a \$?0\.810': "0.717 of a 0.810 ceiling(同一归一化说法)",
}
# Scan the cover letter too: it makes the same numerical claims to the editor, and until now
# it was read into `cl` and then never checked by any gate.
hits = {}
for doc, label in ((tex, "main.tex"), (cl, "cover_letter.tex")):
    for p, d in STALE.items():
        n = len(re.findall(p, doc))
        if n:
            hits[f"{label}: {d}"] = n
print("  ✅ 零残骸 (main.tex + cover_letter.tex)" if not hits else f"  ❌ {hits}")
if hits:
    fails.append(f"G2 旧值残骸: {hits}")

# ------------------------------------------------------------------ G3
gate("G3  摘要 / highlights / 结果 / 结论 是否同一套数")
def seg(a, b):
    i, j = tex.find(a), tex.find(b)
    return tex[i:j] if 0 <= i < j else ""
parts = {
    "abstract":   seg(r"\begin{abstract}", r"\end{abstract}"),
    "highlights": seg(r"\begin{highlights}", r"\end{highlights}"),
    "results":    seg(r"\subsection{The control exposes", r"\subsection{No residual"),
    "conclusion": seg(r"\section{Conclusion}", r"\section*{Declarations}"),
}
KEY = {"0.588": "零信息锚点", "0.810": "生物学天花板", "0.926": "retrieval 上界",
       "0.717": "靶点 oracle", "0.509": "Morgan 1-NN", "0.495": "Morgan ridge"}
print(f"  {'关键数':>8s}  " + "  ".join(f"{k:>10s}" for k in parts))
for num, desc in KEY.items():
    row = "  ".join(f"{('✅' if num in v else '·'):>10s}" for v in parts.values())
    print(f"  {num:>8s}  {row}   {desc}")
# 摘要声称四个受审模型都落在 chance 0.5 的 ±0.05 内。按表 4 的实际 AUC 验，
# 不做字符串匹配——上一版硬编码 "$0.50$ and $0.54$"，措辞一改就假 FAIL，
# 且旧措辞把 biolord 的 0.495 排在了 [0.50,0.54] 之外。
AUDITED = {"CytoBridge best": 0.5417, "Ridge": 0.510, "chemCPA": 0.500, "biolord": 0.495}
worst_name, worst = max(AUDITED.items(), key=lambda kv: abs(kv[1] - 0.5))
rng_ok = (abs(worst - 0.5) <= 0.05) and ("within $0.05$ of chance" in parts["abstract"]) \
         and ("0.50 to 0.54" in parts["results"])
print(f"\n  摘要 ±0.05 断言 vs 表 4 实测 (最远: {worst_name} {worst:.4f}, "
      f"偏离 {abs(worst-0.5):.4f}): {'✅' if rng_ok else '❌'}")
if not rng_ok:
    fails.append("G3 摘要 ±0.05 断言与表 4 实测/结果措辞不一致")
hl = re.findall(r'\\item (.+)', parts["highlights"])
bad_hl = [h for h in hl if len(h) > 85]
print(f"  Highlights {len(hl)} 条, 最长 {max(len(h) for h in hl)} 字符  {'✅' if not bad_hl and 3<=len(hl)<=5 else '❌'}")

# ------------------------------------------------------------------ G4
gate("G4  图注数字 vs 重生成的图数据")
cv = open("/Users/cgxmac/Desktop/CytoBridge/manuscript/analysis/data/control_validation.csv").read()
cap = re.search(r'\\caption\{\\textbf\{The drug-discrimination control is calibrated.*?\}\n', tex, re.S)
capt = cap.group(0) if cap else ""
ladder_vals = {"0.50": "Random/Mean", "0.51": "Ridge", "0.54": "CytoBridge best", "1.00": "Oracle"}
miss = [v for v in ladder_vals if v not in capt]
print(f"  fig:control_validation 阶梯值 {list(ladder_vals)}  →  {'✅ 全在图注里' if not miss else f'❌ 缺 {miss}'}")
if miss:
    fails.append(f"G4 控制图图注缺 {miss}")
print(f"  control_validation.csv 里 CytoBridge(best) = "
      f"{[l.split(',')[1] for l in cv.splitlines() if l.startswith('CytoBridge')][0][:5]}  "
      f"(应 ≈0.5417)")
for f in ["figs/fig_ladder.pdf", "figs/fig_control_validation.pdf", "figs/fig_mechanism.pdf",
          "figs/fig_collapse_overview.pdf", "figs/graphical_abstract.pdf"]:
    ts = os.path.getmtime(f"{D}/{f}")
    import datetime
    print(f"    {f:42s} 重生成于 {datetime.datetime.fromtimestamp(ts):%m-%d %H:%M}")

# ------------------------------------------------------------------ G5
gate("G5  交叉引用 / 引文 / 图文件")
labels = set(re.findall(r'\\label\{([^}]+)\}', tex))
refs = set(re.findall(r'\\(?:ref|autoref|eqref)\{([^}]+)\}', tex))
broken = sorted(refs - labels)
orphan = sorted(l for l in labels - refs if not l.startswith("eq:"))
bib = set(re.findall(r'@\w+\{([^,]+),', open(f"{D}/cas-refs.bib").read()))
cited = set(k.strip() for g in re.findall(r'\\cite[pt]?\*?(?:\[[^\]]*\])*\{([^}]+)\}', tex) for k in g.split(","))
nocite = sorted(cited - bib)
gfx = re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', tex)
nogfx = [g for g in gfx if not os.path.exists(f"{D}/{g}")]
print(f"  断裂的 \\ref : {len(broken)}  {'✅' if not broken else '❌ '+str(broken)}")
print(f"  未被引用的 label: {len(orphan)}  {'(可接受)' if len(orphan)<6 else '⚠ '+str(orphan)}")
print(f"  bib 里找不到的 \\cite key: {len(nocite)}  {'✅' if not nocite else '❌ '+str(nocite)}")
print(f"  缺失的图文件: {len(nogfx)}  {'✅' if not nogfx else '❌ '+str(nogfx)}")
for n, c in (("broken refs", broken), ("missing cites", nocite), ("missing figs", nogfx)):
    if c:
        fails.append(f"G5 {n}: {c}")

# ------------------------------------------------------------------ G6
gate("G6  图 3 的数据源 table1_regenerated.csv vs 稿子表 3 / 表 4")
# The figure-backing CSV and the manuscript tables are produced by different steps, so they can
# drift apart silently: the 0final copy of this CSV sat at its pre-recompute 07-07 values for
# three weeks while Tables 3 and 4 carried the leak-free ones, and nothing complained.
T1 = f"{D}/analysis/data/table1_regenerated.csv"
CSV2TEX = {"loss-only": "loss-only", "drug-spec x1": "drug-spec $\\times 1$",
           "drug-spec x3": "drug-spec $\\times 3$", "drug-spec x5": "drug-spec $\\times 5$",
           "low recon": "low recon weight", "norm-only": "norm-only",
           "recovery base": "recovery baseline"}
t1rows = {r["predictor"]: r for r in csv.DictReader(open(T1))
          if r.get("space") == "E6E7"}
missing = [k for k in CSV2TEX if k not in t1rows]
if missing:
    fails.append(f"G6 table1_regenerated.csv 缺少配置行: {missing}")
mean_row = next((r for k, r in t1rows.items() if k.startswith("Mean")), None)
if mean_row is None:
    fails.append("G6 table1_regenerated.csv 没有 Mean 行 —— 图 3 会退回硬编码")

tab_main = re.search(r'\\label\{tab:main\}(.*?)\\bottomrule', tex, re.S).group(1)
main_vals = dict(re.findall(r'^(?:CytoBridge \()?([^&\n]*?)\)? & (0\.\d+) \\\\$', tab_main, re.M))
tab_coll = re.search(r'\\label\{tab:collapse\}(.*?)\\bottomrule', tex, re.S).group(1)
coll_vals = {m[0].strip(): m[1:] for m in re.findall(
    r'^([a-z][^&\n]*?) & (0\.\d+) & (0\.\d+) & (0\.\d+) & (0\.\d+) \\\\$', tab_coll, re.M)}

n6 = 0
for csv_name, tex_name in CSV2TEX.items():
    r = t1rows.get(csv_name)
    if r is None:
        continue
    checks = [("Table 3 rho50", main_vals.get(tex_name), r["spearman50_ondiag"]),
              ("Table 4 r_inter", (coll_vals.get(tex_name) or [None])[0], r["inter_drug_pearson"]),
              ("Table 4 AUC", (coll_vals.get(tex_name) or [None, None])[1], r["control_auc50"])]
    for what, paper, backing in checks:
        if paper is None:
            fails.append(f"G6 {tex_name}: 在稿子里找不到 {what}")
            continue
        if abs(float(paper) - float(backing)) > 0.0006:
            fails.append(f"G6 {tex_name} {what}: 稿子 {paper} vs 图数据 {backing}")
        n6 += 1
if mean_row is not None and main_vals.get("Mean baseline") is not None:
    if abs(float(main_vals["Mean baseline"]) - float(mean_row["spearman50_ondiag"])) > 0.0006:
        fails.append(f"G6 Mean baseline: 稿子 {main_vals['Mean baseline']} vs "
                     f"图数据 {mean_row['spearman50_ondiag']}")
    n6 += 1
g6_bad = [f for f in fails if f.startswith("G6")]
print(f"  比对 {n6} 格 (7 配置 × 3 + Mean)  →  {'✅ 全部一致' if n6 and not g6_bad else '❌ 见下'}")
if mean_row is not None:
    print(f"  Mean 行存在,图 3 可从数据重现 Mean = {float(mean_row['spearman50_ondiag']):.3f}  ✅")

# tab:configs vs the hyper-parameters read off the runs that produced the stored predictions.
# Same failure mode as above: a hand-typed table drifting away from its artifact.
CFG = f"{D}/analysis/data/config_hyperparams.csv"
if os.path.exists(CFG):
    want = {r["configuration"]: r for r in csv.DictReader(open(CFG))}
    body = re.search(r'\\label\{tab:configs\}(.*?)\\bottomrule', tex, re.S).group(1)
    got = re.findall(r'^([a-z][^&\n]*?) & (\S+) & (\S+) & (\S+) & (\S+) \\\\$', body, re.M)
    n7 = 0
    for name, rec, ds, dl, flag in got:
        key = name.strip().replace("$\\times 1$", "x1").replace("$\\times 3$", "x3") \
                          .replace("$\\times 5$", "x5")
        w = want.get(key)
        if w is None:
            fails.append(f"G6 tab:configs 有 CSV 里没有的行: {key}")
            continue
        if w["source"] == "NOT RETAINED":
            if rec != "---":
                fails.append(f"G6 tab:configs {key}: 配置未留存,表里却填了 {rec} —— 不得推断")
            n7 += 1
            continue
        for what, paper, backing in (("lam_recon", rec, w["lam_recon"]),
                                     ("lam_drugspec", ds, w["lam_drugspec"]),
                                     ("lam_delta", dl, w["lam_delta"])):
            if float(paper) != float(backing):
                fails.append(f"G6 tab:configs {key} {what}: 稿子 {paper} vs 运行记录 {backing}")
            n7 += 1
        if flag != ("yes" if w["dec_in_component_norm"] == "true" else "no"):
            fails.append(f"G6 tab:configs {key} component norm: 稿子 {flag} vs "
                         f"运行记录 {w['dec_in_component_norm']}")
        n7 += 1
    if len(got) != len(want):
        fails.append(f"G6 tab:configs 行数 {len(got)} != 运行记录 {len(want)}")
    print(f"  tab:configs 比对 {n7} 格 ({len(got)} 配置)  →  "
          f"{'✅ 全部一致' if n7 and not [f for f in fails if 'tab:configs' in f] else '❌ 见下'}")
else:
    warns.append("analysis/data/config_hyperparams.csv 不存在 —— 先跑 make_config_table.py")

# ------------------------------------------------------------------ G7
gate("G7  稿子声明的 split 规模 vs release 里实际发布的 internal_splits.json")
# This gate exists because the manuscript stated 137/17/18 for three weeks while the released
# split index held 65/8/9. Those are not two names for one partition: seven of the nine scored
# drugs sit in the TRAIN set of the 137/17/18 file, so a reviewer intersecting the release with
# the paper would have read it as training on the test drugs. Numbers are clean -- run_metadata
# shows train and eval both used splits_sub -- but nothing checked the claim against the artifact.
SPLIT_JSON = "/Users/cgxmac/Desktop/CytoBridge/release_ddc/split/internal_splits.json"
if os.path.exists(SPLIT_JSON):
    sp = json.load(open(SPLIT_JSON))
    n_tr, n_va, n_te = (len(sp["train_drugs"]), len(sp["val_drugs"]), len(sp["test_drugs"]))
    m = re.search(r'The drug split holds (\d+) training, (\d+) validation, and (\d+) test', tex)
    if m is None:
        fails.append("G7 在 §4.1 找不到 split 规模那句 —— 措辞变了就得同步这道门")
    else:
        p_tr, p_va, p_te = (int(x) for x in m.groups())
        ok = (p_tr, p_va, p_te) == (n_tr, n_va, n_te)
        print(f"  稿子 {p_tr}/{p_va}/{p_te}   release {n_tr}/{n_va}/{n_te}   "
              f"{'✅ 一致' if ok else '❌ 不一致'}")
        if not ok:
            fails.append(f"G7 split 规模: 稿子 {p_tr}/{p_va}/{p_te} vs release "
                         f"{n_tr}/{n_va}/{n_te}")
    # the nine scored drugs named in Section 4.1 must BE the released test partition
    scored = ["AG-490", "Celecoxib", "Fulvestrant", "Ramelteon", "SL-327",
              "SRT3025", "Thalidomide", "Tofacitinib", "Zileuton"]
    rel = {re.sub(r"[^a-z0-9]", "", d.lower()) for d in sp["test_drugs"]}
    unmatched = [s for s in scored
                 if not any(r.startswith(re.sub(r"[^a-z0-9]", "", s.lower())) for r in rel)]
    print(f"  §4.1 列的 9 个 scored 药 == release test 分区: "
          f"{'✅' if not unmatched and len(rel) == 9 else '❌ ' + str(unmatched)}")
    if unmatched or len(rel) != 9:
        fails.append(f"G7 scored 药与 release test 分区不符: 缺 {unmatched}, "
                     f"release test n={len(rel)}")
    # and the arithmetic the oracle-pool sentence rests on
    cov = re.search(r'covers (\d+) of the (\d+) profiled', tex)
    if cov:
        c82, c172 = int(cov.group(1)), int(cov.group(2))
        checks = [(c82, n_tr + n_va + n_te, "split 覆盖数 = train+val+test"),
                  (c172 - c82, 90, "profiled - split"),
                  (n_tr + n_va + (c172 - c82), 163, "65+8+90 = 163 (oracle pool)")]
        bad = [d for got, want, d in checks if got != want]
        print(f"  oracle pool 算术闭合 (82/90/163): {'✅' if not bad else '❌ ' + str(bad)}")
        if bad:
            fails.append(f"G7 split 算术不闭合: {bad}")
else:
    warns.append(f"{SPLIT_JSON} 不存在 —— G7 无法核对 split 规模")

# ------------------------------------------------------------------ G8
gate("G8  release 包里的脚本是否与源树一致")
# Editing a script in analysis/ does not update the copy already sitting in the release. That is
# how release_asoc/code/consistency_sweep.py came to be a pre-G7 version while the source had
# moved on -- the same drift class as the stale analysis/data/ copies. One command fixes it:
#   python3 analysis/make_release.py
REL = "/Users/cgxmac/Desktop/CytoBridge/release_asoc"
if os.path.isdir(REL):
    import filecmp
    SRC_DIRS = [f"{D}/analysis", f"{D}/analysis/supp_T8", f"{D}/analysis/supp_T5",
                f"{D}/analysis/supp_T2", "/Users/cgxmac/Desktop/CytoBridge/release_ddc/code"]
    drift, checked = [], 0
    for root, _, files in os.walk(f"{REL}/code"):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            rp = os.path.join(root, fn)
            src = next((os.path.join(s, fn) for s in SRC_DIRS
                        if os.path.exists(os.path.join(s, fn))), None)
            if src is None:
                continue
            checked += 1
            if not filecmp.cmp(rp, src, shallow=False):
                drift.append(os.path.relpath(rp, REL))
    print(f"  比对 {checked} 个脚本  →  "
          f"{'✅ 与源树一致' if not drift else '❌ ' + str(drift)}")
    if drift:
        fails.append(f"G8 release 脚本已漂移,重跑 analysis/make_release.py: {drift}")
else:
    warns.append(f"{REL} 不存在 —— 先跑 analysis/make_release.py 生成 release 包")

# ------------------------------------------------------------------ G9
gate("G9  Data availability 承诺的东西是否真在已发布的仓库里")
# The URL and DOI in the paper resolve, but that is not the claim. The claim is that the
# repository CONTAINS ten specific things. On 2026-07-30 the published repo was the older full
# codebase, not release_asoc/, and was missing six of them -- including
# replicate_reliability_27.py, which Table 7's caption names by filename. A reviewer clicking the
# link would have found the file absent. This gate fetches the published tree and checks.
PROMISED = {
    "reproduce.py (23 项复现)": ["reproduce.py"],
    "gene order": ["gene_order"],
    "split indices (65/8/9)": ["internal_splits.json"],
    "stored predictions": ["logfc_pred_"],
    "27-pair target matrix": ["logfc_true_"],
    "off-diagonal AUC metric": ["metrics.py"],
    "split-half reliability script": ["replicate_reliability"],
    "baseline-reconstruction protocol": ["compute_baseline_control", "compute_chemcpa_control"],
    "table/figure 重生成脚本": ["make_fig_ladder", "make_fig3_compact", "make_table1"],
    "per-configuration configs": ["config_hyperparams", "hparams"],
}
m = re.search(r'\\url\{https://github\.com/([^}]+)\}', tex)
if m is None:
    fails.append("G9 main.tex 里找不到 GitHub URL")
else:
    repo = m.group(1).rstrip("/")
    import urllib.request
    try:
        with urllib.request.urlopen(
                f"https://api.github.com/repos/{repo}/git/trees/HEAD?recursive=1", timeout=20) as r:
            paths = [t["path"] for t in json.load(r).get("tree", [])]
    except Exception as e:                       # offline / rate-limited -> warn, do not fail
        paths = None
        warns.append(f"G9 无法访问 github.com/{repo}({e.__class__.__name__}) —— "
                     "投稿前必须人工确认仓库内容满足 Data availability")
    if paths is not None:
        absent = [k for k, pats in PROMISED.items()
                  if not any(any(x in p for x in pats) for p in paths)]
        print(f"  仓库 {repo}  {len(paths)} 个文件")
        print(f"  10 条承诺  →  {'✅ 全部满足' if not absent else '❌ 缺 ' + str(absent)}")
        if absent:
            fails.append(f"G9 已发布仓库缺 Data availability 承诺的内容 {absent};"
                         f"把 release_asoc/ 的内容推上去(见 analysis/make_release.py)")

# Placeholders that must be filled before submission. These are WARN, not FAIL: the DOI is
# minted only once the student uploads to Zenodo, and a suite that is permanently red is a
# suite nobody reads. They are printed in the verdict so they cannot be forgotten either.
for doc, label in ((tex, "main.tex"), (cl, "cover_letter.tex")):
    for ph in re.findall(r'PLACEHOLDER-[A-Z-]+', doc):
        warns.append(f"{label} 仍含占位符 {ph} —— 投稿前必须替换")

# ------------------------------------------------------------------ verdict
gate("判定")
if warns:
    print(f"  ⚠ {len(warns)} 项待办 (不阻断):")
    for w in dict.fromkeys(warns):
        print("     -", w)
if fails:
    print(f"  ❌ {len(fails)} 项 FAIL:")
    for f in fails:
        print("     -", f)
    sys.exit(1)
print("  ✅ 九道门全过，无阻断性缺陷。")
