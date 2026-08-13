# T1a — Tahoe Morgan-FP Ridge control (赵希宸)

> 任务书 `CytoBridge_冲KBS实验补充_赵希宸.md` §T1a。导师已写好可跑骨架 + 自检,学生只需备数据、跑、回传数字。**不改 paper,不自算 metric。**

## 这是什么 / 为什么

诊断论文的核心是 **drug-discrimination control**:held-out 药预测器在单细胞上 per-pair 相关可以很高,却分不开不同的药(off-diagonal control AUC≈0.5)。sci-Plex 上 Ridge(Morgan)、chemCPA、CytoBridge 全过不了,只有 Oracle 过。

T1 要在**第二个独立数据集 Tahoe-100M** 上复现。**注意:之前 X2 用的是 drug one-hot Ridge,对没见过的 test 药必然平凡退化(每个 unseen 药 → 全零 one-hot → 预测完全相同,inter-drug=1.0),不是有意义的 linear-baseline failure,已从 paper 回退。** T1a 改用 **Morgan 指纹**(与 sci-Plex 同特征 → 可比),test 药有真实分子结构,control 才是公平检验。

## 你要准备的输入(data contract)

| 文件 | 内容 |
|---|---|
| `tahoe_slice.h5ad` | cell-level AnnData。**raw counts 放 `layers["counts"]`(优先)或 `X`**;`obs` 含药名列 + 细胞系列。 |
| `tahoe_drug_smiles.csv` | 两列 `drug_id,smiles`,**覆盖全部 train+test 药**。 |
| `tahoe_split.json` | `{"train": [...87药名...], "test": [...5药名...]}`。或不给 json、用 `--test_drugs "A,B,C,D,E"`,其余自动当 train。 |

### 拿 SMILES(导师已 de-risk 来源)

**主路(推荐):** Tahoe 官方 `drug_metadata` 表自带 `canonical_smiles` + `pubchem_cid`(X2 当初漏取)。

```python
from datasets import load_dataset
meta = load_dataset("tahoebio/Tahoe-100M", "drug_metadata", split="train").to_pandas()
meta = meta.rename(columns={"drug": "drug_id", "canonical_smiles": "smiles"})
meta[["drug_id", "smiles"]].dropna().drop_duplicates("drug_id").to_csv(
    "tahoe_drug_smiles.csv", index=False)
```

**备路(若 metadata 取不到):** 把药名写进 `names.txt`,复用仓库根的脚本(按名查 PubChem,输出同样的 `drug_id,smiles`):

```bash
python ../../../fetch_smiles_local.py names.txt tahoe_drug_smiles.csv
```

> 92 药(87 train + 5 test:APTO-253 / Bentamapimod / Capivasertib / Minodronic acid / NVP-BHG712)都是已知化合物,应能全部解析。**任一 train/test 药缺 SMILES,脚本会直接报错停**(不会静默丢药降低 power)。

## 跑

```bash
# 0) 先自检(任何有 numpy/scipy/sklearn 的 env 都行,不需 Tahoe 数据)
python tahoe_morgan_ridge.py --selftest         # 必须看到 SELFTEST PASSED

# 1) 真跑(cytobridge env,有 scanpy+rdkit + Tahoe h5ad 的机器)
python tahoe_morgan_ridge.py \
    --tahoe_h5ad   tahoe_slice.h5ad \
    --smiles_csv   tahoe_drug_smiles.csv \
    --splits_json  tahoe_split.json \
    --control_label DMSO_TF \
    --drug_col drug --cell_col cell_line \
    --out_dir .
```

> `--control_label` **必须改成你 h5ad 里真实的 vehicle 标签**(Tahoe 的 DMSO 标签和 sci-Plex 的 `DMSO` 不一定一样,先 `adata.obs['drug'].value_counts()` 看一眼)。`--drug_col/--cell_col` 同理对齐你的列名。

## 输出 + 回传契约

脚本写到 `--out_dir`:
- `tahoe_morgan_control_panel.csv` — Random / Mean / Ridge(Morgan-FP) / Oracle 四行,每行 `auc_deg50_pearson`、on/off-diag、gap、wilcoxon p、`auc_all_spearman`、`inter_drug_pearson`、n_pairs。
- `tahoe_ridge_pred.npy` / `tahoe_true.npy` / `tahoe_ridge_meta.csv` — 预测/真值/元信息(导师复核用)。

**交付给导师 = 在本目录建 `result.md`,填:**

```markdown
# T1a result
- 命令:<你跑的完整命令>
- well-posedness gate:Oracle AUC=___(须=1.0)/ Mean AUC=___(须≈0.5)→ PASS/FAIL
- Ridge(Morgan-FP) control:AUC@50-pearson=___,inter_drug_pearson=___,n_pairs=___
- test 药数=___,cell line 数=___,test pairs=___
- 一句话:Tahoe 上 Morgan-Ridge {过不了 / 过了} control(AUC ≈ ___)
- 异常/坑:<有就写,没有写无>
```

## 怎么读结果(两条分叉,都 OK,如实报)

- **Ridge AUC ≈ 0.5** → 线性 baseline 在 Tahoe 也分不开 held-out 药,**复现 sci-Plex 结论**(连 Morgan 线性都没真药物区分力),这是双数据集证据,最想要的结果。
- **Ridge AUC ≫ 0.5(明显 >0.5)** → 如实报告,不藏。说明 control 能区分好坏预测器(自带 positive-control 效果),导师据此调整 narrative。

## 红线(铁律)

1. **不重写 metric** — control 一律走 `code/eval/metrics.py::drug_discrimination_score`(脚本已复用)。
2. **well-posedness gate 不过(Oracle≠1.0 或 Mean 偏离 0.5)= 重建/单位坏了,数字作废**,脚本会自动退出,别绕过。
3. **进 paper 的数字导师复核后才写**(reported number 导师把关)。出现 AUC 反常 / 泄漏 / gate 不过 → 停,写 POSTMORTEM,别"再跑一次"。
4. T1a 只做 Morgan-Ridge(快赢);**CytoBridge-on-Tahoe 是 T1b**(整套嵌入 + GPU + control 粘合),导师另给 step-by-step。
