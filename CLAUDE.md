# 项目:iBEC 竞赛参赛材料构建

## 任务性质
把一篇已完成的学术论文,改造成 iBEC(International Bioinformatics Engineering
Competition)初赛所需的三份提交材料,并补建一个论文缺失的工程化交付物。

## 参赛赛道
新型生物信息学算法 / 方法 (Novel Bioinformatics Algorithms / Methods)

## 论文信息(唯一事实来源)
标题:Beyond per-pair correlation: a drug-discrimination control exposes
prediction collapse in single-cell multi-drug perturbation models
作者:Xichen Zhao (Shenzhen MSU-BIT University, Department of Biology);
      Guanxing Chen (City University of Hong Kong (Dongguan), Department of
      Computer Science, 通讯作者)
状态:Preprint submitted to Elsevier
资助:Guangdong Basic and Applied Basic Research Foundation, No. 2025A1515110446
代码:https://github.com/schambergeredmund7111992-cmyk/cytobridge-benchmark
归档:Zenodo DOI 10.5281/zenodo.21912287(concept DOI 10.5281/zenodo.21911960)
许可:代码 MIT,衍生数据 CC BY 4.0
数据:sci-Plex (GEO GSE139944);Hallmark gene sets (MSigDB)

(注:早期提示词包中的 CytoBridge-Agent 链接与 DOI 21701930 已随仓库重建废弃,
以本文件与仓库实际状态为准。)

## 核心贡献(按竞赛创新性重要度重排,不是论文原顺序)
1. **Vehicle anchor 校准**:按 drug-cell pair 估计 vehicle,而非按 cell line
   池化。这是最原创的一步,使一个不含任何药物信息的预测器达到 DDC 0.588,
   反超所有被审模型。竞赛材料中必须作为第一创新点呈现。
2. **DDC(drug-discrimination control)**:off-diagonal 药物判别 AUC。论文
   诚实说明这是 "a within-context retrieval rank already in use",本项目的
   贡献是补上它缺失的 calibration 与明确的 chance baseline。
   **不得声称是全新发明的指标。**
3. **Oracle ladder**:把失效定位到 structure-to-response 映射的诊断范式。

## 关键数值(禁止改动、禁止四舍五入、禁止编造新数字)
- 四个被审模型的 DDC 全部落在 chance 0.5 的 ±0.05 内
- 无药物信息预测器(per drug-cell pair vehicle):DDC = 0.588
- 跨板生物学重复:per-pair Spearman 仅 0.21,但 DDC = 0.810(可达天花板)
- Hindsight-selected training compound:DDC = 0.926
- Annotated pharmacological target:DDC = 0.717
- Nearest Morgan-fingerprint neighbor:DDC = 0.509(近似随机)
- Morgan ridge over all training compounds:DDC = 0.495(近似随机)
- Ridge fingerprint baseline:DDC = 0.51(不 collapse 但仍然失败)
- chemCPA:DDC = 0.50;biolord:DDC = 0.50
- Cell-line Mean baseline:Spearman@50 = 0.491,击败所有 CytoBridge 配置
- Pathway 案例:r ≈ 0.95,但 on-vs-off-diagonal gap 仅 0.00006
- 七种 loss 配置均留下 r_inter > 0.98
- 论文设定的可用性阈值:0.70
- 留出集:9 drugs,27-pair target matrix

## 硬性规则
1. **不得编造任何数字、指标、图表、引用或实验结果。** 所有定量陈述必须能在
   paper.pdf 或仓库中找到出处。找不到出处时,写 `[TODO: 需人工确认]`,
   不要猜测填充。
2. 主语言英文。中文仅在必要处补充。
3. CytoBridge **不是**一个可用的预测器,它是被诊断的对象。任何材料中都不得
   把它描述成有效模型或宣称其性能优势。这是论文的核心立场,写反了会直接
   摧毁可信度。
4. 这是诊断性工作,不声称提出了更强的预测模型。Scope boundary 必须显式声明。
5. 遇到需要主观判断或事实缺口时,停下来问我,不要自行发挥。
6. 每完成一个阶段,列出你写入/修改的全部文件路径,等我确认后再继续。

## 评分标准(所有产出都要往这五项上打)
- 原创性与创新性 25%
- 技术严谨性 25%
- 工程化设计与转化潜力 20%  ← **当前最薄弱,是本项目的主要补强目标**
- 科学与社会意义 20%
- 可重复性与数据透明度 10%
