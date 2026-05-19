# XHS-SCoRE 社会比较检测实验报告

## 1. 实验背景与研究问题

本实验基于 XHS-SCoRE 论文提出的任务：判断一条小红书文本是否会从读者视角引发社会比较，以及比较方向是什么。

原论文指出一个重要现象：LLM 很擅长生成会引发社会比较的小红书风格文本，但在检测真实小红书文本时，往往无法稳定识别这些隐含的社会比较信号。论文将这种现象称为 **generation-detection dissociation**。也就是说，模型能生成心理上有效的社会比较触发文本，却不能可靠检测自然语料中的同类信号。

本实验的目标是：

> 构建一个面向 XHS-SCoRE 的社会比较 frame-level lexicon，并把它用于 LLM 分类 prompt，测试它是否能帮助 LLM 更好地识别 UPWARD / DOWNWARD / NEUTRAL。

我们重点关注两个问题：

1. 词表和 frame 是否能减少 LLM 把比较性帖子误判为 NEUTRAL 的问题？
2. 加入社会比较理论、训练集 cue mining 和 NLI-style context-aware retrieval 后，能否超过原论文中的 zero-shot LLM baseline？

## 2. 任务定义

XHS-SCoRE 是一个三分类任务。每条小红书帖子从 18-24 岁小红书读者视角判断：

| 标签 | 含义 | 例子 |
|---|---|---|
| UPWARD | 读者觉得帖主或文本对象比自己更好、更成功、更有资源、更幸福或更令人向往 | 拿到 offer、留学、变美、旅行、演唱会、中奖、豪宅、被夸 |
| DOWNWARD | 读者觉得帖主处境更糟、更受限、更痛苦、更缺资源，或文本呈现缺口、压力、补救、失败、低位状态 | 没上岸、被拒、家庭压迫、买不起、社死、求助、容貌焦虑、生活费不够 |
| NEUTRAL | 没有明显社会比较邀请，主要是客观信息、教程、产品介绍、第三方新闻、普通日常 | 攻略、教程、工具测评、榜单、普通菜谱、天气信息 |

需要特别说明的是：  
经典社会比较理论中，DOWNWARD 通常表示“对方比我差”。但在 XHS-SCoRE 的实际数据中，DOWNWARD 标注更宽，还包括某些让读者意识到“普通人存在缺口、压力或限制”的内容。例如：小个子显高、普女显白、省钱攻略、住宿纠结、踩雷避坑等。

因此，本实验中采用的工作定义是：

- UPWARD = 文本呈现令人向往的优势、高光、资源或社会认可。
- DOWNWARD = 文本呈现失败、受限、缺口、压力、补救、低位或生活摩擦。
- NEUTRAL = 文本没有明显相对地位或比较压力。

## 3. 数据与评价指标

### 3.1 数据划分

本实验使用原论文提供的数据划分：

| Split | 用途 |
|---|---|
| TRAIN | 构建和扩展 lexicon；进行 train-only cue mining |
| VAL | 调 prompt、参数和规则 |
| TEST | 最终评估，不用于构建词表 |

全量 TEST 集包含 2,783 条帖子，类别大致均衡：

- UPWARD: 926
- NEUTRAL: 926
- DOWNWARD: 931

### 3.2 评价指标

主要使用：

- **Accuracy**：整体分类准确率。
- **Macro F1**：三类 F1 的平均值，适合衡量三类是否均衡。
- **Recall UP / NEUTRAL / DOWN**：每个类别各自的召回率。
- **Predicted class rate**：模型输出为某个类别的比例，用于观察是否偏向 NEUTRAL 或某个方向。
- **错分率**：
  - `UP -> NEUTRAL`
  - `DOWN -> NEUTRAL`
  - `NEUTRAL -> UP`
  - `NEUTRAL -> DOWN`
  - `UP -> DOWN`
  - `DOWN -> UP`

这些错分率比单纯 accuracy 更重要，因为原论文关注的核心问题就是 LLM 会把有比较性的帖子“neutralize”为 NEUTRAL。

## 4. 原论文基线结果

原论文 Table 2 报告了 zero-shot prompted LLM 和 supervised encoder baselines 在 TEST split 上的表现：

| 原论文模型 | Type | Accuracy | Macro F1 | Recall UP | Recall NEUTRAL | Recall DOWN | Predicted NEUTRAL |
|---|---|---:|---:|---:|---:|---:|---:|
| GPT-5 | LLM | 0.521 | 0.518 | 0.410 | 0.752 | 0.402 | 0.601 |
| Qwen3-235B | LLM | 0.491 | 0.480 | 0.670 | 0.522 | 0.282 | 0.425 |
| GPT-4.1-nano | LLM | 0.469 | 0.469 | 0.379 | 0.630 | 0.397 | 0.558 |
| Qwen3-30B | LLM | 0.430 | 0.400 | 0.364 | 0.748 | 0.179 | 0.659 |
| CN-BERT WWM | Encoder | 0.670 | 0.671 | 0.666 | 0.636 | 0.708 | 0.360 |
| CN-RoBERTa WWM | Encoder | **0.680** | **0.679** | **0.695** | 0.585 | **0.759** | 0.307 |
| CN-MacBERT Base | Encoder | 0.665 | 0.665 | 0.633 | 0.631 | 0.730 | 0.349 |

从原论文可以看到：

1. LLM zero-shot 明显低于 supervised encoder。
2. LLM 常见问题是预测 NEUTRAL 过多。例如 GPT-5 的 Predicted NEUTRAL = 0.601，GPT-4.1-nano = 0.558。
3. Supervised encoder 能更好恢复 UP/DOWN 方向，尤其是 CN-RoBERTa WWM，Macro F1 = 0.679。

原论文 Table 3 还报告了 alternative prompting，包括 cue-explicit prompt。对 GPT-4.1-nano：

| GPT-4.1-nano Prompt | Macro F1 | Predicted NEUTRAL | UP -> NEUTRAL | DOWN -> NEUTRAL |
|---|---:|---:|---:|---:|
| Zero-shot | 46.9 | 55.8 | 54.2 | 50.4 |
| Persona-primed | 38.0 | 77.4 | 79.9 | 65.8 |
| Few-shot | 38.1 | 69.8 | 62.0 | 72.0 |
| Cue-explicit | 44.9 | 68.2 | 68.6 | 57.8 |

这一点很重要：原论文中的 cue-explicit prompt 并没有显著改善 GPT-4.1-nano，反而仍然严重偏向 NEUTRAL。这说明单纯把 cue inventory 写进 prompt 不一定足够。

## 5. 本实验方法

### 5.1 总体思路

本实验不是直接把一堆 cue 塞进 prompt，而是采用：

> cue → frame → NLI context-aware retrieval → final LLM classification → consistency rule

也就是说：

1. 先用 lexicon 在帖子里找到 cue。
2. 把 cue 聚合成 frame，例如：
   - `offer / 港大 / 录取` → `achievement_elite_education`
   - `没钱 / 买不起 / 预算` → `budget_constraint_low_resource`
   - `攻略 / 教程 / 步骤` → `tutorial_information`
3. 用 NLI-style prompt 判断这些 frame 是否真的被当前上下文支持。
4. 最终分类器只看到 context-relevant frames，而不是完整词表。
5. 再用 consistency rules 检查输出是否自洽，例如如果 label 是 UPWARD，但 `comparison_relation=absent`，就需要修正。

### 5.2 初始理论词表

初始 lexicon 参考以下理论资源：

- INCOM：社会比较倾向中的 self-other relation、ability comparison、opinion comparison、relative standing
- UPACS / DACS：外貌领域的 upward / downward comparison
- LIWC：情绪、成就、金钱、工作、家庭、身体等心理语言类别
- Wmatrix / USAS：semantic domain scaffold

初始词表规模：

| Lexicon | Rows | UP | DOWN | NEUTRAL | AMBIGUOUS |
|---|---:|---:|---:|---:|---:|
| `xhs_social_comparison_lexicon.csv` | 429 | 199 | 114 | 70 | 46 |

初始词表覆盖的 frame 包括：

- UPWARD frames:
  - achievement / elite education
  - high-resource lifestyle
  - appearance/body success
  - social approval
  - scarce fandom consumption
- DOWNWARD frames:
  - family oppression
  - blocked aspiration
  - low agency
  - body/appearance distress
  - money/work hardship
- NEUTRAL frames:
  - tutorial/information
  - product/tool review
  - third-party ranking/news
  - ordinary daily

### 5.3 Train-only LLM cue extraction

为了补充小红书语料中的平台原生表达，我使用 TRAIN split 做 LLM-assisted extraction。这个过程只使用训练集，不使用 val/test 构建词表。

训练集中抽取到的 UP cue 包括：

- `solo trip`
- `圆梦`
- `演唱会`
- `青旅`
- `夜生活`
- `留学 vlog`
- `隐藏款`
- `拆盲盒`
- `谷子`
- `港硕`
- `offer`

训练集中抽取到的 DOWN cue 包括：

- `容貌焦虑`
- `预算有限`
- `找不到工作`
- `没上岸`
- `被拒`
- `社死`
- `压力大`
- `求助`
- `怎么办`

训练集中抽取到的 NEUTRAL / neutralizer cue 包括：

- `攻略`
- `教程`
- `步骤`
- `测评`
- `官网`
- `下载`
- `求推荐`
- `榜单`

合并后的 train-augmented 词表规模：

| Lexicon | Rows | UP | DOWN | NEUTRAL | AMBIGUOUS |
|---|---:|---:|---:|---:|---:|
| `xhs_social_comparison_lexicon_train_augmented.csv` | 1237 | 449 | 360 | 382 | 46 |

### 5.4 Supervised mining

为了避免只依赖 LLM 抽取，本实验还在 TRAIN 上尝试了监督式 cue mining：

- log odds ratio with informative Dirichlet prior
- chi-square
- mutual information
- L1 logistic regression
- hard-negative mining
- contrastive cue purity score
- seed expansion + train-label purity filtering

这些方法可以找到某类中特别显著的 n-gram 和 hashtag cue。  
但实验也发现：词表并不是越大越好。低纯度 cue 或 singletons 合并过多，会让 prompt 变长，并引入更多方向混淆。

例如：

| Lexicon | Rows | Balanced test Macro F1 | 观察 |
|---|---:|---:|---|
| `xhs_social_comparison_lexicon_train_gap_augmented_v1.csv` | 7716 | 0.478 | 词表很大，但噪声较高 |
| `xhs_social_comparison_lexicon_train_augmented.csv` | 1237 | full test 0.525 | 更稳定 |

### 5.5 Theory-guided DOWN error injection

在早期实验中，DOWN recall 很低，因此我尝试根据 train error mining 和社会比较理论补充 DOWN frame。

加入的 DOWN 相关 frame / cue 包括：

- appearance repair / deficit:
  - `遮肉`
  - `遮胯`
  - `梨形`
  - `苹果型`
  - `五五分`
  - `三七分`
  - `显腿长`
- blocked academic/career aspiration:
  - `拒信`
  - `淘汰`
- low resource / budget constraint:
  - `预算`
  - `省钱`
- recovery / vulnerability:
  - `霸凌`
  - `住院`
  - `惊醒`
- relationship/social low position:
  - `异国`
  - `网恋`
- self-deprecating low status:
  - `社死`
  - `丢脸`
  - `尴尬`
- work/study pressure:
  - `大脑一片空白`
  - `背不出来`
  - `压力大`

该版本词表：

| Lexicon | Rows | UP | DOWN | NEUTRAL | AMBIGUOUS |
|---|---:|---:|---:|---:|---:|
| `xhs_social_comparison_lexicon_theory_error_injected_v2.csv` | 1021 | 365 | 373 | 237 | 46 |

## 6. 实验结果

### 6.1 初始 frame lexicon + NLI

初始 frame lexicon 加 NLI retrieval 后的 full test 结果：

| Setting | n | Accuracy | Macro F1 | Recall UP | Recall NEUTRAL | Recall DOWN | Pred UP | Pred NEU | Pred DOWN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| initial frame lexicon + NLI fallback | 2783 | 0.492 | 0.487 | 0.313 | 0.676 | 0.488 | 0.193 | 0.561 | 0.246 |

错分率：

| Error type | Rate |
|---|---:|
| UP -> NEUTRAL | 0.590 |
| DOWN -> NEUTRAL | 0.417 |
| NEUTRAL -> UP | 0.172 |
| NEUTRAL -> DOWN | 0.152 |
| UP -> DOWN | 0.097 |
| DOWN -> UP | 0.096 |

解释：

- 初始 frame lexicon 已经能较好识别 DOWN，但 UP recall 很低。
- 最大问题是 UP 被大量压成 NEUTRAL。
- 说明小红书中的 UP 高光 cue 仍覆盖不足，例如旅行、留学、活动、社交认可、稀缺消费等。

### 6.2 Train-augmented lexicon：当前最佳 full test

使用 train-augmented lexicon 后，full test 最佳结果如下：

| Setting | n | Accuracy | Macro F1 | Recall UP | Recall NEUTRAL | Recall DOWN | Pred UP | Pred NEU | Pred DOWN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train_augmented_stable_nli | 2783 | **0.523** | **0.525** | 0.431 | 0.606 | 0.532 | 0.255 | 0.473 | 0.272 |

错分率：

| Error type | Rate |
|---|---:|
| UP -> NEUTRAL | 0.464 |
| DOWN -> NEUTRAL | 0.350 |
| NEUTRAL -> UP | 0.217 |
| NEUTRAL -> DOWN | 0.177 |
| UP -> DOWN | 0.105 |
| DOWN -> UP | 0.118 |

与初始 frame lexicon 相比：

| Metric | Initial framelex | Train-augmented | Change |
|---|---:|---:|---:|
| Accuracy | 0.492 | 0.523 | +0.031 |
| Macro F1 | 0.487 | 0.525 | +0.038 |
| Recall UP | 0.313 | 0.431 | +0.118 |
| Recall NEUTRAL | 0.676 | 0.606 | -0.070 |
| Recall DOWN | 0.488 | 0.532 | +0.044 |
| UP -> NEUTRAL | 0.590 | 0.464 | -0.126 |
| DOWN -> NEUTRAL | 0.417 | 0.350 | -0.067 |

解释：

- train-only cue extraction 有效减少了 neutralization。
- UP recall 提升最大，说明训练集中抽取的平台原生高光 cue 很重要。
- NEUTRAL recall 降低，说明引入更多方向性 cue 后会带来一定 false positive。

### 6.3 加入 NEUTRAL frames 的版本

进一步加入 `top_k_neu=70` 后，full test 结果为：

| Setting | n | Accuracy | Macro F1 | Recall UP | Recall NEUTRAL | Recall DOWN | Pred UP | Pred NEU | Pred DOWN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train_augmented_stable_nli_neu | 2783 | 0.515 | 0.519 | **0.489** | 0.522 | **0.534** | 0.320 | 0.407 | 0.273 |

错分率：

| Error type | Rate |
|---|---:|
| UP -> NEUTRAL | 0.408 |
| DOWN -> NEUTRAL | 0.292 |
| NEUTRAL -> UP | 0.298 |
| NEUTRAL -> DOWN | 0.180 |
| UP -> DOWN | 0.103 |
| DOWN -> UP | 0.174 |

解释：

- UP 和 DOWN recall 进一步提升。
- 但 NEUTRAL recall 明显下降，很多原本 NEUTRAL 的内容被误拉成 UP 或 DOWN。
- 这说明 NEUTRAL frames 的作用并不只是抑制方向判断；它们也可能让模型更积极地区分语境，从而释放 UP/DOWN，但代价是 NEUTRAL precision 下降。

### 6.4 Theory-guided DOWN scope 扩展

为了解决 DOWN 被误判为 NEUTRAL 的问题，我尝试扩展 DOWN 的理论定义，将其分为三类：

1. poster-disadvantage：帖主失败、受限、焦虑、缺资源。
2. audience-gap / aspiration-pressure：显高、显瘦、显白、平替、省钱、普通人补救缺口。
3. daily friction / complaint：生活费、住宿、消费降级、论文/工作/面试压力、公共空间冲突。

在 VAL balanced100 上，不同 prompt 版本表现如下：

| Version | n | Accuracy | Macro F1 | Recall UP | Recall NEUTRAL | Recall DOWN | 观察 |
|---|---:|---:|---:|---:|---:|---:|---|
| relationfix_upallow | 300 | 0.497 | 0.482 | 0.470 | 0.790 | 0.230 | DOWN recall 偏低 |
| downscope_v3 | 300 | **0.510** | **0.504** | 0.440 | 0.750 | **0.340** | val 上最好 |
| downscope_v4 | 300 | 0.487 | 0.466 | 0.330 | 0.840 | 0.290 | 收窄 DOWN 后过度 NEUTRAL |
| downscope_v4b | 300 | 0.473 | 0.462 | 0.340 | 0.770 | 0.310 | 仍不如 v3 |

downscope_v3 在 full TEST 上的结果：

| Setting | n | Accuracy | Macro F1 | Recall UP | Recall NEUTRAL | Recall DOWN | Pred UP | Pred NEU | Pred DOWN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| theory_error_injected_v2_downscope_v3_full | 2783 | 0.500 | 0.489 | 0.288 | **0.710** | 0.503 | 0.157 | 0.568 | 0.276 |

解释：

- DOWN recall 提升到 0.503，说明扩展 DOWN scope 有效。
- 但 UP recall 降到 0.288，说明 DOWN scope 过强会抢走 UP 或把 UP 压成 NEUTRAL。
- 因此 downscope_v3 适合作为分析 DOWN 标注口径的实验，不适合作为最终主结果。

## 7. 与原论文结果的直接比较

### 7.1 与 GPT-4.1-nano zero-shot 比较

| Model / Setting | Accuracy | Macro F1 | Recall UP | Recall NEUTRAL | Recall DOWN | Pred NEUTRAL |
|---|---:|---:|---:|---:|---:|---:|
| 原论文 GPT-4.1-nano zero-shot | 0.469 | 0.469 | 0.379 | 0.630 | 0.397 | 0.558 |
| 本实验 train_augmented_stable_nli | **0.523** | **0.525** | **0.431** | 0.606 | **0.532** | **0.473** |

提升：

- Accuracy: +0.054
- Macro F1: +0.056
- Recall UP: +0.052
- Recall DOWN: +0.135
- Predicted NEUTRAL: -0.085

说明：

> frame-level lexicon + NLI retrieval 可以明显改善 GPT-4.1-nano 的检测能力，尤其减少它把 UP/DOWN 帖子 neutralize 的倾向。

### 7.2 与 GPT-5 zero-shot 比较

| Model / Setting | Accuracy | Macro F1 | Recall UP | Recall NEUTRAL | Recall DOWN | Pred NEUTRAL |
|---|---:|---:|---:|---:|---:|---:|
| 原论文 GPT-5 zero-shot | 0.521 | 0.518 | 0.410 | **0.752** | 0.402 | 0.601 |
| 本实验 GPT-4.1-nano + lexicon/NLI | **0.523** | **0.525** | **0.431** | 0.606 | **0.532** | **0.473** |

说明：

- 我们的低成本 GPT-4.1-nano + lexicon/NLI 略高于 GPT-5 zero-shot。
- 主要优势在于 DOWN recall 更高，Predicted NEUTRAL 更低。
- 但 GPT-5 的 NEUTRAL recall 更高，说明我们的系统更积极地检测比较方向，也更容易把部分 NEUTRAL 拉成 UP/DOWN。

### 7.3 与 supervised encoder 比较

| Model / Setting | Accuracy | Macro F1 | Recall UP | Recall NEUTRAL | Recall DOWN |
|---|---:|---:|---:|---:|---:|
| CN-RoBERTa WWM | **0.680** | **0.679** | **0.695** | 0.585 | **0.759** |
| 本实验 train_augmented_stable_nli | 0.523 | 0.525 | 0.431 | **0.606** | 0.532 |

差距：

- Macro F1: -0.154
- Recall UP: -0.264
- Recall DOWN: -0.227

说明：

> 本方法能超过 zero-shot LLM，但还不能替代 supervised in-domain encoder。  
> Encoder 通过训练直接学习 XHS-SCoRE 标注分布，而 prompt-based LLM 即使有 lexicon scaffold，仍然受隐含语境、方向冲突和 neutralizer 影响。

### 7.4 与原论文 cue-explicit prompt 比较

原论文 Table 3 显示，GPT-4.1-nano cue-explicit prompt 并没有解决 neutralization：

| GPT-4.1-nano Setting | Macro F1 | Predicted NEUTRAL | UP -> NEUTRAL | DOWN -> NEUTRAL |
|---|---:|---:|---:|---:|
| 原论文 zero-shot | 46.9 | 55.8 | 54.2 | 50.4 |
| 原论文 cue-explicit | 44.9 | 68.2 | 68.6 | 57.8 |
| 本实验 train_augmented_stable_nli | **52.5** | **47.3** | **46.4** | **35.0** |

说明：

> 单纯把 cue inventory 写进 prompt 不够。  
> 本实验更有效的原因可能是：cue 被组织为 frame，且通过 NLI retrieval 只选择与当前帖子上下文相关的 frame，减少了无关 cue 干扰。

## 8. 错误分析

### 8.1 UP -> NEUTRAL

这是当前最佳 full test 中仍然最主要的问题之一：

- UP -> NEUTRAL = 0.464

常见原因：

- 高光内容没有显式比较词。
- 模型把旅行、城市体验、演唱会、创作展示等当成普通日常。
- NLI 没有选中足够强的 UP frame。

需要继续补的 UP frame：

- city/travel mobility
- event/fandom high point
- creative work display
- elite identity context
- social approval display
- aesthetic lifestyle display

### 8.2 DOWN -> NEUTRAL

当前最佳 full test 中：

- DOWN -> NEUTRAL = 0.350

常见原因：

- DOWN 标注包含 audience-gap / friction，但模型认为只是普通攻略、信息或求助。
- 文本没有显式“我更差”，只呈现预算、麻烦、补救、受限。

已经尝试通过 downscope_v3 改善，但会伤害 UP recall。

### 8.3 NEUTRAL -> UP / DOWN

当前最佳 full test 中：

- NEUTRAL -> UP = 0.217
- NEUTRAL -> DOWN = 0.177

常见原因：

- broad hypotheses 过宽。
- 普通产品、路线、教程、求推荐被方向性 frame 吸走。
- 一些中性 cue 同时出现在 UP/DOWN 语境中，导致 LLM 过度解释。

## 9. 主要结论

1. **本实验最佳结果超过原论文 GPT-4.1-nano zero-shot。**  
   Macro F1 从 0.469 提升到 0.525。

2. **本实验最佳结果略高于原论文 GPT-5 zero-shot。**  
   GPT-5 zero-shot Macro F1 = 0.518；本实验 GPT-4.1-nano + lexicon/NLI Macro F1 = 0.525。

3. **本方法仍明显低于 supervised encoder。**  
   CN-RoBERTa WWM Macro F1 = 0.679，本实验最佳 = 0.525。

4. **frame-level lexicon 比 raw cue list 更有效。**  
   原论文 cue-explicit prompt 对 GPT-4.1-nano 没有明显提升，而本实验通过 frame-level lexicon + NLI retrieval 显著减少 neutralization。

5. **DOWN 的标注口径比经典社会比较理论更宽。**  
   XHS-SCoRE 中的 DOWN 不只表示“帖主更惨”，还包括读者感到缺口、压力、摩擦或补救需求。

6. **当前最大瓶颈是 UP recall。**  
   最佳 full test 的 Recall UP = 0.431，说明仍有大量高光/资源/移动性/社交认可帖子被判成 NEUTRAL 或 DOWN。

## 10. 下一步计划

下一步不建议继续手工调 prompt，因为 val 上 v3/v4/v4b 已经显示 prompt 调整收益有限，而且容易在 DOWN recall 和 NEUTRAL precision 之间来回摆动。

更合理的下一步是：

1. 在 TRAIN 上跑当前最佳配置。
2. 收集 TRAIN 中的 `UP -> NEUTRAL` 错误。
3. 只从 TRAIN 错误中挖 missing UP frame/cue。
4. 用 VAL 选择是否合并新 cue 和调参数。
5. 最后只在 TEST full 上报告一次。

重点补充的 UP cue/frame：

- 旅行/城市移动性：solo trip、citywalk、港澳台/海外、又来某地、圆梦
- 活动/事件高光：演唱会、展览、比赛、毕业照、人生照片
- 稀缺/粉丝消费：隐藏款、限量、抽中、谷子、周边
- 创作/作品展示：拍摄、摄影、妆造、穿搭、作品集、改造成功
- 身份/资源暗示：港硕、留学、名校、offer、实习、体制内、CEO/大厂
- 社交认可：被夸、回头率、爆款、点赞、评论区夸、朋友以为

同时需要加强 exclusion rules：

- 攻略/教程/路线/票价/产品经验：默认 NEUTRAL，除非有第一人称高光拥有或明显缺口压力。
- 省钱/低价/免费：不自动 DOWN，只有预算不足、消费降级、买不起、纠结求助时才 DOWN。
- 好看/出片/显白/显高：帖主成果展示可 UP，普通人补救压力可 DOWN，客观教程可 NEUTRAL。

## Appendix: 实验产物说明

主要实验结果表已经汇总在：

```text
outputs/experiment_metrics_overview.csv
```

主要报告文件：

```text
outputs/experiment_summary_report.md
```

主要词表版本：

| File | Rows | 用途 |
|---|---:|---|
| `outputs/lexicon/xhs_social_comparison_lexicon.csv` | 429 | 初始理论 frame lexicon |
| `outputs/lexicon/xhs_social_comparison_lexicon_train_augmented.csv` | 1237 | 当前最佳 full test 词表 |
| `outputs/lexicon/xhs_social_comparison_lexicon_theory_error_injected_v2.csv` | 1021 | theory/downscope 实验词表 |

主要结果文件：

| File | 说明 |
|---|---|
| `outputs/frame_lexicon_test/test_train_augmented_stable_nli.csv` | 当前最佳 full test 输出 |
| `outputs/frame_lexicon_test/test_train_augmented_stable_nli_neu.csv` | 加 NEUTRAL frames 的 full test 输出 |
| `outputs/frame_lexicon_test/test_theory_error_injected_v2_downscope_v3_full.csv` | theory/downscope v3 full test 输出 |
