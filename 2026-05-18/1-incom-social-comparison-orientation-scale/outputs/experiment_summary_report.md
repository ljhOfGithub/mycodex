# XHS-SCoRE Social Comparison Lexicon Experiments Summary

## 1. 研究目标

本轮实验的目标是构建一个用于 XHS-SCoRE 的 frame-level social comparison lexicon，并把它接入 LLM prompt，以缓解 generation-detection dissociation：LLM 能生成具有社会比较压力的内容，但在检测时容易把隐含比较判断为 NEUTRAL，或者混淆 UPWARD / DOWNWARD 方向。

任务标签的工作定义如下：

- **UPWARD**：文本呈现令人向往的优势/高光/资源，读者感觉“对方/帖主/文本对象比我好”。
- **DOWNWARD**：文本呈现缺口、压力、失败、受限、补救或低位状态，读者感觉“对方处境更差”或“普通人存在缺口/压力/限制”。
- **NEUTRAL**：没有明显相对地位、比较邀请或社会比较压力，主要是信息、教程、产品、第三方说明或普通日常。

一个重要发现是：经典社会比较理论中的 DOWNWARD 更偏“对方比我差”，但 XHS-SCoRE 数据中的 DOWNWARD 标注更宽，还包含 audience-gap / aspiration-pressure / daily friction 等内容。

## 2. 方法路线

### 2.1 初始理论词表

初始 lexicon 不是简单 n-gram，而是按社会比较理论组织成 frame：

- INCOM：self-other relation、relative standing、ability / opinion comparison
- UPACS / DACS：appearance upward / downward comparison
- LIWC：情绪、成就、金钱、工作、家庭、身体等辅助维度
- Wmatrix / USAS：semantic domain scaffold，例如 achievement、education、money、travel、appearance、relationship、family conflict、low agency、tutorial/product/news

初始词表文件：

- `outputs/lexicon/xhs_social_comparison_lexicon.csv`
- 行数：429
- 标签分布：UP 199, DOWN 114, NEUTRAL 70, AMBIGUOUS 46

### 2.2 LLM train-only cue extraction

为了补充小红书原生表达，我使用 train split 做 LLM-assisted frame/cue extraction。该步骤不使用 val/test 构建词表。

重点补充的 XHS cue 包括：

- UP：solo trip、圆梦、演唱会、青旅、夜生活、留学 vlog、隐藏款、拆盲盒、谷子、港硕、offer 等
- DOWN：容貌焦虑、预算有限、找不到工作、没上岸、被拒、社死、压力大、求助、怎么办等
- NEUTRAL：攻略、教程、产品测评、第三方信息、求推荐等 neutralizer

主要输出：

- `outputs/lexicon/xhs_social_comparison_lexicon_train_augmented.csv`
- 行数：1237
- 标签分布：UP 449, DOWN 360, NEUTRAL 382, AMBIGUOUS 46

### 2.3 Supervised mining / contrastive filtering

在 train 上尝试了多种监督式词表筛选：

- log odds ratio with informative Dirichlet prior
- chi-square / mutual information
- L1 logistic feature selection
- hard-negative mining
- contrastive cue purity score
- seed expansion with train-label purity filtering

这些方法能发现高纯度 cue，但直接扩大词表容易引入噪声。尤其是 singletons 或低纯度 cue 合并后，会造成 prompt 过长和方向混淆。

相关输出：

- `outputs/lexicon/train_supervised_mining/train_supervised_ngram_features.csv`
- `outputs/lexicon/train_supervised_mining/train_cv_hard_negatives.csv`
- `outputs/lexicon/train_supervised_mining/train_negation_scope_patterns.csv`

### 2.4 NLI context-aware retrieval

参考 CACLP 的 context-aware lexical retrieval 思路，加入 NLI-style frame retrieval：

1. 先用 lexicon cue 命中候选 frame。
2. 再让 NLI prompt 判断 frame 是否被当前上下文支持。
3. final classifier 只看到 context-relevant frame，而不是整张词表。
4. 用 consistency rules 修正 `comparison_relation=absent` 与 UP/DOWN 标签冲突的情况。

实验中发现 NLI 很关键，但过宽的 broad hypotheses 会造成普通正面体验被拉成 UP，或普通求助/教程被拉成 DOWN。

## 3. 主要实验结果

### 3.1 全量 test 结果

| Run | Lexicon / Setting | n | Acc | Macro F1 | R_UP | R_NEU | R_DOWN | 主要现象 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| baseline framelex NLI fallback | initial frame lexicon + NLI | 2783 | 0.492 | 0.487 | 0.313 | 0.676 | 0.488 | DOWN 较好，但 UP 被大量压成 NEUTRAL |
| train_augmented_stable_nli | train-augmented lexicon, no NEU prompt block | 2783 | **0.523** | **0.525** | 0.431 | 0.606 | 0.532 | 当前全量 test 最佳 macro F1，三类较均衡 |
| train_augmented_stable_nli_neu | train-augmented + NEUTRAL frames | 2783 | 0.515 | 0.519 | **0.489** | 0.522 | **0.534** | UP/DOWN recall 更高，但 NEUTRAL 被误拉到 UP/DOWN |
| theory_error_injected_v2_downscope_v3_full | theory + train-error DOWN injection + downscope prompt | 2783 | 0.500 | 0.489 | 0.288 | **0.710** | 0.503 | DOWN 提升明显，但 UP recall 崩得较多 |

结论：  
**当前全量 test 最佳是 `xhs_social_comparison_lexicon_train_augmented.csv` + stable NLI 设置，macro F1 = 0.525。**  
theory/downscope 系列对 DOWN 有帮助，但代价是 UP 被过度压制，因此不适合作为当前最终版本。

### 3.2 Balanced test / val 调参结果

| Run | Split | n | Acc | Macro F1 | R_UP | R_NEU | R_DOWN | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| theory_error_v2_relationfix_upallow | test balanced100 | 300 | **0.530** | **0.506** | 0.480 | 0.850 | 0.260 | 小样本上效果最好，但 DOWN recall 仍偏低 |
| theory_error_v2_downscope_v3 | val balanced100 | 300 | 0.510 | 0.504 | 0.440 | 0.750 | **0.340** | val 上最佳 prompt tuning 版本 |
| theory_error_v2_downscope_v4 | val balanced100 | 300 | 0.487 | 0.466 | 0.330 | 0.840 | 0.290 | 收窄 DOWN 后整体变差，过度 NEUTRAL |
| theory_error_v2_downscope_v4b | val balanced100 | 300 | 0.473 | 0.462 | 0.340 | 0.770 | 0.310 | 保留 DOWN override 但 prompt 收窄仍伤 recall |

结论：  
val 上 v3 是 prompt-tuning 中最好的版本，但迁移到 full test 后 UP recall 明显下降，说明 prompt 扩展 DOWN scope 不能替代更稳的 train-only cue mining。

### 3.3 Balanced300 异常结果

`test_train_augmented_stable_nli_neu_balanced300`：

- n = 900
- Acc = 0.476
- Macro F1 = 0.432
- R_UP = 0.310
- R_NEU = 0.923
- R_DOWN = 0.193

这个结果明显不同于 full test 的同配置结果。原因可能是 `balanced_limit_per_class` 取的是每类前 N 条，不是随机采样；前 300 条可能存在分布偏移或顺序效应。因此 balanced subset 只能用于快速 debug，不应作为最终结论。

## 4. 关键 ablation 观察

### 4.1 `comparison_relation=absent` consistency rule 很敏感

早期规则过严时：

- UP/DOWN 如果 relation=absent，会被压成 NEUTRAL。
- 结果：NEUTRAL recall 高，但 UP/DOWN recall 很低。

后来放宽为：

- 有同方向 frame evidence 时，可以把 relation 修正为 implicit。
- 强 UP dominant frame 可以放行。
- 强 DOWN dominant frame 可以放行。

效果：

- `DOWN->UP` 明显下降。
- 但若 UP 放行太窄，会导致 `UP->NEUTRAL` 激增。
- 若 DOWN 放行太宽，会导致 `NEUTRAL->DOWN` 和 `UP->DOWN` 上升。

### 4.2 DOWN 的定义不能只按经典“对方比我差”

实验中发现，XHS-SCoRE 的 DOWN 包含三类：

1. **poster-disadvantage**：帖主失败、焦虑、受限、缺资源。
2. **audience-gap / aspiration-pressure**：显高、显瘦、显白、低成本、平替、省钱、普通人补救缺口。
3. **daily friction / complaint**：生活费、住宿、消费降级、设备损坏、论文/面试/工作、公共空间冲突、踩雷求助。

加入这些 frame 后，val DOWN recall 从 0.23 提升到 0.34；full test DOWN recall 达到 0.503。  
但代价是 UP recall 降低，说明 DOWN scope 需要靠更精确 cue 和 exclusion rule 控制。

### 4.3 加 NEUTRAL frames 有双刃剑效果

`top_k_neu=70` 的 full test 版本：

- UP recall 从 0.431 提升到 0.489
- DOWN recall 从 0.532 提升到 0.534
- NEUTRAL recall 从 0.606 降到 0.522

解释：NEUTRAL frames 并不只是“压制方向判断”，它们也给模型更多语境，可能释放一些 UP/DOWN 判断；但同时会让 NEUTRAL precision 下降。

### 4.4 继续扩大词表并不一定提升结果

例如 `xhs_social_comparison_lexicon_train_gap_augmented_v1.csv` 行数达到 7716，但 balanced test macro F1 只有 0.478。说明 prompt lexicon 不是越大越好，过多低纯度 cue 会稀释 frame 信号。

## 5. 当前最佳配置

### 推荐用于汇报的主结果

**Full test 最佳：**

- Lexicon：`outputs/lexicon/xhs_social_comparison_lexicon_train_augmented.csv`
- Output：`outputs/frame_lexicon_test/test_train_augmented_stable_nli.csv`
- n = 2783
- Accuracy = 0.523
- Macro F1 = 0.525
- Recall UP = 0.431
- Recall NEUTRAL = 0.606
- Recall DOWN = 0.532

这个版本三类 recall 相对均衡，是目前最适合作为主实验结果的版本。

### 备选结果

**更强调 UP/DOWN 检出：**

- Output：`outputs/frame_lexicon_test/test_train_augmented_stable_nli_neu.csv`
- Accuracy = 0.515
- Macro F1 = 0.519
- Recall UP = 0.489
- Recall NEUTRAL = 0.522
- Recall DOWN = 0.534

这个版本适合说明：加入 NEUTRAL frames 后，UP/DOWN recall 更高，但 NEUTRAL 类更容易被误判为方向性比较。

## 6. 错误类型总结

### 6.1 UP -> NEUTRAL

这是 full test 中仍然很严重的问题。

常见原因：

- 高光内容没有显式比较词，被模型判断为 `comparison_relation=absent`。
- 普通正向体验与社会比较高光之间边界模糊。
- NLI 没有检索到足够强的 UP frame。

典型需要补的 frame：

- city/travel mobility
- event/fandom high point
- creative work display
- elite identity context
- social approval display
- aesthetic lifestyle display

### 6.2 DOWN -> NEUTRAL

常见原因：

- DOWN 标注口径包含 audience-gap / daily friction，但 prompt 容易把它当作普通攻略/信息/日常。
- 文本没有直接说“我更差”，只是呈现限制、麻烦、预算、补救、求助。

已尝试通过 downscope v3 补充，但 full test 中会伤害 UP recall。

### 6.3 NEUTRAL -> UP / DOWN

常见原因：

- broad hypotheses 太宽。
- 普通产品、路线、教程、求推荐被 direction frame 吸走。
- 单个 DOWN cue 触发 override，导致 NEUTRAL 被拉成 DOWN。

## 7. 下一步建议

### 7.1 不建议继续手工调 prompt

val 上 v3、v4、v4b 已经显示：prompt 调整会在 DOWN recall 与 NEUTRAL precision 之间来回拉扯，收益有限。

### 7.2 推荐下一步：train-only UP gap mining

当前 full test 最大短板是 UP recall：

- `test_train_augmented_stable_nli`：UP recall = 0.431
- `theory_error_injected_v2_downscope_v3_full`：UP recall = 0.288

因此下一步应该优先从 train 的 `UP->NEUTRAL` 错误中挖 missing UP frame/cue，而不是继续补 DOWN。

重点挖：

- 旅行/城市移动性：solo trip、citywalk、港澳台/海外、又来某地、圆梦
- 活动/事件高光：演唱会、展览、比赛、毕业照、人生照片
- 稀缺/粉丝消费：隐藏款、限量、抽中、谷子、周边
- 创作/作品展示：拍摄、摄影、妆造、穿搭、作品集、改造成功
- 身份/资源暗示：港硕、留学、名校、offer、实习、体制内、CEO/大厂
- 社交认可：被夸、回头率、爆款、点赞、评论区夸、朋友以为

### 7.3 加强 exclusion rules

对容易误判的 cue 建议加入 context rule：

- `攻略/教程/路线/票价/产品经验`：默认 NEUTRAL，除非出现第一人称高光拥有或明显缺口压力。
- `省钱/低价/免费`：不自动 DOWN；若是“预算不足/消费降级/买不起/纠结求助”才 DOWN。
- `好看/出片/显白/显高`：若是帖主成果展示可 UP；若是普通人补救压力可 DOWN；若是客观教程可 NEUTRAL。

### 7.4 使用 val 做选择，test 做最终报告

建议后续流程：

1. 在 train 上跑当前最佳配置，收集 `UP->NEUTRAL` 和 `DOWN->NEUTRAL`。
2. 只从 train 错误中抽 cue。
3. 用 val balanced/full 选择 prompt 和参数。
4. 最后只在 test full 上报告一次。

## 8. 简短结论

本轮实验表明，frame-level social comparison lexicon + NLI context-aware retrieval 能比初始 frame lexicon 更好地捕捉 XHS-SCoRE 中的隐含社会比较方向。当前最佳 full test 结果为 macro F1 = 0.525。主要瓶颈已经从 DOWN recall 转移到 UP recall，说明下一步应从 train 的 UP->NEUTRAL 错误中继续补充小红书原生高光/资源/移动性/社交认可 cue，同时控制教程、产品、路线、求推荐等 neutralizer，避免方向性误判。

## Appendix: 指标总表

完整自动汇总表已保存为：

- `outputs/experiment_metrics_overview.csv`

该表包含所有已保存 run 的 accuracy、macro F1、三类 recall、预测类别比例和主要错分率。
