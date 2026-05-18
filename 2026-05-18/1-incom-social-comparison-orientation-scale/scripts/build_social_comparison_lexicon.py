from pathlib import Path
import csv
import json
import re

import pandas as pd


ROOT = Path("/Users/jackie/Documents/Codex/2026-05-18/1-incom-social-comparison-orientation-scale")
DOWNLOADS = Path("/Users/jackie/Downloads/Yu EMNLP")
OUT = ROOT / "outputs" / "lexicon"
OUT.mkdir(parents=True, exist_ok=True)


FILES = {
    "DOWN": DOWNLOADS / "DC_full.xlsx",
    "NEUTRAL": DOWNLOADS / "NC_full.xlsx",
    "UP": DOWNLOADS / "UC_full.xlsx",
}


def load_texts():
    data = {}
    for label, path in FILES.items():
        df = pd.read_excel(path)
        data[label] = df["content"].fillna("").astype(str).tolist()
    return data


DATA = load_texts()


def count_cue(cue: str, texts):
    if cue.startswith("re:"):
        pat = re.compile(cue[3:])
        return sum(1 for t in texts if pat.search(t))
    return sum(1 for t in texts if cue in t)


def add(rows, label, frame, cues, cue_type, source, rationale, weight=2, rule=None):
    for cue in cues:
        counts = {k: count_cue(cue, v) for k, v in DATA.items()}
        total = sum(counts.values())
        if total:
            empirical_label = max(counts, key=counts.get)
            empirical_share = round(counts[empirical_label] / total, 3)
        else:
            empirical_label = ""
            empirical_share = ""
        rows.append({
            "target_label": label,
            "frame": frame,
            "cue": cue,
            "cue_type": cue_type,
            "source_basis": source,
            "rationale": rationale,
            "weight_1_3": weight,
            "context_rule": rule or "",
            "count_UP": counts["UP"],
            "count_DOWN": counts["DOWN"],
            "count_NEUTRAL": counts["NEUTRAL"],
            "empirical_label": empirical_label,
            "empirical_share": empirical_share,
        })


rows = []

# Relation markers from INCOM: self-other relation, ability/opinion comparison,
# similarity/difference, relative standing. These are direction-sensitive and
# should not decide UP/DOWN alone.
add(rows, "AMBIGUOUS", "comparison_marker:self_other_relation", [
    "比我", "比自己", "比别人", "和别人比", "跟别人比", "和同龄人比", "同龄人",
    "身边的人", "身边朋友", "别人都", "大家都", "人家都", "我却", "我还在",
    "只有我", "就我", "凭什么", "差距", "距离感", "落差", "天花板", "卷不过",
    "羡慕", "嫉妒", "破防", "酸了", "emo了", "不如", "赶不上", "配不上",
], "relational_marker", "INCOM+XHS", "显式或半显式把 self 与 other 放进同一比较关系，是 xhs-score 里最容易被 LLM 漏掉的关系性线索。", 3,
    "先识别比较关系，再结合 poster advantaged/disadvantaged frame 判定方向。")

add(rows, "AMBIGUOUS", "comparison_marker:standing_rank", [
    "排名", "榜单", "top", "第一", "前十", "天花板", "段位", "级别", "层次",
    "同龄", "同届", "同班", "同事", "peer", "同龄人已经", "同龄人都",
], "relative_standing", "INCOM+USAS", "相对位置、排序和同侪参照会把普通事实变成可比较事实。", 2,
    "若只是第三方榜单/公告且无 reader-poster positioning，降为 NEUTRAL。")

# UP frames: poster is better off, successful, advantaged, more resourced.
add(rows, "UP", "achievement_elite_education", [
    "上岸", "成功上岸", "保研", "拟录取", "录取", "offer", "港大offer", "港硕",
    "牛津", "剑桥", "哈佛", "清华", "北大", "复旦", "上交", "985", "211",
    "博士毕业", "硕士毕业", "奖学金", "全奖", "绩点", "GPA", "第一名",
    "逆袭", "高分", "考过了", "拿到了", "毕业快乐",
], "domain_frame", "XHS+USAS achievement", "成就/教育资源把帖主放在更高相对位置，常触发读者向上比较。", 3,
    "若文本是教程、经验分享、申请攻略，需看是否仍突出帖主优势；纯步骤说明可降权。")

add(rows, "UP", "career_income_success", [
    "年薪", "月薪", "百万", "存款", "攒了", "副业收入", "赚了一千", "工资翻倍",
    "升职", "大厂", "实习offer", "入职", "自由职业", "被猎头", "创业成功",
    "接到offer", "转正", "财富自由", "搞钱", "搞到钱", "收入", "奖金",
], "domain_frame", "LIWC money/work+XHS", "金钱、工作和成就类资源直接构成 relative standing。", 3,
    "金额/职位若与第一人称收获绑定，偏 UP；若是泛泛理财教程，偏 NEUTRAL。")

add(rows, "UP", "high_resource_lifestyle", [
    "定居LA", "定居", "买房", "豪宅", "大平层", "别墅", "海景房", "搬家快乐",
    "环球旅行", "欧洲旅行", "瑞士", "冰岛", "巴黎", "伦敦", "纽约", "LA",
    "Airbnb", "头等舱", "商务舱", "五星酒店", "度假", "看展", "米其林",
    "奢侈品", "香奈儿", "爱马仕", "劳力士", "人生照片", "住在喜欢的地方",
], "domain_frame", "USAS money/travel/leisure+XHS", "高资源生活方式会把日常展示变成可比较的生活水平信号。", 3,
    "旅游攻略/酒店测评不自动 UP；第一人称拥有、享受、长期居住更强。")

add(rows, "UP", "mobility_peak_experience", [
    "solo trip", "Solo trip", "一个人旅游", "又来台北", "台北", "台湾", "澳大利亚",
    "留学澳大利亚", "留学", "vlog", "演唱会", "看演唱会", "泰妍", "圆梦",
    "圆了我", "圆了梦", "毕业前", "落地", "青旅", "外国朋友", "夜生活",
    "好好感受", "非常开心", "爱上一个人旅游", "幸运的是", "中了5000",
    "大悲大喜", "第一次误机", "四天三夜", "港澳台", "海外", "旅行vlog",
], "domain_frame", "XHS mobility/leisure+Appendix B.5", "XHS-SCoRE 的 UP 不要求显式比较词；solo trip、留学、演唱会、海外/港澳台移动性和圆梦体验会构成高光/理想生活展示。", 3,
    "若是帖主自己的移动性、圆梦、旅行/演唱会体验，即使夹杂小挫折也常偏 UP；若只是攻略或第三方介绍才降为 NEUTRAL。")

add(rows, "UP", "scarce_fandom_consumption", [
    "隐藏款", "拆盲盒", "盲盒", "限量", "绝版", "谷子", "痛包", "周边",
    "抽到了", "中了", "就剩", "非人哉", "萌粒", "隐藏", "纯享拆",
    "开出", "一发入魂", "欧气", "好运", "限时", "抢到", "捡漏",
], "domain_frame", "XHS consumption/fandom+Appendix B.5", "稀缺款、隐藏款、限量周边等消费/粉丝文化高光，是小红书里常见的轻量 UP 触发，不一定表现为豪宅奢侈品。", 3,
    "若帖子强调自己拆到/抢到/拥有稀缺物，偏 UP；若只是商品介绍、交易信息或求推荐，可能 NEUTRAL。")

add(rows, "UP", "aspirational_opportunity_performance", [
    "丝芭", "广芭", "SNH48", "GNZ48", "丝芭面试", "女团", "爱豆", "团播",
    "面试", "投了小程序", "回邮件", "声乐专业", "唱歌跳舞", "路演",
    "有基础", "跳的也算", "专业", "试一下", "想试一下", "机会",
    "被选中", "入选", "出道", "舞台", "练习生",
], "domain_frame", "XHS opportunity/appearance/performance", "面试、女团/爱豆、专业能力和舞台机会把帖主放在更接近稀缺机会和可羡慕身份的位置，容易触发 UP。", 3,
    "若文本主要强调机会、专业能力、被看见或接近理想身份，偏 UP；若强调失败、被拒、焦虑崩溃，则转 DOWN。")

add(rows, "UP", "appearance_body_success", [
    "变美", "变漂亮", "变好看", "瘦了", "瘦下来", "瘦到", "马甲线", "直角肩",
    "漫画腿", "天鹅颈", "氛围感", "神仙颜值", "有效变美", "逆袭变美",
    "被夸", "回头率", "素颜也", "皮肤变好", "发量", "穿搭", "妆容",
    "女明星", "爱豆同款", "韩味", "身材管理", "上镜", "拍照好看",
], "domain_frame", "UPACS+LIWC body+XHS", "UPACS 指向更好看的他人/身体；小红书中变美、身材、穿搭成功是典型上行 cue。", 3,
    "若出现容貌焦虑/自卑/失败共现，转入 appearance_downward 或 mixed。")

add(rows, "UP", "social_approval_popularity", [
    "被夸", "被表白", "被要微信", "回头率", "夸爆", "好多赞", "爆了",
    "被老板认可", "被老师夸", "被朋友羡慕", "人缘", "脱单", "恋爱脑被治好",
], "social_reward", "LIWC social/reward+XHS", "外部认可会强化帖主优势与读者落差。", 2,
    "单独的夸奖弱于和成就/外貌/资源 frame 共现。")

# DOWN frames: poster is worse off, distressed, blocked, less resourced.
add(rows, "DOWN", "family_oppression_low_support", [
    "原生家庭", "父母控制", "父母不同意", "道德绑架", "被骂", "窒息", "家里不让",
    "重男轻女", "催婚", "催生", "不被理解", "断亲", "家暴", "冷暴力",
    "控制欲", "逃离家庭", "妈妈骂", "爸爸骂",
], "domain_frame", "LIWC family+XHS", "家庭压迫/低支持把帖主放在受限和低资源位置，容易触发下行比较或同情。", 3,
    "若只是家庭日常记录且无受害/压迫，不自动 DOWN。")

add(rows, "DOWN", "blocked_aspiration_failure", [
    "失败", "没上岸", "落榜", "被拒", "拒信", "没录取", "挂科", "延毕",
    "找不到工作", "失业", "裸辞后悔", "面试失败", "考砸", "没考上",
    "又失败了", "白努力", "崩了", "废了", "没希望", "焦虑到睡不着",
], "domain_frame", "USAS achievement/opposite+XHS", "目标受阻是 UP 成就 frame 的反向集合，符合 CACLP 的 contrastive lexicon 思路。", 3,
    "如果失败后马上给出教程/复盘且不定位帖主困境，可降权。")

add(rows, "DOWN", "low_agency_constraint", [
    "没办法", "只能", "被迫", "没有选择", "不得不", "不敢", "不配", "撑不住",
    "熬不住", "走不出来", "困住", "逃不掉", "摆烂", "躺平", "无力",
    "麻木", "崩溃", "救命", "求助", "怎么办",
], "relational_state", "INCOM self-position+XHS", "低能动性不是情绪词本身，而是把 self/poster 放在受限位置的关系线索。", 3,
    "与教程问答共现时需区分真实困境和普通求推荐。")

add(rows, "DOWN", "appearance_body_distress", [
    "容貌焦虑", "身材焦虑", "胖", "丑", "不敢拍照", "不上镜", "痘痘", "爆痘",
    "脱发", "秃", "自卑", "反弹", "暴食", "水肿", "脸垮", "法令纹",
    "皮肤差", "减肥失败", "越减越胖", "体重焦虑", "腿粗", "小肚子",
], "domain_frame", "DACS/UPACS+LIWC body+XHS", "外貌/身体痛苦是小红书中最强的下行或脆弱性 cue 之一。", 3,
    "若同时出现变美成果，对比前后可能整体为 UP；需看当前帖主定位是成功还是困境。")

add(rows, "DOWN", "money_work_hardship", [
    "没钱", "穷", "负债", "还不起", "房租", "被裁", "裁员", "加班到崩溃",
    "工资低", "欠款", "省钱到", "吃土", "打工人崩溃", "被辞退", "社畜",
    "生活费不够", "月光", "破产",
], "domain_frame", "LIWC money/work+risk+XHS", "经济/工作困境是高资源生活方式的反向集合。", 3,
    "省钱攻略和薅羊毛通常是 NEUTRAL，除非强调匮乏和无助。")

add(rows, "DOWN", "negative_affect_social_pain", [
    "焦虑", "抑郁", "emo", "破防", "崩溃", "委屈", "孤独", "失眠", "内耗",
    "被孤立", "被背刺", "分手", "失恋", "冷暴力", "不被爱", "没人懂",
], "affect_supporting", "LIWC negative emotion+social+XHS", "负面情绪只能辅助方向；当它与低资源/受阻/受害 frame 结合时才支持 DOWN。", 2,
    "不要把所有负面情绪直接判 DOWN；读者比较方向取决于帖主相对位置。")

# NEUTRAL frames and neutralizers.
add(rows, "NEUTRAL", "tutorial_information", [
    "教程", "攻略", "步骤", "整理", "合集", "干货", "保姆级", "清单", "模板",
    "方法", "指南", "避坑", "测评", "参数", "官网", "下载", "入口",
    "怎么做", "如何", "笔记", "资料", "经验贴",
], "neutralizer_frame", "Wmatrix/USAS information+XHS", "信息/教程语体通常缺少 reader-poster standing，能防止 LLM 把领域词误判为比较。", 3,
    "若教程开头强展示个人优势，如'我拿到港大 offer 后...'，不要完全中和。")

add(rows, "NEUTRAL", "product_tool_review", [
    "测评", "开箱", "参数", "链接", "官网", "下载", "AI工具", "插件",
    "软件", "app", "使用感", "平替", "好物分享", "购物车", "优惠券",
    "薅羊毛", "推荐", "种草", "拔草",
], "neutralizer_frame", "LIWC reward/work+USAS object", "产品/工具评价常有正负情绪但不一定比较 self-other standing。", 3,
    "奢侈品拥有展示可能转 UP；普通测评保持 NEUTRAL。")

add(rows, "NEUTRAL", "third_party_news_ranking", [
    "新闻", "公告", "名单", "排行榜", "排名公布", "录取名单", "政策", "通知",
    "官方", "数据", "报告", "趋势", "盘点", "明星", "电视剧", "综艺",
], "neutralizer_frame", "Wmatrix/USAS news/ranking", "第三方信息不必然触发读者和帖主比较，尤其没有第一人称获益/受损时。", 2,
    "如果帖子把榜单与'我/我的学校/我的收入'绑定，重新判定。")

add(rows, "NEUTRAL", "request_recommendation_chat", [
    "求推荐", "求问", "问问", "有没有", "姐妹们", "哪个好", "怎么选",
    "可以吗", "适合吗", "想买", "有人知道吗", "蹲", "码住",
], "neutralizer_frame", "XHS conversational", "求助/询问语体常是平台互动，不等于下行；只有真实受限/受苦才 DOWN。", 2,
    "与'不敢/焦虑/没办法'强共现时提高 DOWN 风险。")


rows.sort(key=lambda r: (r["target_label"], r["frame"], r["cue"]))

csv_path = OUT / "xhs_social_comparison_lexicon.csv"
with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

json_path = OUT / "xhs_social_comparison_lexicon.json"
json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

summary = {}
for row in rows:
    summary.setdefault(row["target_label"], {}).setdefault(row["frame"], 0)
    summary[row["target_label"]][row["frame"]] += 1

prompt_lines = [
    "# XHS-SCoRE Social Comparison Lexicon",
    "",
    "Use this lexicon as context-aware evidence, not as a hard keyword classifier.",
    "",
    "Decision order:",
    "1. Identify self-other / reader-poster positioning markers.",
    "2. Identify whether the poster is advantaged, disadvantaged, or not positioned relative to the reader.",
    "3. Apply contrastive checks: UP achievement/resource/appearance-success cues are weakened by failure/distress cues; DOWN hardship/failure cues are weakened by tutorial/news/product frames.",
    "4. If the text is mainly tutorial, product review, third-party news, or casual recommendation without reader-poster standing, choose NEUTRAL.",
    "",
]

for label in ["UP", "DOWN", "NEUTRAL", "AMBIGUOUS"]:
    prompt_lines.append(f"## {label}")
    for frame in sorted({r["frame"] for r in rows if r["target_label"] == label}):
        cues = [r["cue"] for r in rows if r["target_label"] == label and r["frame"] == frame]
        prompt_lines.append(f"- {frame}: " + "、".join(cues[:28]))
    prompt_lines.append("")

(OUT / "xhs_social_comparison_lexicon_for_prompt.md").write_text("\n".join(prompt_lines), encoding="utf-8")

print(json.dumps({
    "rows": len(rows),
    "csv": str(csv_path),
    "json": str(json_path),
    "prompt_md": str(OUT / "xhs_social_comparison_lexicon_for_prompt.md"),
    "summary": summary,
}, ensure_ascii=False, indent=2))
