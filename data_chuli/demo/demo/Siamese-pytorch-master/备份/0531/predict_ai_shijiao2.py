import json
from pathlib import Path

import ollama


class TailVehicleCheck:
    """
    使用本地 Ollama 直接比较两张原图中中央车辆的尾部，
    判断是否为“正常”“换挂”或“无法判断”。
    """

    VALID_LABELS = ["正常", "换挂", "无法判断"]

    def __init__(self, model_name: str = "gemma4:latest"):
        self.model_name = model_name
        self.last_error = ""
        self.last_raw_output = ""

    def _build_tail_compare_prompt(self) -> str:
        return (
            "软限制行为规则：\n"
            "0. 先做资格审查，再做编号或结构比较。若两侧挂车号牌已清晰可读且明确一致，直接输出正常（见下条），不得因结构被篷布遮挡而改判无法判断。\n"
            "0.1 不要为了得出换挂而放宽证据标准，也不要围绕同一处不可靠区域反复比较。\n"
            "0.2 理由最多使用1到2句话，不要输出长篇分步推理。\n"
            "你是一名车辆尾部复核员，需要比较两张原始图片中“中央车辆”的尾部是否属于同一辆挂车。\n\n"
            "成对可比对性审查（第0步，先于号牌/结构/颜色，必须最先执行）：\n"
            "- 分别判断图1、图2是否拍到挂车尾部（尾门/栏板/挂车号牌或放大号/挂车尾灯区），而非仅牵引车车头或驾驶室侧面。\n"
            "- 填写 img1_trailer_rear_visible、img2_trailer_rear_visible（是/否）；两侧均为 是 时 pair_comparable=是，否则 pair_comparable=否。\n"
            "- pair_comparable=否 → label 必须 无法判断，reason 写明哪一侧未见挂车尾、需回退主视角车尾AI；禁止换挂，禁止用颜色定案。\n"
            "- 禁止把牵引车头颜色与挂车颜色比较；颜色仅可在两侧均可见挂车尾部时参与 Tier-A 交叉校验，且不能单独判换挂。\n"
            "- pair_comparable=是 但 plate 与 structure 均为 无法确认 → label 必须 无法判断。\n"
            "- label=换挂 仅当 pair_comparable=是 且（plate=不一致 或 structure=不一致）。\n\n"
            "任务范围：\n"
            "1. 只关注两张图中央位置的那一辆挂车/半挂车。\n"
            "2. 只分析这辆车的尾部区域。\n"
            "3. 忽略其他车辆、路面、背景、天气、时间、阴影、反光、灯光和无关干扰；车斗内货物形状/颜色不参与判断，但不得用货物或临时篷布解释尾门型、侧围笼架等挂车本体结构。\n\n"
            "判定优先级必须严格遵守：\n"
            "最高优先级：号牌/放大号一致即正常（压过结构比对与无法判断）。\n"
            "- 若两侧挂车号牌或尾部放大号均清晰完整可读，且能明确确认省份简称、字母、数字及“挂”字等关键位一致，必须直接输出“正常”。\n"
            "- 此时 plate_or_number_consistency 填“一致”，structure_consistency 填“未检验”，立即停止，不得再进入第二优先级结构比对。\n"
            "- 即使一侧或两侧尾门、栏板、侧围因临时篷布、货物、阴影导致无法确认，也不得因此输出“无法判断”；reason 写明“号牌一致，结构因遮挡未检验”。\n"
            "- 仅当任一侧未拍到挂车尾部、挂车尾部过小过糊、或号牌/放大号均无法可靠读取时，才因证据不足输出“无法判断”。\n\n"
            "第一优先级：比对“真正的挂车身份编号”（与最高优先级衔接）。\n"
            "- 真正可作为挂车身份依据的信息包括：挂车号牌、尾部放大号、车架号/车身编号、正式喷涂的挂车识别编号。\n"
            "- 这些挂车身份编号必须直接来自中央车辆挂车本体本身，例如尾部号牌安装区、尾门/尾板放大号区域、车架或车身正式喷涂编号区域；不能从背景电子屏、道闸显示牌、场内指示牌、建筑物牌子、路边提示牌或任何非挂车本体对象上读取编号。\n"
            "- 如果画面中出现红色电子屏、道闸牌、场内车辆引导屏、停车提示牌等带数字或车牌样式的背景信息，这些大概率是场内显示信息或车头车牌信息，不属于挂车号牌或挂车放大号，不能据此判定“换挂”。\n"
            "- 中国挂车号牌通常由“省份简称 + 大写英文字母 + 数字”组成，例如“鲁A1234挂”“粤B5678挂”；这类信息属于强身份信息。\n"
            "- 只有当两张图中的强身份信息都清晰可见、字符完整可读，并且能够明确确认“一致”或“明显不一致”时，才可以直接据此判定“正常”或“换挂”。\n"
            "- 如果挂车号牌或放大号区域过小、距离过远、拍摄角度过斜、被尾板/车身/货物遮挡、夜间过暗、局部过曝、反光严重、画面模糊，导致只能勉强辨认出部分字符或疑似字符，则该编号证据一律视为“不可靠”，不能直接作为“换挂”依据。\n"
            "- 如果一张图能看到较清晰编号，另一张图的编号区域却很小、被遮挡或只能猜测字符，也不能直接用“看起来不同”判定换挂，而是要放弃编号比较，转入结构特征比对。\n"
            "- 对于只看到末尾几位、只看到局部字符、只能猜测是 Z9285/Z7409 这类不完整或不稳定识别结果，必须视为“无法可靠确认编号”，不能直接判定“换挂”。\n"
            "- 如果两张图中的挂车号牌或放大号都清晰可见、字符完整可读，并且明确一致，则直接判定为正常，不再比较栏板、尾门、尾灯等任何结构，不得输出无法判断。\n"
            "- 如果强身份信息被遮挡、缺失、看不清，或者无法确认是不是正式挂车身份编号，不要直接判定“换挂”，才进入第二优先级结构特征比对。\n\n"
            "挂车号牌眩光与“单侧清晰、单侧不可靠”专节（高优先级，严防误换挂）：\n"
            "- 夜间、雨夜、过磅灯、路灯、车头灯、现场补光灯等外来光源直射号牌区域时，常出现号牌中央字符被强光淹没、发白、过曝成亮斑，两侧字符仍隐约可见的情况；此时读出的号牌一律视为不可靠，plate_or_number_consistency 必须填“无法确认”，不能填“不一致”。\n"
            "- 典型不可靠情形：号牌中间几位被亮斑盖住只能“猜”为某数字或字母；在无法确认模糊侧是否读错、是否同一号牌被眩光改写的情况下，禁止仅凭号牌字符串不同就判“换挂”。\n"
            "- 规则：只要任意一侧挂车号牌存在眩光、过曝亮斑、字符缺失、只能辨认部分位、或两侧清晰度明显不对称（一侧清晰一侧难读），就不得把号牌差异作为换挂依据；必须立即转入第二优先级，按 Tier-A→颜色交叉校验→Tier-B 顺序比对。\n"
            "- 号牌不可靠时：必须先完成第二优先级 Tier-A（后开口、侧围）；仅当 Tier-A 全部一致时，才可用 Tier-B（尾灯、反光条、轴数、保险杠、挡泥板等）佐证“正常”，并在 reason 中写明“号牌不可靠，已依 Tier-A/B 结构比对”。\n"
            "- 仅当两侧号牌都清晰完整可读，且能明确看到省份简称、字母、数字及“挂”字等关键位均不同（不是眩光造成的假差异）时，才可仅凭号牌不一致判“换挂”。\n\n"
            "重要排除规则：以下内容不是挂车身份编号，不能单独作为“换挂”依据。\n"
            "- 危险品运输标识代码、介质编码、联合国编号、货物编号。\n"
            "- 货物/罐体/箱体上的两行或多行数字代码，例如“60 2874”“33 1114”这类危险品标识代码。\n"
            "- 危险品菱形标牌、燃/腐/爆等类别标识、限速标志、载重标识、公司广告、罐体宣传字样。\n"
            "- 货物品牌、介质名称、运输提示语、警示牌文字。\n"
            "- 背景电子屏、道闸显示牌、停车提示牌、场内指示牌、建筑物牌子、路边告示牌上的编号、车牌样式字符串或滚动文字。\n"
            "- 这些信息即使不同，也不能直接判定为“换挂”；同一挂车允许运输不同货物，因此可能出现不同货物标识代码。\n\n"
            "第二优先级（仅当第一优先级编号无法可靠下结论时启用）：\n"
            "步骤1 - 同位分区：尾门区 | 侧围下/中/上带 | 底盘灯带。只比较同一分区内的对应构件，禁止用侧栏字去对尾门区、禁止跨高度带混比。\n\n"
            "【货物误判防护-超短硬规则】\n"
            "1) 禁止把车斗内货物/堆料/绑带/篷布的轮廓当作尾门或侧围；无固定门板边界、中缝、侧梁立柱等车体连接证据，不得判“有尾门/有侧挡板”。\n"
            "2) 尾门/侧围任一项无法确认时不得写“结构一致”；一侧为实体尾门/侧围、另一侧仅为货物轮廓 → Tier-A 冲突或无法判断，禁止判正常。\n\n"
            "步骤2 - Tier-A（任一项“不一致”即换挂，禁止用光照、积灰、货物、篷布解释）：\n"
            "2A 后开口（尾门区，按序逐项比对，任一项冲突即不一致）：\n"
            "  (1) 有无尾门：无尾门敞顶 / 有尾板或尾门 / 厢式或罐式封闭尾 / 无法确认。\n"
            "  (2) 门型：无尾门敞顶 / 矮尾板单扇横门 / 全高竖门单扇 / 全高竖门双扇对开 / 后翻栏板 / 厢式后门 / 罐式尾封 / 无法确认。\n"
            "  (3) 门高比例：全高(占车尾大部分高度) / 半高 / 矮尾板(仅下部) / 无尾门 / 无法确认。\n"
            "  (4) 门扇数与中缝：0扇 / 1扇 / 2扇及可见中缝 / 无法确认。写“双尾门”时两侧都必须能看到两扇门或中缝，禁止对只有矮横板的一侧推断双扇。\n"
            "  (5) 固定顶/笼顶 vs 敞口：无固定顶敞口 / 固定顶梁或雨棚架 / 仓栅固定笼顶 / 活动软篷布(不算车体，但若下方门型已冲突仍判不一致) / 无法确认。\n"
            "  硬规则：无尾门 vs 全高双扇、矮尾板 vs 全高竖门 → 一律不一致，不得写成装载或夜间导致。\n"
            "2B 侧围（同一高度带比对，任一项冲突即不一致）：\n"
            "  (1) 有无侧挡板：平板无栏 / 有栏板 / 厢式或罐式侧壁 / 无法确认。\n"
            "  (2) 高低栏/笼车：低栏 / 中栏 / 高栏 / 仓栅笼车 / 无法确认。\n"
            "  (3) 实心/镂空/组合：实心竖板或横波纹实心 / 上半镂空下半实心 / 全镂空网格笼 / 仅骨架 / 无法确认。\n"
            "  (4) 侧栏喷涂：同高度带是否有永久放大号或大白字；一侧清晰有字、另一侧同带无喷涂 → 不一致。\n"
            "  硬规则：低栏实心 vs 仓栅上半镂空、实心侧板 vs 全镂空笼 → 一律不一致；禁止空泛写“均为栏板式”。\n"
            "  固定金属笼架/立柱网格属于车体侧围，不得写成货物或篷布遮挡顶部。\n\n"
            "步骤3 - 颜色交叉校验（仅 pair_comparable=是 且 Tier-A 之后执行）：\n"
            "  分别记录每张图挂车尾部的 body_hue（红/橙/黄/蓝/绿/白/灰/黑褐/不可辨）与 appearance_note（光照深暗/积灰/顶棚阴影/反光发白/无异常）。\n"
            "  - 禁止用牵引车头色相与挂车尾部色相比较。\n"
            "  - 仅 body_hue 不同且 Tier-A 全部一致 → 视为光照或脏污，不能单独换挂。\n"
            "  - body_hue 不同且 Tier-A 任一项不一致 → 换挂，reason 写明色相与门型/栏型均不同。\n"
            "  - 禁止在门型、栏高、镂空型已冲突时，仍写“阴影或顶棚导致色变”。\n\n"
            "步骤4 - Tier-B（仅当 Tier-A 全部一致，或 Tier-A 关键项均为无法确认且无冲突疑点时）：\n"
            "  尾灯总成外形(方灯/圆灯组合等)、尾部横反光条有无、轴数/可见轮组、下护栏保险杠形态、挡泥板、号牌架形态、侧挂附件。\n"
            "  Tier-B 不能压过 Tier-A：门型或侧围已冲突时，不得用“三轴、尾灯位置类似”判正常。\n"
            "  只有确认两侧为同一挂车本体且对应位置确实无同类安装位时，才可因 Tier-B 差异判换挂；不能仅凭一侧看不见就下结论。\n\n"
            "步骤5 - 结论：\n"
            "  - Tier-A 任一项不一致 → 换挂，structure_consistency 填“不一致”。\n"
            "  - Tier-A 一致且 Tier-B 明确不一致（双侧清晰） → 换挂。\n"
            "  - Tier-A 与 Tier-B 均一致 → 正常，structure_consistency 填“一致”。\n"
            "  - 关键 Tier-A 项无法确认且无结构冲突疑点 → 若号牌已一致则不得走到本步，应已在最高优先级判正常；仅号牌不可靠且结构也无法比对时才无法判断。\n\n"
            "侧挂附件与水箱/储物箱的特殊判读（高优先级，易误判）：\n"
            "- 平板车、栏板车、半挂车底盘侧面常见侧挂附件：水箱、储物箱、工具箱、阀门箱、电瓶箱等，通常安装在车架或悬挂梁下方、车轮上方区域。\n"
            "- 同一附件在两张图中可能呈现“一张近景清晰可见（如浅灰箱体），另一张远景/阴暗/沾灰后几乎看不见”——这通常不是换挂，而是成像距离、拍摄远近、车底阴影、雨天反光、长期积灰泥污掩盖箱体本色导致的可见性差异。\n"
            "- 若一张图在挂车侧面某固定相对位置（如尾挂侧梁下方、左/右侧车轮上方）有清晰浅灰/白色矩形箱体，另一张图在对应相对位置仅为深色阴影、黑块或与底盘融为一体的暗区，且两张图 Tier-A（后开口、侧围）已全部一致，应优先判断为“同一附件成像差异”，输出“正常”，并在 reason 中说明可能由距离远近或沾灰遮盖导致，不能写成“图2有箱图1无箱故换挂”。\n"
            "- 只有当一张图明确可见某类侧挂附件（含安装支架、固定螺栓位、箱体轮廓），另一张图在对应位置能清晰看到车架/侧梁但确实没有任何安装支架、开孔或箱体占位，且两侧成像清晰度足以排除“只是被阴影/灰尘遮住”时，才可把侧挂附件差异作为换挂依据。\n"
            "- 禁止仅凭“一侧有灰色储物箱/水箱、另一侧对应位置看不出箱体”就判定换挂；若无法确认另一侧是“真的没有”还是“有但被遮住/太远/太暗”，应输出“无法判断”或“正常”，不要强行换挂。\n\n"
            "注意事项：\n"
            "1. 必须只看中央车辆，不能拿边缘车辆或背景目标做判断。\n"
            "2. 颜色深浅、白天黑夜、雨雪雾、阴影、反光、模糊、视角轻微变化、摄像头远近（近景大、远景小），不能单独作为正常或换挂依据。\n"
            "3. **顶棚遮光与环境光照差异（仅适用于 Tier-A 已全部一致时）**：\n"
            "   - 过磅现场顶棚阴影可使车身发深黑/深灰/暗褐，露天可呈鲜红/橙等；这属于 appearance_note，不是单独换挂依据。\n"
            "   - **判定规则**：仅当 body_hue 不同且 Tier-A（门型、门高、侧围栏型、实心/镂空）全部一致时，才可因光照/积灰判正常；若门型或侧围已冲突（如矮尾板 vs 全高双扇、低栏 vs 仓栅笼、红 vs 黄且栏型不同），禁止引用本条掩盖换挂。\n"
            "4. 车辆长期运行后的积灰、泥污、锈蚀、掉漆、补漆，会显著改变表观明暗；不能仅凭“灰/深/浅”或“一侧有灰箱一侧无”就判定结构不同或换挂，但若 Tier-A 已冲突则与脏污无关仍应换挂。\n"
            "5. 临时篷布/货物遮挡尾门、栏板时，不得据此判换挂；但若号牌两侧清晰一致，必须判正常，不能写“结构无法确认故无法判断”。\n"
            "6. 如果看到的是货物标识代码、背景电子屏编号或场内指示牌编号，而不是挂车身份编号，必须在 reason 中明确说明“该编号不属于挂车号牌或放大号，因此不能据此判换挂”。\n"
            "7. 仅当号牌/放大号不可靠且结构也无法可靠比较时，才输出“无法判断”并回退主视角裁切车尾图；号牌可靠且一致时禁止输出无法判断。\n"
            "8. “无法判断”不得用于「号牌一致但结构被篷布挡住」的情形。\n\n"
            "请按以下 JSON 格式输出，且只能输出一个 JSON 对象，不要输出额外解释：\n"
            "{\n"
            '  "label": "正常/换挂/无法判断",\n'
            '  "reason": "一句到两句中文说明",\n'
            '  "img1_trailer_rear_visible": "是/否",\n'
            '  "img2_trailer_rear_visible": "是/否",\n'
            '  "pair_comparable": "是/否",\n'
            '  "plate_or_number_consistency": "一致/不一致/无法确认",\n'
            '  "structure_consistency": "一致/不一致/未检验/无法确认"\n'
            "}\n"
        )

    def _extract_json_payload(self, text: str) -> dict:
        text = (text or "").strip()
        if not text:
            return {}

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}

        candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return {}

    def _normalize_label(self, value: str) -> str:
        text = str(value or "").strip().lower()
        if not text:
            return "未知"

        if text in {"正常", "normal", "same", "一致"}:
            return "正常"
        if text in {"换挂", "change_trailer", "different", "异常", "不一致"}:
            return "换挂"
        if text in {"无法判断", "无法判定", "undetermined", "unknown", "insufficient", "insufficient_tail_evidence"}:
            return "无法判断"

        raw = str(value or "").strip()
        if raw in self.VALID_LABELS:
            return raw
        return "未知"

    def _normalize_consistency(self, value: str, *, kind: str) -> str:
        text = str(value or "").strip().lower()
        if kind == "number":
            if text in {"一致", "相同", "same", "normal"}:
                return "一致"
            if text in {"不一致", "不同", "different", "换挂"}:
                return "不一致"
            if text in {"无法确认", "未知", "看不清", "unclear", "unknown", "未识别"}:
                return "无法确认"
            return "无法确认"

        if text in {"一致", "相同", "same", "normal"}:
            return "一致"
        if text in {"不一致", "不同", "different", "换挂"}:
            return "不一致"
        if text in {"未检验", "未检查", "not_checked", "not checked"}:
            return "未检验"
        if text in {"无法确认", "未知", "看不清", "unclear", "unknown", "无法比较", "不可比"}:
            return "无法确认"
        return "未检验"

    def _normalize_visibility(self, value: str) -> str:
        text = str(value or "").strip().lower()
        if text in {"是", "yes", "y", "true", "1", "可见", "有"}:
            return "是"
        if text in {"否", "no", "n", "false", "0", "不可见", "无"}:
            return "否"
        return "未知"

    def _normalize_pair_comparable(self, value: str) -> str:
        text = str(value or "").strip().lower()
        if text in {"是", "yes", "true", "1", "可比", "可以"}:
            return "是"
        if text in {"否", "no", "false", "0", "不可比", "无法比对"}:
            return "否"
        return "未知"

    def _apply_comparability_rules(
        self,
        *,
        label: str,
        reason: str,
        img1_visible: str,
        img2_visible: str,
        pair_comparable: str,
        plate_consistency: str,
        structure_consistency: str,
    ) -> dict:
        reason = str(reason or "").strip()
        fallback_reason = "一侧未拍到挂车尾部，本视角无法成对比对，需回退主视角车尾裁切图"

        if pair_comparable == "未知":
            if img1_visible == "是" and img2_visible == "是":
                pair_comparable = "是"
            elif img1_visible == "否" or img2_visible == "否":
                pair_comparable = "否"

        if pair_comparable == "否" or img1_visible == "否" or img2_visible == "否":
            if fallback_reason not in reason:
                reason = f"{reason}；{fallback_reason}" if reason else fallback_reason
            return {
                "label": "无法判断",
                "reason": reason,
                "img1_trailer_rear_visible": img1_visible if img1_visible != "未知" else "否",
                "img2_trailer_rear_visible": img2_visible if img2_visible != "未知" else "否",
                "pair_comparable": "否",
                "plate_or_number_consistency": plate_consistency,
                "structure_consistency": structure_consistency,
            }

        insufficient_keywords = (
            "未拍到挂车尾部",
            "未见挂车尾",
            "未拍摄到挂车",
            "仅拍到车头",
            "仅牵引车",
            "无法确认结构",
            "无法成对比对",
            "需回退主视角",
        )
        if label == "换挂" and any(keyword in reason for keyword in insufficient_keywords):
            if fallback_reason not in reason:
                reason = f"{reason}；{fallback_reason}"
            return {
                "label": "无法判断",
                "reason": reason,
                "img1_trailer_rear_visible": img1_visible,
                "img2_trailer_rear_visible": img2_visible,
                "pair_comparable": pair_comparable,
                "plate_or_number_consistency": plate_consistency,
                "structure_consistency": structure_consistency,
            }

        if label == "换挂" and plate_consistency != "不一致" and structure_consistency != "不一致":
            weak_reason = "号牌与结构均无明确不一致证据，不得仅凭颜色判换挂，需回退主视角车尾裁切图"
            if weak_reason not in reason:
                reason = f"{reason}；{weak_reason}" if reason else weak_reason
            return {
                "label": "无法判断",
                "reason": reason,
                "img1_trailer_rear_visible": img1_visible,
                "img2_trailer_rear_visible": img2_visible,
                "pair_comparable": pair_comparable,
                "plate_or_number_consistency": plate_consistency,
                "structure_consistency": structure_consistency,
            }

        if (
            label == "换挂"
            and plate_consistency == "无法确认"
            and structure_consistency in {"无法确认", "未检验"}
        ):
            weak_reason = "号牌与结构均无法确认，本视角不得仅凭颜色判换挂，需回退主视角车尾裁切图"
            if weak_reason not in reason:
                reason = f"{reason}；{weak_reason}" if reason else weak_reason
            return {
                "label": "无法判断",
                "reason": reason,
                "img1_trailer_rear_visible": img1_visible,
                "img2_trailer_rear_visible": img2_visible,
                "pair_comparable": pair_comparable,
                "plate_or_number_consistency": plate_consistency,
                "structure_consistency": structure_consistency,
            }

        return {
            "label": label,
            "reason": reason,
            "img1_trailer_rear_visible": img1_visible,
            "img2_trailer_rear_visible": img2_visible,
            "pair_comparable": pair_comparable if pair_comparable != "未知" else "是",
            "plate_or_number_consistency": plate_consistency,
            "structure_consistency": structure_consistency,
        }

    def _fallback_label_from_text(self, text: str) -> str:
        plain = str(text or "")
        plain_lower = plain.lower()
        if (
            "无法判断" in plain
            or "无法判定" in plain
            or "证据不足" in plain
            or "信息不足" in plain
            or "未拍到尾部" in plain
            or "未拍到挂车" in plain
            or "未见挂车尾" in plain
            or "仅拍到车头" in plain
            or "无法成对比对" in plain
            or "需回退主视角" in plain
            or "回退主视角" in plain
            or "undetermined" in plain_lower
            or "insufficient" in plain_lower
        ):
            return "无法判断"
        if "换挂" in plain or "change_trailer" in plain.lower():
            return "换挂"
        if "正常" in plain or "normal" in plain.lower():
            return "正常"
        return "未知"

    def _empty_tail_result(self) -> dict:
        return {
            "label": "未知",
            "reason": "",
            "img1_trailer_rear_visible": "未知",
            "img2_trailer_rear_visible": "未知",
            "pair_comparable": "未知",
            "plate_or_number_consistency": "无法确认",
            "structure_consistency": "未检验",
        }

    def _call_model(self, img1_path: str, img2_path: str) -> dict:
        img1 = Path(img1_path)
        img2 = Path(img2_path)
        if not img1.exists() or not img2.exists():
            self.last_error = f"image not found: {img1_path} | {img2_path}"
            return self._empty_tail_result()

        try:
            self.last_error = ""
            self.last_raw_output = ""
            stream = ollama.chat(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": self._build_tail_compare_prompt(),
                    "images": [str(img1), str(img2)],
                }],
                stream=True,
            )

            print("\n--- AI分析中 ---\n")
            for chunk in stream:
                content = chunk.get("message", {}).get("content", "")
                if content:
                    print(content, end="", flush=True)
                    self.last_raw_output += content
            print("\n\n--- AI分析结束 ---\n")

            payload = self._extract_json_payload(self.last_raw_output)

            label = self._normalize_label(payload.get("label"))
            if label == "未知":
                label = self._fallback_label_from_text(self.last_raw_output)

            reason = str(payload.get("reason") or "").strip()
            if not reason:
                lines = [line.strip() for line in self.last_raw_output.splitlines() if line.strip()]
                reason = lines[0] if lines else ""

            plate_or_number_consistency = self._normalize_consistency(
                payload.get("plate_or_number_consistency"),
                kind="number",
            )
            structure_consistency = self._normalize_consistency(
                payload.get("structure_consistency"),
                kind="structure",
            )
            img1_visible = self._normalize_visibility(payload.get("img1_trailer_rear_visible"))
            img2_visible = self._normalize_visibility(payload.get("img2_trailer_rear_visible"))
            pair_comparable = self._normalize_pair_comparable(payload.get("pair_comparable"))

            if label == "未知":
                label = "无法判断"
                if not reason:
                    reason = "尾部视角信息不足或模型未能稳定输出标准结果，需要回退主视角裁切车尾图继续判断。"
                if plate_or_number_consistency == "一致":
                    structure_consistency = "未检验"

            result = self._apply_comparability_rules(
                label=label,
                reason=reason,
                img1_visible=img1_visible,
                img2_visible=img2_visible,
                pair_comparable=pair_comparable,
                plate_consistency=plate_or_number_consistency,
                structure_consistency=structure_consistency,
            )
            if result["label"] != label:
                print(
                    f"[tail-ai] comparability guard adjusted label: {label!r} -> {result['label']!r}"
                )
            return result
        except Exception as e:
            self.last_error = str(e)
            print(f"调用异常: {e}")
            print("请检查 Ollama 服务是否已启动，以及模型名称是否可用。")
            return self._empty_tail_result()

    def check_tail_on_original(self, img1_path: str, img2_path: str) -> dict:
        return self._call_model(img1_path, img2_path)


if __name__ == "__main__":
    checker = TailVehicleCheck()

    img1 = r"D:\\project\\image1.jpg"
    img2 = r"D:\\project\\image2.jpg"

    p1 = Path(img1)
    p2 = Path(img2)
    print(f"model: {checker.model_name}")
    print(f"img1 exists: {p1.exists()} -> {p1.resolve() if p1.exists() else img1}")
    print(f"img2 exists: {p2.exists()} -> {p2.resolve() if p2.exists() else img2}")

    result = checker.check_tail_on_original(img1, img2)
    print("\nfinal result:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if checker.last_error:
        print(f"last_error: {checker.last_error}")
