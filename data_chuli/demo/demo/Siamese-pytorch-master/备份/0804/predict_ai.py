import json
from typing import Any, Dict, Optional

import ollama
from pathlib import Path


def _crop_flag_text(ok: Optional[bool]) -> str:
    return "成功" if ok else "失败"


def _build_head_crop_context(crop_status: Optional[Dict[str, Any]]) -> str:
    if not crop_status:
        return ""

    return (
        "【裁切状态（由系统提供，必须采信）】\n"
        f"- 图1整车裁切：{_crop_flag_text(crop_status.get('vehicle1_ok'))}\n"
        f"- 图2整车裁切：{_crop_flag_text(crop_status.get('vehicle2_ok'))}\n"
        f"- 图1车头裁切：{_crop_flag_text(crop_status.get('head1_ok'))}\n"
        f"- 图2车头裁切：{_crop_flag_text(crop_status.get('head2_ok'))}\n\n"
        "【不对称裁切规则（最高优先，高于一票否决）】\n"
        "C1. 若仅一侧车头裁切失败：优先观察裁切失败侧画面中，过磅车道中央是否存在清晰卡车车头。\n"
        "C2. 若失败侧根本无车（空车道、仅行人/地磅设备/背景建筑）：直接判 fake_plate，reason 必须写「裁切失败侧无目标车辆」。\n"
        "C3. 若失败侧仍有车（只是裁切失败导致为全景）：进入「全景 vs 特写」规则，不得仅凭尺度差异判套牌。\n\n"
        "【全景 vs 特写规则】\n"
        "P1. 禁止比对格栅条纹数、小字、局部反光等微小结构。\n"
        "P2. 只比整体外观：驾驶室轮廓、导流罩形态、大灯布局、保险杠大体形状、品牌区域有无。\n"
        "P3. 若因一张全景一张特写导致上述整体特征仍无法可靠对比："
        "reason 必须写「输入图片质量太差，AI无法判断」，label 必须设为 unknown，"
        "禁止强行 fake_plate 或 normal。\n\n"
    )


def _build_main_tail_crop_context(crop_status: Optional[Dict[str, Any]]) -> str:
    if not crop_status:
        return ""

    return (
        "【裁切状态（由系统提供，必须采信）】\n"
        f"- 图1整车裁切：{_crop_flag_text(crop_status.get('vehicle1_ok'))}\n"
        f"- 图2整车裁切：{_crop_flag_text(crop_status.get('vehicle2_ok'))}\n"
        f"- 图1主视角车尾裁切：{_crop_flag_text(crop_status.get('main_tail1_ok'))}\n"
        f"- 图2主视角车尾裁切：{_crop_flag_text(crop_status.get('main_tail2_ok'))}\n\n"
        "【不对称裁切规则（最高优先）】\n"
        "C1. 若仅一侧主视角车尾裁切失败：优先观察失败侧画面中，是否可见牵引车+挂车侧后段/栏板/轮轴。\n"
        "C2. 若失败侧根本无挂车侧后段（空车道、仅行人/设备）：reason 写「输入图片质量太差，AI无法判断」，label 设为 unknown。\n"
        "C3. 若失败侧仍有挂车侧后段（只是裁切失败导致为全景）：进入「全景 vs 特写」规则。\n\n"
        "【全景 vs 特写规则】\n"
        "P1. 禁止比对栏板纹理、小安装件等微小结构。\n"
        "P2. 只比整体：栏高大类、轴数布局、侧挡板有无、侧挂附件大体形态。\n"
        "P3. 若整体仍无法可靠对比：reason 写「输入图片质量太差，AI无法判断」，label 设为 unknown，"
        "禁止强行 change_trailer 或 normal。\n\n"
    )


class VehicleCheck:
    """
    使用视觉模型对车头/主视角车尾裁切图做二次复核（fake_plate 或 change_trailer / normal）。
    """

    HEAD_VALID_LABELS = ["fake_plate", "normal"]
    TAIL_VALID_LABELS = ["change_trailer", "normal"]

    def __init__(self, model_name="gemma4:latest"):
        self.model_name = model_name
        self.last_error = ""
        self.last_raw_output = ""

    def _build_head_prompt(
        self,
        low_similarity_fallback_label: str = "normal",
        crop_status: Optional[Dict[str, Any]] = None,
    ):
        fallback_conclusion_text = (
            "车头相似度低于阈值，判定为套牌"
            if low_similarity_fallback_label == "fake_plate"
            else "车头相似度大于阈值，判定为正常"
        )
        crop_context = _build_head_crop_context(crop_status)
        return (
            "你现在只比较两张车头裁切图，只判断：fake_plate 或 normal。\n\n"
            f"{crop_context}"
            "一票否决（最高优先级，违反任一条则不得仅凭文字判 fake_plate）：\n"
            "V1. 一侧有字、一侧无字，或一侧发白/字符看不清 → 不得单独判套牌；须先判断是否为过曝、反光、阴影、背光或时段光照所致。\n"
            "V2. 任一侧导流罩、引擎盖、车门、遮阳板文字区存在强反光、过曝、发白、深阴影或亮斑盖住字符 → 该子区域文字证据作废，reason 须写明成像不可靠。\n"
            "V3. 格栅、大灯、保险杠、车标等硬结构整体一致时，默认 normal；仅凭文字可见性不同不能改判套牌。\n"
            "V4. 导流罩文字须先对齐子区域（deflector_front_center / deflector_side_model）；跨子区域或正侧面混比 → 证据无效。\n"
            "V5. 仅当同一子区域、两侧都清晰可读时，才可用品牌/厂家字样冲突作为套牌依据。\n"
            "V6. reason 若写「图1无、图2有」或相反，必须同时写明两侧可读性（清晰/过曝/阴影/反光）；未写明则 label 必须为 normal。\n"
            "V7. 清晨顶面直射、夜间路灯或车头灯、强背光等大光比场景下，导流罩仅一张可见文字 → 优先归光照干扰，不得写成稳定标识差异。\n\n"
            "同位比对原则：\n"
            "1. 只比两图「同一固定部位、同一朝向面」上的长期标识或稳定结构；禁止图1一面文字对图2另一面。\n"
            "2. 无法确认同一部位、同一朝向面 → 文字差异无效，改比格栅、车标、保险杠、后视镜等硬结构，或判 normal。\n"
            "3. 引用文字/标识差异前必须先完成部位对齐；未对齐则无效。\n"
            "4. 禁止「都在导流罩上」即同部位；仅 deflector_front_center（正前中央品牌条）与 deflector_side_model（侧面型号条）可互比，且须同子区域。\n"
            "5. 引擎盖、车门、遮阳板：左对左、右对右、中央对中央；禁止侧面字对正面字。\n\n"
            "总原则：\n"
            "6. 只依据稳定结构和稳定标识；不得单凭色号深浅、亮暗、反光、阴影、过曝、污渍、角度、开灯、打码下结论。\n"
            "7. 建筑物遮挡、玻璃反光、夜间灯光、单边开灯、车外投影等默认成像干扰，不是套牌依据。\n"
            "8. 车内物品、单边车牌打码、货物橙黄编号牌、纯数字编号块、车牌黑块打码，均不是车体结构或身份标识依据。\n"
            f"9. 若任一张图主要为过磅设备、建筑物、招牌等非车头主体，或主体过小、过糊、强反光、过曝、遮挡、靠猜测 → 写「输入图片质量太差，AI无法判断、{fallback_conclusion_text}」，label 设为 {low_similarity_fallback_label}，勿展开长篇分析。\n"
            "10. 关键证据不可靠时不得为得出 fake_plate 而放宽标准；快速收束结论。\n\n"
            "高优先级观察项：\n"
            "11. 优先比：格栅结构、大灯外轮廓、保险杠、后视镜总成形状与分色、车标轮廓与图案、驾驶室与引擎盖整体造型。\n"
            "12. 引擎盖固定标识、导流罩同子区域喷涂字、车门编号、后视镜分色——仅在同部位对齐、双侧清晰、非反光/过曝/遮挡造成时，可作辅证；不含一侧有字一侧无字（见 V1）。\n"
            "13. 车标不比发光/反光强弱；大灯比外轮廓与布局，开灯差异不算结构差异；后视镜比总成主体形状与分色。\n\n"
            "低优先级或排除项：\n"
            "14. 细小装饰、后加装件、格栅条幅亮暗与金属光泽、跨面文字差异 → 不得单独作套牌依据。\n"
            "15. 文字区仅当两图同子区域均清晰、无强反光/过曝/阴影遮盖时，才可据文字内容判异；任一侧不可靠则作废（见 V1–V2），禁止写成稳定标识差异。\n\n"
            "光照与成像（摘要）：\n"
            "过磅现场正午直射、清晨顶光、傍晚斜射、夜间点光源、背光等常使导流罩/车门文字一侧清晰一侧不可见或顶面发白洗字，属成像差异。车身色号在早晚光下可漂移；硬结构一致时色号差判 normal。车门编号仅双侧清晰且确认非光照洗掉才可采信。顶棚/雨棚致一侧深阴影、另一侧明亮时，禁止写「图1无、图2有」。\n"
            "顶示廓灯（子区域=deflector_top_lamp_strip）：两侧均能见灯座或透镜外形才可比，否则写单侧不可读、成像差异 → normal。雨刷位置不算结构差异；玻璃上字须确认固定标识且双侧清晰。\n\n"
            "思考顺序（按序执行，不可跳步）：\n"
            "步骤1 硬结构：格栅、大灯、保险杠、车标、后视镜 → 整体一致则记下：仅凭文字差异不得判套牌（V3）。\n"
            "步骤2 子区域对齐：导流罩等文字须 deflector_front_center 或 deflector_side_model 对齐；不对齐则文字全部作废。\n"
            "步骤3 可读性：各涉及子区域分别判图1/图2 为清晰、过曝、阴影或反光；任一侧不可靠则该子区域文字作废（V2）。\n"
            "步骤4 定案：仅当硬结构有明确稳定差异，或（同子区域 + 双侧清晰 + 品牌/厂家级文字明显冲突且不触发 V1–V3）时 → fake_plate；否则 normal。\n\n"
            "分类规则：\n"
            "- fake_plate：须满足——硬结构存在清晰稳定差异且不能用光照、反光、过曝、遮挡等解释；或同子区域、双侧清晰可读、部位对齐且品牌/厂家级文字明显冲突（且不触发一票否决 V1–V3）。\n"
            "- normal：硬结构一致且仅文字一侧可见一侧不可见 → normal；或差异可归因光照/反光/过曝/污渍/遮挡/跨面混比/货物牌/打码/普通装饰/成像不可靠。\n\n"
            "输出要求：\n"
            "请按以下 JSON 格式输出，且只能输出一个 JSON 对象，不要输出额外解释：\n"
            "{\n"
            '  "label": "fake_plate 或 normal 或 unknown",\n'
            f'  "reason": "优先按模板：子区域=…；图1可读性=清晰|过曝|阴影|反光；图2可读性=…；硬结构=一致|不一致；文字证据=采纳|作废(原因)。'
            "一句到两句中文；触发 C2 无目标车辆时 label=fake_plate；触发 P3 全景vs特写不可比时写输入图片质量太差、AI无法判断，label=unknown；"
            f"其他图片质量太差（非裁切问题）时可写输入图片质量太差、AI无法判断、{fallback_conclusion_text}，label 设为 {low_similarity_fallback_label}；"
            "同部位对齐=否或任一侧不可读或触发 V1–V2 时 label 必须为 normal；"
            "禁止在 reason 写稳定标识差异的同时未说明两侧可读性；"
            'reason 只用中文，禁止出现 fake_plate、normal 等英文 label 词，可写套牌、正常、不能据此判套牌、不作为套牌依据"\n'
            "}\n"
        )

    def _build_tail_prompt(
        self,
        low_similarity_fallback_label: str = "normal",
        crop_status: Optional[Dict[str, Any]] = None,
    ):
        """主视角/车头方向下的车尾裁切图（tail1/tail2），非正后方尾部视角。"""
        fallback_conclusion_text = (
            "车尾相似度低于或等于阈值，判定为换挂"
            if low_similarity_fallback_label == "change_trailer"
            else "车尾相似度高于阈值，判定为正常"
        )
        crop_context = _build_main_tail_crop_context(crop_status)
        return (
            f"{crop_context}"
            "软限制：\n"
            "0. 先做资格审查；主体过小、过糊、强反光、过曝、深阴影遮挡、只能猜测时，"
            f"写「输入图片质量太差，AI无法判断、{fallback_conclusion_text}」，label 设为 {low_similarity_fallback_label}，勿展开长篇分析。\n"
            "0.1 不要为了 change_trailer 放宽标准；理由最多1到2句。\n"
            "0.2 你只判断 change_trailer 或 normal；全景vs特写不可比（P3）时 label 设为 unknown。\n\n"
            "视角（最高优先级）：\n"
            "输入是主视角/车头方向下的车尾裁切图，常见为挂车侧后段、侧挡板、轮轴、挡泥板、底盘侧挂附件。\n"
            "禁止因本视角看不到的部位猜测换挂；不得把货物、篷布、堆料轮廓当作侧挡板或车体结构。\n\n"
            "总原则：禁止单凭栏板颜色、明暗、材质表观、附件颜色、昼夜差异判 change_trailer；货物/篷布不算车体结构。\n\n"
            "主流程（按序执行，禁止跳步）：\n"
            "Step0 每图光照归因：分别审查 img1/img2 栏板区；存在「顶棚遮挡欠曝、夜间或暗环境、外来强光/车灯、栏板光滑反光、积灰泛白」任一项 → 该侧 panel_hue_reliable=否，栏板颜色/材质不得单独定换挂。\n"
            "Step1 几何（优先于颜色）：比竖筋/立柱数量与间距、栏高（低/中/高栏）、镂空笼 vs 实心栏、波形/平板轮廓、轴数与轮组布局；遮挡写无法确认，不得猜测；禁止凭表观色推断材质（金属/木质等）。\n"
            "Step2 安装件（须双侧清晰才可判差异）：侧面反光条/下护栏、挡泥板形态、侧挂水箱/储物箱/工具箱；一侧仅阴影/暗区/远景看不见 → 无法确认，不得判有无差异；附件颜色 alone 不定换挂。\n"
            "Step3 栏板色（末位）：仅当 Step1 几何一致、Step2 安装件一致或均无法确认、且无放大号/号牌可比对时，才比栏板色相；且两侧 panel_hue_reliable 均为是才可引用颜色。\n"
            "Step4 色相不可信兜底：任一侧 panel_hue_reliable=否，且仅有栏板颜色/明暗/材质表观差 → label 必须 normal。\n\n"
            "分类规则：\n"
            "- change_trailer：Step1 或 Step2 存在清晰稳定差异，且非 Step0 光照/脏污/货物/篷布可解释；reason 须点明竖筋/栏高/轴数/安装件等几何依据，禁止空写「颜色/材质不同」。\n"
            "- normal：Step1 与 Step2 一致或无法确认；或差异可归因 Step0 成像；或触发 Step4。\n\n"
            "输出要求：\n"
            "请按以下 JSON 格式输出，且只能输出一个 JSON 对象，不要输出额外解释：\n"
            "{\n"
            '  "label": "change_trailer 或 normal 或 unknown",\n'
            '  "reason": "一句到两句中文，点明栏型/轴数/侧挂附件等具体依据，禁止空泛；'
            "触发 P3 全景vs特写不可比时写输入图片质量太差、AI无法判断，label=unknown；"
            "证据不足、栏板色相不可信或差异可归因光照/阴影/货物/篷布/成像时 label 必须为 normal；"
            'reason 只用中文，禁止出现 change_trailer、normal 等英文 label 词，可写换挂、正常、不能据此判换挂；'
            "栏板色相不可信时 reason 须写几何/安装件一致或证据不足，禁止仅以颜色/材质定换挂\"\n"
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

    def _normalize_head_label(self, value: str) -> str:
        text = str(value or "").strip().lower()
        if not text:
            return "unknown"

        if text in {"normal", "正常", "same", "一致"}:
            return "normal"
        if text in {"fake_plate", "套牌", "fake", "cloned", "different", "异常"}:
            return "fake_plate"
        if text in {"unknown", "无法判断", "无法判定", "undetermined"}:
            return "unknown"

        raw = str(value or "").strip()
        if raw in self.HEAD_VALID_LABELS:
            return raw
        return "unknown"

    def _fallback_head_label_from_text(self, text: str) -> str:
        lines = [line.strip().lower() for line in str(text or "").splitlines() if line.strip()]
        if not lines:
            return "unknown"

        prefixes = ("label:", "label：", "结论:", "结论：", "result:", "result：")
        for line in reversed(lines[-3:]):
            normalized = line.strip(" `\"'")
            for prefix in prefixes:
                if normalized.startswith(prefix):
                    normalized = normalized[len(prefix):].strip()
                    break
            if normalized in self.HEAD_VALID_LABELS:
                return normalized

        last_line = lines[-1].strip(" `\"'")
        if last_line in self.HEAD_VALID_LABELS:
            return last_line
        return "unknown"

    def _parse_head_response(self, full_output: str) -> dict:
        payload = self._extract_json_payload(full_output)
        label = self._normalize_head_label(payload.get("label"))
        if label == "unknown":
            label = self._fallback_head_label_from_text(full_output)

        reason = str(payload.get("reason") or "").strip()
        if not reason:
            lines = [line.strip() for line in full_output.splitlines() if line.strip()]
            reason = lines[0] if lines else ""

        if label == "unknown":
            print("车头 AI JSON 解析失败，返回 unknown 交由上层回退")

        return {"label": label, "reason": reason}

    def _normalize_tail_label(self, value: str) -> str:
        text = str(value or "").strip().lower()
        if not text:
            return "unknown"

        if text in {"normal", "正常", "same", "一致"}:
            return "normal"
        if text in {
            "change_trailer",
            "换挂",
            "换车厢",
            "trailer_changed",
            "different_trailer",
        }:
            return "change_trailer"
        if text in {"unknown", "无法判断", "无法判定", "undetermined"}:
            return "unknown"

        raw = str(value or "").strip()
        if raw in self.TAIL_VALID_LABELS:
            return raw
        return "unknown"

    def _fallback_tail_label_from_text(self, text: str) -> str:
        lines = [line.strip().lower() for line in str(text or "").splitlines() if line.strip()]
        if not lines:
            return "unknown"

        prefixes = ("label:", "label：", "结论:", "结论：", "result:", "result：")
        for line in reversed(lines[-3:]):
            normalized = line.strip(" `\"'")
            for prefix in prefixes:
                if normalized.startswith(prefix):
                    normalized = normalized[len(prefix):].strip()
                    break
            if normalized in self.TAIL_VALID_LABELS:
                return normalized

        last_line = lines[-1].strip(" `\"'")
        if last_line in self.TAIL_VALID_LABELS:
            return last_line
        return "unknown"

    def _parse_tail_response(self, full_output: str) -> dict:
        payload = self._extract_json_payload(full_output)
        label = self._normalize_tail_label(payload.get("label"))
        if label == "unknown":
            label = self._fallback_tail_label_from_text(full_output)

        reason = str(payload.get("reason") or "").strip()
        if not reason:
            lines = [line.strip() for line in full_output.splitlines() if line.strip()]
            reason = lines[0] if lines else ""

        if label == "unknown":
            print("车尾 AI JSON 解析失败，返回 unknown 交由上层回退")

        return {"label": label, "reason": reason}

    def _extract_result(self, text: str, valid_keywords: list = None) -> str:
        if valid_keywords is None:
            valid_keywords = ["fake_plate", "change_trailer", "normal"]

        text_lower = text.lower()
        lines = [line.strip().lower() for line in text.split("\n") if line.strip()]

        if not lines:
            return "unknown"

        normalized_prefixes = [
            "结论：",
            "结论:",
            "答案：",
            "答案:",
            "result：",
            "result:",
            "label：",
            "label:",
        ]

        valid_keyword_set = set(valid_keywords)
        
        # 优先从最后几行提取（AI通常在最后输出结论）
        for line in reversed(lines):
            normalized = line
            for prefix in normalized_prefixes:
                normalized = normalized.replace(prefix, "")
            normalized = normalized.strip()
            if normalized in valid_keyword_set:
                return normalized

        # 如果最后几行没找到，检查最后一行是否包含关键词（单独一行）
        if lines:
            last_line = lines[-1]
            for keyword in valid_keywords:
                if keyword == last_line:
                    return keyword
        
        # 改进：使用更智能的匹配策略
        # 1. 优先匹配否定句式（"并非 xxx"、"不是 xxx"）
        negative_patterns = {
            "fake_plate": ["并非 fake_plate", "不是 fake_plate", "并非fake_plate", "不是fake_plate"],
            "normal": ["并非 normal", "不是 normal", "并非normal", "不是normal"],
            "change_trailer": ["并非 change_trailer", "不是 change_trailer", "并非change_trailer", "不是change_trailer"],
        }
        
        for keyword in valid_keywords:
            for neg_pattern in negative_patterns.get(keyword, []):
                if neg_pattern in text_lower:
                    # 如果找到否定句式，排除这个关键词
                    valid_keywords = [k for k in valid_keywords if k != keyword]
                    break
        
        # 2. 优先匹配肯定句式（"是 xxx"、"判定为 xxx"、"属于 xxx"）
        positive_patterns = {
            "fake_plate": ["是 fake_plate", "判定为 fake_plate", "属于 fake_plate", "是fake_plate", "判定为fake_plate", "属于fake_plate"],
            "normal": ["是 normal", "判定为 normal", "属于 normal", "是normal", "判定为normal", "属于normal", "是 **normal**"],
            "change_trailer": ["是 change_trailer", "判定为 change_trailer", "属于 change_trailer", "是change_trailer", "判定为change_trailer", "属于change_trailer"],
        }
        
        for keyword in valid_keywords:
            for pos_pattern in positive_patterns.get(keyword, []):
                if pos_pattern in text_lower:
                    return keyword
        
        # 3. 最后才使用简单的关键词匹配（作为兜底）
        for keyword in valid_keywords:
            if keyword in text_lower:
                return keyword

        return "unknown"

    def _extract_reason(self, text: str, valid_keywords: list = None) -> str:
        if valid_keywords is None:
            valid_keywords = ["fake_plate", "change_trailer", "normal"]

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return ""

        valid_keyword_set = {keyword.lower() for keyword in valid_keywords}
        reason_lines = list(lines)

        for idx in range(len(lines) - 1, -1, -1):
            normalized = lines[idx].strip().lower()
            normalized = normalized.replace("result:", "").replace("label:", "").strip()
            if normalized in valid_keyword_set:
                reason_lines = lines[:idx]
                break

        return " ".join(reason_lines).strip().strip("\"'")

    def _call_model(self, prompt: str, img1_path: str, img2_path: str, valid_keywords: list = None) -> str:
        if not Path(img1_path).exists() or not Path(img2_path).exists():
            self.last_error = f"image not found: {img1_path} | {img2_path}"
            return "unknown"

        try:
            self.last_error = ""
            stream = ollama.chat(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": prompt,
                    "images": [img1_path, img2_path]
                }],
                stream=True
            )

            full_output = ""
            print("\n--- AI分析中 ---\n")

            for chunk in stream:
                content = chunk.get("message", {}).get("content", "")
                if content:
                    print(content, end="", flush=True)
                    full_output += content

            print("\n\n--- AI分析结束 ---\n")

            result = self._extract_result(full_output, valid_keywords)
            if result == "unknown":
                print("模型结果不可判定，返回 unknown 交由上层回退")
                return "normal"

            return result
        except Exception as e:
            self.last_error = str(e)
            print(f"调用异常: {e}")
            print("请检查 Ollama 服务是否启动，以及模型名是否已拉取。")
            return "unknown"

    def _call_head_model_with_reason(
        self,
        prompt: str,
        img1_path: str,
        img2_path: str,
    ) -> dict:
        if not Path(img1_path).exists() or not Path(img2_path).exists():
            self.last_error = f"image not found: {img1_path} | {img2_path}"
            return {"label": "unknown", "reason": ""}

        try:
            self.last_error = ""
            self.last_raw_output = ""
            stream = ollama.chat(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": prompt,
                    "images": [img1_path, img2_path],
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

            return self._parse_head_response(self.last_raw_output)
        except Exception as e:
            self.last_error = str(e)
            print(f"调用异常: {e}")
            print("请检查 Ollama 服务是否启动，以及模型名是否已拉取。")
            return {"label": "unknown", "reason": ""}

    def _call_tail_model_with_reason(
        self,
        prompt: str,
        img1_path: str,
        img2_path: str,
    ) -> dict:
        if not Path(img1_path).exists() or not Path(img2_path).exists():
            self.last_error = f"image not found: {img1_path} | {img2_path}"
            return {"label": "unknown", "reason": ""}

        try:
            self.last_error = ""
            self.last_raw_output = ""
            stream = ollama.chat(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": prompt,
                    "images": [img1_path, img2_path],
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

            return self._parse_tail_response(self.last_raw_output)
        except Exception as e:
            self.last_error = str(e)
            print(f"调用异常: {e}")
            print("请检查 Ollama 服务是否启动，以及模型名是否已拉取。")
            return {"label": "unknown", "reason": ""}

    def check_head(self, head1_path: str, head2_path: str) -> str:
        payload = self._call_head_model_with_reason(
            self._build_head_prompt(),
            head1_path,
            head2_path,
        )
        label = str(payload.get("label") or "unknown")
        if label == "unknown":
            return "unknown"
        return label

    def check_tail(self, tail1_path: str, tail2_path: str) -> str:
        payload = self._call_tail_model_with_reason(
            self._build_tail_prompt(),
            tail1_path,
            tail2_path,
        )
        label = str(payload.get("label") or "unknown")
        if label == "unknown":
            return "unknown"
        return label

    def _call_model_with_reason(
        self,
        prompt: str,
        img1_path: str,
        img2_path: str,
        valid_keywords: list = None,
    ) -> dict:
        if not Path(img1_path).exists() or not Path(img2_path).exists():
            self.last_error = f"image not found: {img1_path} | {img2_path}"
            return {"label": "unknown", "reason": ""}

        try:
            self.last_error = ""
            stream = ollama.chat(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": prompt,
                    "images": [img1_path, img2_path]
                }],
                stream=True
            )

            full_output = ""
            print("\n--- AI分析中 ---\n")

            for chunk in stream:
                content = chunk.get("message", {}).get("content", "")
                if content:
                    print(content, end="", flush=True)
                    full_output += content

            print("\n\n--- AI分析结束 ---\n")

            result = self._extract_result(full_output, valid_keywords)
            reason = self._extract_reason(full_output, valid_keywords)
            if result == "unknown":
                print("模型结果不可判定，返回 unknown 交由上层回退")
                return {"label": "unknown", "reason": reason}

            return {"label": result, "reason": reason}
        except Exception as e:
            self.last_error = str(e)
            print(f"调用异常: {e}")
            print("请检查 Ollama 服务是否启动，以及模型名是否已拉取。")
            return {"label": "unknown", "reason": ""}

    def check_head_with_reason(
        self,
        head1_path: str,
        head2_path: str,
        low_similarity_fallback_label: str = "normal",
        crop_status: Optional[Dict[str, Any]] = None,
    ) -> dict:
        return self._call_head_model_with_reason(
            self._build_head_prompt(low_similarity_fallback_label, crop_status=crop_status),
            head1_path,
            head2_path,
        )

    def check_tail_with_reason(
        self,
        tail1_path: str,
        tail2_path: str,
        low_similarity_fallback_label: str = "normal",
        crop_status: Optional[Dict[str, Any]] = None,
    ) -> dict:
        return self._call_tail_model_with_reason(
            self._build_tail_prompt(low_similarity_fallback_label, crop_status=crop_status),
            tail1_path,
            tail2_path,
        )


if __name__ == "__main__":
    checker = VehicleCheck()

    img1 = r"D:\\project\\data_chuli\\demo\\demo\\Siamese-pytorch-master\\exports\\export_20260315_150806\\fake_plate\\20260312_104459_ac742494_fake_plate\\vehicle1.jpg"
    img2 = r"D:\\project\\data_chuli\\demo\\demo\\Siamese-pytorch-master\\exports\\export_20260315_150806\\fake_plate\\20260312_104459_ac742494_fake_plate\\vehicle2.jpg"

    p1 = Path(img1)
    p2 = Path(img2)
    print(f"model: {checker.model_name}")
    print(f"img1 exists: {p1.exists()} -> {p1.resolve() if p1.exists() else img1}")
    print(f"img2 exists: {p2.exists()} -> {p2.resolve() if p2.exists() else img2}")

    result = checker.check_head(img1, img2)
    print(f"\nfinal result: {result}")
    if checker.last_error:
        print(f"last_error: {checker.last_error}")
