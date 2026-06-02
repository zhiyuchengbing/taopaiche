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
            "你是一名车辆尾部复核员，需要比较两张原始图中“中央车辆”尾部是否属于同一辆挂车。\n"
            "只输出 JSON；reason 最多 1-2 句话，不要长篇推理。\n\n"
            "【任务边界】\n"
            "1. 只看中央挂车/半挂车尾部。\n"
            "2. 忽略其他车辆、路面、背景、天气、时间、阴影、反光、灯光等无关干扰。\n"
            "3. 货物形状/颜色不参与判断，但也不得把货物或临时篷布当成挂车本体结构。\n\n"
            "【硬约束（最高优先）】\n"
            "H1. 先做可比对性：若任一侧未拍到挂车尾部（只见牵引车头/驾驶室侧面等），pair_comparable=否，label 必须 无法判断，并说明需回退主视角车尾AI。\n"
            "H2. 号牌/放大号清晰一致即正常：若两侧挂车号牌或放大号都清晰完整且关键位一致（省份简称、字母、数字、“挂”字），直接判 正常；plate_or_number_consistency=一致，structure_consistency=未检验，并停止后续结构比对。\n"
            "H3. 换挂成立条件：label=换挂 仅当 pair_comparable=是 且（plate_or_number_consistency=不一致 或 structure_consistency=不一致）。\n"
            "H4. 编号可用性对称规则：仅当两侧挂车号牌/放大号都清晰完整可读时，才可使用编号一致/不一致定案；任一侧不可读、缺失、眩光、过曝或仅见局部字符时，plate_or_number_consistency 必须填 无法确认，并放弃编号比较转结构比对。\n"
            "H5. 编号来源白名单：编号仅可来自挂车本体合法区域（号牌安装区、尾部放大号规范区域、车架正式编号区）；背景指示屏/道闸屏/建筑牌编号一律不是挂车身份编号。\n"
            "H6. 禁止仅凭颜色、单侧不可靠编号、背景编号、货物编号判换挂。\n"
            "H7. pair_comparable=是 但 plate 与 structure 均为 无法确认 时，label 必须 无法判断。\n\n"
            "【主流程（5步，按序执行）】\n"
            "Step0 可比对性审查：\n"
            "- 填写 img1_trailer_rear_visible、img2_trailer_rear_visible（是/否）；两侧均为是时 pair_comparable=是，否则为否。\n"
            "- pair_comparable=否 时，禁止判换挂，禁止用颜色定案。\n\n"
            "Step1 身份编号可靠性与一致性：\n"
            "- 可作为挂车身份依据的信息：挂车号牌、尾部放大号、车架号/车身正式编号。\n"
            "- 这些编号必须来自挂车本体（尾部号牌区、尾门/尾板放大号区、车架正式喷涂区）。\n"
            "- 危险品码、货物码、背景电子屏/道闸/指示牌编号均不是挂车身份编号，不能据此判换挂。\n"
            "- 编号不可靠情形（过小、过远、过斜、遮挡、过暗、过曝、反光、模糊、仅部分字符、只能猜测）一律记为 plate_or_number_consistency=无法确认，不得据此判不一致。\n"
            "- 典型眩光场景（单侧清晰、单侧亮斑淹没）属于编号不可靠，必须转结构比对。\n\n"
            "- 单侧编号禁止定换挂：仅一侧可读到挂车号牌/放大号，另一侧不可见或不可靠时，必须放弃编号比较，着重比较侧挡板、前/后挡板、顶棚/顶架结构。\n\n"
            "Step2 Tier-A（后开口 + 侧围，核心结构，编号不可靠时的主判断依据）：\n"
            "- 任一项结构明确不一致即可判换挂（structure_consistency=不一致）。\n"
            "- 禁止用光照、积灰、货物、篷布解释 Tier-A 的硬冲突。\n"
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
            "Step3 颜色交叉校验（仅辅助，不可单独定案）：\n"
            "  分别记录每张图挂车尾部的 body_hue（红/橙/黄/蓝/绿/白/灰/黑褐/不可辨）与 appearance_note（光照深暗/积灰/顶棚阴影/反光发白/无异常）。\n"
            "  - 禁止用牵引车头色相与挂车尾部色相比较。\n"
            "  - 仅 body_hue 不同且 Tier-A 全部一致 → 视为光照或脏污，不能单独换挂。\n"
            "  - body_hue 不同且 Tier-A 任一项不一致 → 换挂，reason 写明色相与门型/栏型均不同。\n"
            "  - 禁止在门型、栏高、镂空型已冲突时，仍写“阴影或顶棚导致色变”。\n\n"
            "Step4 Tier-B（尾灯/反光条/轴数/保险杠/挡泥板/号牌架/侧挂附件）：\n"
            "  尾灯总成外形(方灯/圆灯组合等)、尾部横反光条有无、轴数/可见轮组、下护栏保险杠形态、挡泥板、号牌架形态、侧挂附件。\n"
            "  Tier-B 不能压过 Tier-A：门型或侧围已冲突时，不得用“三轴、尾灯位置类似”判正常。\n"
            "  只有确认两侧为同一挂车本体且对应位置确实无同类安装位时，才可因 Tier-B 差异判换挂；不能仅凭一侧看不见就下结论。\n\n"
            "Step5 结论：\n"
            "  - Tier-A 任一项不一致 → 换挂，structure_consistency 填“不一致”。\n"
            "  - Tier-A 一致且 Tier-B 明确不一致（双侧清晰） → 换挂。\n"
            "  - Tier-A 与 Tier-B 均一致 → 正常，structure_consistency 填“一致”。\n"
            "  - 关键 Tier-A 项无法确认且无结构冲突疑点：仅在号牌不可靠且结构也无法比对时，输出无法判断并回退主视角车尾图。\n\n"
            "【特殊防误判（短版）】\n"
            "1. 货物误判防护：无固定门板边界、中缝、侧梁立柱等证据，不得把货物/篷布轮廓当尾门或侧围。\n"
            "2. 号牌眩光防护：单侧清晰、单侧眩光/过曝/字符缺失时，编号证据作废，不得据此判不一致。\n"
            "3. 侧挂附件防护：仅一侧看见水箱/储物箱而另一侧看不清时，不得强判换挂；需能明确排除“距离/阴影/积灰遮挡”后才可作为不一致证据。\n"
            "4. 号牌一致时不得输出无法判断（结构可写未检验）。\n\n"
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
