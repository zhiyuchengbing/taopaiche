import json
import os
import time
from pathlib import Path

import ollama

# 本机 Ollama 直连：trust_env=False 避免 httpx 走系统代理(127.0.0.1:7897)导致 502，
# 也避免 httpx 的默认超时掐断流式输出
# 超时中断默认关闭 (AI_OLLAMA_TIMEOUT_S 默认 0=不设超时):
# 开启时 client connect/read 超时 + _iter_with_deadline 流式总时长上限,
# 超时抛 TimeoutError -> 上层返回 unknown -> 相似度兜底强行判换挂, 会绕过AI误判.
# 如需开启: 设 AI_OLLAMA_TIMEOUT_S=120 (秒).
_OLLAMA_TIMEOUT_S = float(os.environ.get("AI_OLLAMA_TIMEOUT_S", "0"))
_OLLAMA = ollama.Client(trust_env=False, timeout=(_OLLAMA_TIMEOUT_S or None))


def _iter_with_deadline(stream, timeout_s):
    """流式遍历加总时长上限, 防模型长时间生成拖住整个请求.

    timeout_s<=0 (默认) 时不做超时中断, 直接透传流.
    """
    if timeout_s and timeout_s > 0:
        deadline = time.monotonic() + timeout_s
        for chunk in stream:
            if time.monotonic() > deadline:
                raise TimeoutError(f"Ollama 响应超过 {timeout_s:.0f}s, 中断")
            yield chunk
    else:
        for chunk in stream:
            yield chunk


class TailVehicleCheck:
    """
    使用本地 Ollama 直接比较两张原图中中央车辆的尾部，
    判断是否为“正常”“换挂”或“无法判断”。
    """

    VALID_LABELS = ["正常", "换挂", "无法判断"]

    def __init__(self, model_name: str = "qwen3.5:9b"):
        self.model_name = model_name
        self.last_error = ""
        self.last_raw_output = ""

    def _build_tail_compare_prompt(self, char_hint: str = "") -> str:
        prompt = (
            "你是一名车辆尾部复核员，比较两张原始图中“中央车辆”挂车尾部是否同一辆。\n"
            "只输出 JSON；reason 最多 1-2 句，不要长篇推理。\n\n"
            "【任务边界】\n"
            "1. 只看中央挂车/半挂车尾部。\n"
            "2. 忽略其他车辆、路面、背景、天气、时间、阴影、反光、灯光等无关干扰。\n"
            "3. 货物形状/颜色不参与判断，但也不得把货物或临时篷布当成挂车本体结构。\n\n"
            "【可比对性（Step0）】\n"
            "C0. pair_comparable 只看挂车尾部车体是否可见：两侧尾部车体（尾门/栏板/侧围/轮轴等）都可见即=是。\n"
            "    号牌不可读、放大号不可读、反光/过曝盖住号牌——只影响“编号比对”，不影响可比对性。\n"
            "C1. 任一侧完全未见挂车尾部车体（只见牵引车头/驾驶室侧面等）→ pair_comparable=否，label=无法判断，reason 说明需回退主视角车尾AI。\n"
            "C2. 两侧尾部车体均可见 → pair_comparable=是；即使编号都不可读，也必须继续结构比对定案，禁止因编号不可读直接判无法判断。\n\n"
            "【编号一致性（优先，简短）】\n"
            "N1. 优先看图中车挂号/放大号的检测框（绿框=chegua 车挂号，橙框=fangdahao 放大号）：两侧同类编号在框内都清晰可读时，直接比较关键位（省字/字母/数字），"
            "关键位一致 → 正常（H2），明显不同 → 换挂（无需再比结构）。\n"
            "N2. 外部字符检测(专用OCR)给出 一致/不一致 时直接采信；本对进入此复核时外部检测多为 无法判断/作废，此时不代表无法判断，仍按 N1 自行读框内编号或转结构比对。\n"
            "N3. 编号定案要求双侧同类编号都清晰完整；任一侧不可读/缺失/跨类/仅见局部/眩光过曝 → 编号视为无法确认，转结构比对。\n"
            "N4. 编号仅可来自挂车本体合法区域（号牌区/尾门放大号区/车架编号区）；背景指示屏/道闸/建筑牌/货物码/危险品码都不是挂车身份编号。\n"
            "N5. 结构相似时禁止凭单侧字符碎片或猜测强读不同编号；编号不可靠则编号判定为无法确认，以结构定案。\n\n"
            "【结构比对（Tier-A，编号无法确认时的主依据）】\n"
            "S1. 后开口：有无尾门/门型/门高/门扇与中缝/固定顶或敞口；无尾门 vs 全高双扇、矮尾板 vs 全高竖门 → 一律不一致。\n"
            "S2. 侧围：有无侧挡板/栏高/实心镂空/侧栏喷涂；低栏实心 vs 仓栅镂空、实心 vs 全镂空笼 → 一律不一致。\n"
            "S3. 任一项结构明确不一致即可判换挂；但若编号一致（N1/N2 已定案正常），结构结论无效。\n"
            "S4. 光照/积灰/货物/篷布不得解释 Tier-A 硬冲突；任一侧夜间欠曝/顶棚阴影/积灰泛白/强反光 → 该侧色相不可信，不得单独作为换挂依据。\n"
            "S5. 几何优先：先数金属外框线、开口/窗洞数量、立柱/横梁布局，再判实心/镂空/组合；禁止仅凭洞内或板面黄/黑/亮/暗填色定结构型。\n"
            "S6. 同位置 N 个固定形状外框一致时，窗洞内亮/暗/阴影只是光照差，不得据此判实心 vs 镂空。\n\n"
            "【颜色与 Tier-B（仅辅助）】\n"
            "B1. 颜色不可单独定案：仅颜色不同且结构全一致 → 光照/脏污，正常；颜色不同且结构有冲突 → 换挂。\n"
            "B2. Tier-B（尾灯外形/反光条/轴数/保险杠/挡泥板/号牌架/侧挂附件）不能压过 Tier-A；仅两侧都清晰可见对应安装位时，差异才可作为换挂辅证。\n\n"
            "【防误判（短版）】\n"
            "1. 无固定门板边界/中缝/侧梁立柱等证据，不得把货物/篷布轮廓当尾门或侧围。\n"
            "2. 单侧编号清晰、单侧眩光/过曝/缺失 → 编号证据作废，不得据此判不一致。\n"
            "3. 放大号≠号牌不可混比（尾门喷涂大字 vs 号牌区）→ 跨类编号一律无效。\n"
            "4. 编号确凿不同（双侧同类编号清晰且关键位不同）→ 直接换挂，不必等结构证实。\n"
            "5. 禁止仅凭颜色、单侧不可靠编号、背景编号、货物编号判换挂。\n\n"
            "【结论（Step5）】\n"
            "D1. 双侧同类编号清晰且关键位一致 → 正常（H2），plate_or_number_consistency=一致，structure_consistency=未检验。\n"
            "D2. 双侧同类编号清晰且关键位不同 → 换挂，plate_or_number_consistency=不一致。\n"
            "D3. 编号无法确认时，Tier-A 任一项不一致 → 换挂，structure_consistency=不一致。\n"
            "D4. Tier-A 一致且 Tier-B 明确不一致（双侧清晰）→ 换挂；Tier-A 与 Tier-B 均一致 → 正常。\n"
            "D5. 只有两侧尾部车体都不可比（C1）或 编号与结构都确实无法判断且无任何不一致证据 时，才输出 无法判断并回退主视角车尾图。\n"
            "D6. 禁止因号牌不可读、反光、过曝等成像原因输出 无法判断——这些只导致放弃编号转结构，结构仍须给结论。\n\n"
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
        if char_hint:
            prompt += (
                "\n【外部字符检测辅助信息（专用OCR管线）：外部给出 一致/不一致 时优先采信；"
                "若为 作废/无法判断，则不代表本对不可判断，继续按 N1 自行读框内编号或转结构比对】\n"
                f"{char_hint}\n"
            )
        return prompt

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

        return self._apply_h2_plate_match_guard(
            label=label,
            reason=reason,
            img1_visible=img1_visible,
            img2_visible=img2_visible,
            pair_comparable=pair_comparable,
            plate_consistency=plate_consistency,
            structure_consistency=structure_consistency,
        )

    def _apply_h2_plate_match_guard(
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
        """H2 硬拦截：双侧号牌/放大号一致时强制正常，结构结论无效。"""
        h2_reason = "双侧挂车号牌/放大号关键位一致，按H2规则直接判定正常，结构比对结论无效"
        base_result = {
            "img1_trailer_rear_visible": img1_visible,
            "img2_trailer_rear_visible": img2_visible,
            "pair_comparable": pair_comparable if pair_comparable != "未知" else "是",
            "plate_or_number_consistency": plate_consistency,
            "structure_consistency": structure_consistency,
        }

        comparable = (
            pair_comparable != "否"
            and img1_visible != "否"
            and img2_visible != "否"
        )
        if not comparable or plate_consistency != "一致":
            return {
                "label": label,
                "reason": reason,
                **base_result,
            }

        if label != "正常":
            original_label = label
            original_reason = reason
            if original_reason:
                reason = f"{h2_reason}（原模型结论：{original_label}，{original_reason}）"
            else:
                reason = h2_reason
            print(
                f"[tail-ai] H2 guard adjusted label: {original_label!r} -> '正常' "
                f"(plate_or_number_consistency=一致)"
            )

        return {
            "label": "正常",
            "reason": reason or h2_reason,
            "img1_trailer_rear_visible": base_result["img1_trailer_rear_visible"],
            "img2_trailer_rear_visible": base_result["img2_trailer_rear_visible"],
            "pair_comparable": base_result["pair_comparable"],
            "plate_or_number_consistency": "一致",
            "structure_consistency": "未检验",
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

    def _call_model(self, img1_path: str, img2_path: str, char_hint: str = "") -> dict:
        img1 = Path(img1_path)
        img2 = Path(img2_path)
        if not img1.exists() or not img2.exists():
            self.last_error = f"image not found: {img1_path} | {img2_path}"
            return self._empty_tail_result()

        try:
            self.last_error = ""
            self.last_raw_output = ""
            stream = _OLLAMA.chat(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": self._build_tail_compare_prompt(char_hint),
                    "images": [str(img1), str(img2)],
                }],
                stream=True,
            )

            print("\n--- AI分析中 ---\n")
            for chunk in _iter_with_deadline(stream, _OLLAMA_TIMEOUT_S):
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

    def check_tail_on_original(self, img1_path: str, img2_path: str, char_hint: str = "") -> dict:
        return self._call_model(img1_path, img2_path, char_hint)


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
