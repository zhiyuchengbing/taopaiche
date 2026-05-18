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
            "你是一名车辆尾部复核员，需要比较两张原始图片中“中央车辆”的尾部是否属于同一辆挂车。\n\n"
            "任务范围：\n"
            "1. 只关注两张图中央位置的那一辆挂车/半挂车。\n"
            "2. 只分析这辆车的尾部区域。\n"
            "3. 忽略其他车辆、路面、背景、天气、时间、阴影、反光、灯光、货物和无关干扰。\n\n"
            "判定优先级必须严格遵守：\n"
            "最高优先级：先检查两张图是否都包含足够的中央车辆尾部有效信息。\n"
            "- 只有当两张图都能看到中央车辆的明确尾部区域，且至少具备“真正可用于识别挂车身份的编号信息”或可比对的尾部结构特征时，才允许继续做“正常/换挂”判断。\n"
            "- 如果任意一张图没有拍到中央车辆尾部，或尾部区域过小、过糊、过曝、被遮挡，导致无法形成有效尾部比对，必须直接输出“无法判断”。\n"
            "- 遇到“无法判断”时，不要勉强输出“正常”或“换挂”；原因中要明确说明尾部视角证据不足，需回退主视角裁切车尾图继续判断。\n\n"
            "第一优先级：先比对“真正的挂车身份编号”。\n"
            "- 真正可作为挂车身份依据的信息包括：挂车号牌、尾部放大号、车架号/车身编号、正式喷涂的挂车识别编号。\n"
            "- 中国挂车号牌通常由“省份简称 + 大写英文字母 + 数字”组成，例如“鲁A1234挂”“粤B5678挂”；这类信息属于强身份信息。\n"
            "- 只有当两张图中的强身份信息都清晰可见，并且能够明确确认“一致”或“明显不一致”时，才可以直接据此判定“正常”或“换挂”。\n"
            "- 如果挂车号牌或放大号区域过小、距离过远、拍摄角度过斜、被尾板/车身/货物遮挡、夜间过暗、局部过曝、反光严重、画面模糊，导致只能勉强辨认出部分字符或疑似字符，则该编号证据一律视为“不可靠”，不能直接作为“换挂”依据。\n"
            "- 如果一张图能看到较清晰编号，另一张图的编号区域却很小、被遮挡或只能猜测字符，也不能直接用“看起来不同”判定换挂，而是要放弃编号比较，转入结构特征比对。\n"
            "- 对于只看到末尾几位、只看到局部字符、只能猜测是 Z9285/Z7409 这类不完整或不稳定识别结果，必须视为“无法可靠确认编号”，不能直接判定“换挂”。\n"
            "- 如果强身份信息被遮挡、缺失、看不清，或者无法确认是不是正式挂车身份编号，不要直接判定“换挂”，而是进入结构特征比对。\n\n"
            "重要排除规则：以下内容不是挂车身份编号，不能单独作为“换挂”依据。\n"
            "- 危险品运输标识代码、介质编码、联合国编号、货物编号。\n"
            "- 货物/罐体/箱体上的两行或多行数字代码，例如“60 2874”“33 1114”这类危险品标识代码。\n"
            "- 危险品菱形标牌、燃/腐/爆等类别标识、限速标志、载重标识、公司广告、罐体宣传字样。\n"
            "- 货物品牌、介质名称、运输提示语、警示牌文字。\n"
            "- 这些信息即使不同，也不能直接判定为“换挂”；同一挂车允许运输不同货物，因此可能出现不同货物标识代码。\n\n"
            "第二优先级：只有在强身份信息无法确认且不能直接下结论时，再比较挂车尾部结构特征。\n"
            "重点关注以下稳定结构特征：\n"
            "- 罐体/车厢的整体类型与轮廓，例如罐式、平板、栏板、厢式、后门结构。\n"
            "- 尾部门开合方式。\n"
            "- 栏杆样式、立柱分布、边框结构。\n"
            "- 尾灯数量、位置、布局。\n"
            "- 后保险杠/防护栏、挡泥板、反光条、号牌安装区域、连接结构。\n"
            "- 轴数、轮组布局、尾部踏板/爬梯/阀门箱/附件布局等稳定特征。\n"
            "- 其他稳定且不易受光照影响的尾部结构特征。\n"
            "- 判断结构时优先看轮廓、分段、开孔、立柱、横梁、连接件、灯具布局等几何和构造特征，不要把表面颜色深浅直接当成结构差异。\n"
            "- 只要存在明显、稳定、非临时性的结构不一致，就判定为“换挂”。\n"
            "- 如果结构上看不出明显不一致，才判定为“正常”。\n\n"
            "注意事项：\n"
            "1. 必须只看中央车辆，不能拿边缘车辆或背景目标做判断。\n"
            "2. 颜色深浅、白天黑夜、雨雪雾、阴影、反光、模糊、视角轻微变化，不能单独作为正常或换挂依据。\n"
            "3. 车辆长期运行后的积灰、泥污、锈蚀、掉漆、补漆，会显著改变车厢、尾门、保险杠的表观颜色和明暗，不能仅凭“灰/深/浅”差异就判定为结构不同或换挂。\n"
            "4. 即使一张图看起来偏红、另一张偏灰，只要核心结构轮廓、开孔形式、立柱分布、尾灯和连接关系一致，就应优先认为这是表面状态或光照差异，而不是换挂依据。\n"
            "5. 货物多少、货物形状、货物颜色、遮挡物，不作为正常依据；如果它导致尾部有效信息不足，应输出“无法判断”，而不是直接判定“换挂”。\n"
            "6. 如果看到的是货物标识代码而不是挂车身份编号，必须在 reason 中明确说明“该编号属于货物/危险品标识，不属于挂车号牌或放大号，因此不能据此判换挂”。\n"
            "7. 如果编号证据不可靠，但结构特征仍可比较，应优先依据结构特征判断；只有当结构也无法可靠比较时，才输出“无法判断”，并回退主视角裁切车尾图继续判断。\n"
            "8. 允许输出第三种标签“无法判断”，仅在尾部视角有效信息不足、无法形成可靠尾部比对时使用。\n\n"
            "请按以下 JSON 格式输出，且只能输出一个 JSON 对象，不要输出额外解释：\n"
            "{\n"
            '  "label": "正常/换挂/无法判断",\n'
            '  "reason": "一句到两句中文说明；如果为无法判断，必须明确写出尾部视角信息不足，需要回退主视角裁切车尾图；如果编号区域过小、被遮挡、只能猜测字符，也要明确说明编号证据不可靠；如果颜色差异可能来自积灰、污渍、锈蚀或光照，也要明确说明颜色不能单独作为换挂依据；如果看到的是货物标识代码，也要明确说明它不是挂车身份编号",\n'
            '  "plate_or_number_consistency": "一致/不一致/无法确认",\n'
            '  "structure_consistency": "一致/不一致/未检验"\n'
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
        return "未检验"

    def _fallback_label_from_text(self, text: str) -> str:
        plain = str(text or "")
        plain_lower = plain.lower()
        if (
            "无法判断" in plain
            or "无法判定" in plain
            or "证据不足" in plain
            or "信息不足" in plain
            or "未拍到尾部" in plain
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

    def _call_model(self, img1_path: str, img2_path: str) -> dict:
        img1 = Path(img1_path)
        img2 = Path(img2_path)
        if not img1.exists() or not img2.exists():
            self.last_error = f"image not found: {img1_path} | {img2_path}"
            return {
                "label": "未知",
                "reason": "",
                "plate_or_number_consistency": "无法确认",
                "structure_consistency": "未检验",
            }

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

            if label == "未知":
                label = "无法判断"
                if not reason:
                    reason = "尾部视角信息不足或模型未能稳定输出标准结果，需要回退主视角裁切车尾图继续判断。"
                if plate_or_number_consistency == "一致":
                    structure_consistency = "未检验"

            return {
                "label": label,
                "reason": reason,
                "plate_or_number_consistency": plate_or_number_consistency,
                "structure_consistency": structure_consistency,
            }
        except Exception as e:
            self.last_error = str(e)
            print(f"调用异常: {e}")
            print("请检查 Ollama 服务是否已启动，以及模型名称是否可用。")
            return {
                "label": "未知",
                "reason": "",
                "plate_or_number_consistency": "无法确认",
                "structure_consistency": "未检验",
            }

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
