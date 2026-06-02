import json

import ollama
from pathlib import Path


class VehicleCheck:
    """
    使用视觉模型对车头/主视角车尾裁切图做二次复核（fake_plate 或 change_trailer / normal）。
    """

    HEAD_VALID_LABELS = ["fake_plate", "normal"]

    def __init__(self, model_name="gemma4:latest"):
        self.model_name = model_name
        self.last_error = ""
        self.last_raw_output = ""

    def _build_head_prompt(self, low_similarity_fallback_label: str = "normal"):
        fallback_conclusion_text = (
            "车头相似度低于阈值，判定为套牌"
            if low_similarity_fallback_label == "fake_plate"
            else "车头相似度大于阈值，判定为正常"
        )
        return (
            "你现在只比较两张车头裁切图，只判断：fake_plate 或 normal。\n\n"
            "同位比对原则（最高优先级，必须优先遵守）：\n"
            "1. 只允许比较两张图中「同一固定部位、同一朝向面」上的长期标识或稳定结构；禁止把图1某一面上的文字与图2另一面上的文字直接对比。\n"
            "2. 若无法确认两图看到的是同一部位、同一朝向面（例如一张是导流罩正面、另一张是导流罩侧面），则该文字差异无效，不得单独作为 fake_plate 依据，应改比格栅、车标、保险杠、后视镜总成等硬结构，或判 normal。\n"
            "3. 引用任何文字/标识差异前，必须先完成部位对齐；未对齐则视为无效证据。\n"
            "4. 禁止把「都在导流罩上」当成「同部位」；导流罩至少分两个子区域，只能同子区域互比：\n"
            "   - deflector_front_center：导流罩正前/朝镜头一侧的中央品牌条。\n"
            "   - deflector_side_model：导流罩侧面型号条。\n"
            "6. 引擎盖固定标识、车门固定编号区、遮阳板文字区同样适用同位比对：左门对左门、右门对右门、引擎盖中央对引擎盖中央；不能拿一张图的侧面字去对另一张图的正面字。\n\n"
            "总原则：\n"
            "7. 只允许依据车头稳定结构和稳定标识做判断，不能只凭颜色深浅、亮暗变化、反光、阴影、局部发白、过曝、污渍、积灰、泥渍、轻微角度变化、开灯状态、打码状态或拍摄条件下结论。\n"
            "8. 建筑物遮挡、背光、玻璃反光、车头强反光、雨刷阴影、车外物体投影、夜间灯光、单边开灯、局部过曝、玻璃上的暗带，都默认属于成像或光照干扰，不是 fake_plate 依据。\n"
            "9. 判定依据只能从车体本身寻找。车内驾驶室物品、人物姿态、摆件、纸巾盒、挂饰、瓶子、包、座椅套、遮阳帘，以及单边车牌打码或未打码，均不能作为车头结构差异依据。\n"
            f"10. 如果其中一张图里没有清晰可见的车头主体，而主要是过磅自助机、建筑物、背景招牌、地磅设备、路面设施或其他非车辆对象，这本身就属于输入图片质量太差的情况；不要把这些场景设备与另一张图中的车辆做 fake_plate 比较，而要明确说明“输入图片质量太差，AI无法判断，{fallback_conclusion_text}”，并在最后输出 {low_similarity_fallback_label}。\n\n"
            "11. 先做资格审查，再做细节比对：如果关键区域本身不可稳定比较，例如主体过小、严重模糊、强反光、过曝、遮挡、只能看到局部疑似字符或只能靠猜测，不要继续展开长篇分析，而要尽快按输入图片质量太差处理。\n"
            "12. 不要围绕同一处不可靠区域反复比较，不要为了得出 fake_plate 而放宽证据标准；只要关键证据不可靠，就应停止深挖并快速收束到最终结论。\n\n"
            "高优先级观察项：\n"
            "13. 重点比较这些稳定特征：车头整体造型、格栅整体结构、大灯外轮廓、保险杠主体结构、后视镜总成主体形状与分色、车标结构、引擎盖整体造型与固定标识、车门固定文字或编号区域、导流罩长期文字区域（须先完成同位对齐）。\n"
            "14. 以下内容不应被当作普通装饰忽略：引擎盖中央或前脸固定标识、稳定品牌字样、固定翼形标、导流罩同子区域长期喷涂文字、车门固定编号区域、后视镜总成的主体造型与配色分区。这些若在同部位对齐且清晰可见、内容或结构明显不同，可以作为 fake_plate 依据。\n"
            "15. 例如引擎盖上的固定标识或字样、导流罩同子区域长期喷涂文字、后视镜外壳主体配色和总成造型，如果不是临时贴纸、反光或偶发遮挡造成，而是清晰稳定且部位对齐的差异，应视为有效差异，而不是普通装饰。\n\n"
            "低优先级或排除项：\n"
            "16. 细小装饰块、局部颜色块、后加装小配件，默认不作为车头主体结构差异依据，不能仅凭这些普通装饰细节判为 fake_plate。\n"
            "17. 车头区域如果出现橙色或黄色编号牌、危险品或货物标识牌、纯数字编号块，这类内容通常是运载货物或运输类别标识，不属于车辆身份标识；即使两图数字不同，也不能单独作为 fake_plate 依据。\n"
            "18. 程序可能会对真正车牌区域打上黑色矩形框，这只是预处理结果，不属于车辆本体结构或稳定标识；不能把黑色矩形框、黑块大小差异、黑块有无，当成 fake_plate 依据。\n"
            "19. 格栅条幅细节、局部纹理、表面光泽、亮面暗面变化，容易受阳光、反光、污渍和曝光影响；如果差异主要表现为条幅亮暗、表面反光或金属光泽强弱不同，不能直接作为结构差异依据。\n"
            "20. 跨面文字差异（如导流罩正面对侧面、左门文字对右门文字）一律无效，不得写入 fake_plate 理由。\n\n"
            "特殊判读规则：\n"
            "21. 车标要比较轮廓、外框、内部图案和安装位置，不要只看发光颜色、反光强弱或是否发白。\n"
            "22. 大灯要比较总轮廓、外框形状、安装位置和整体布局；如果一张图开灯、另一张不开灯，不能把被灯光遮住的内部灯组细节差异当成 fake_plate 依据。\n"
            "23. 导流罩、引擎盖顶部遮阳板、车头文字区域、喷涂标识区域，只有在两张图「同一子区域」都清晰可见、部位对齐、没有被强反光、过曝、发白、眩光、污渍或阴影遮盖时，才可以依据文字内容是否明显不同来判断 fake_plate。\n"
            "24. 对纯数字编号块、橙色编号牌、黑色车牌打码框，要优先判断它们是不是货物标识或预处理黑块，而不是车辆稳定标识；这类区域默认降权，除非你能明确确认它不是货物编号牌也不是打码框。\n"
            "25. 后视镜不能只看反光亮暗，但要比较后视镜总成主体形状、外壳分色、安装方式和支架关系；如果这些差异清晰稳定可见，属于有效差异。\n\n"
            "强光照射、色号漂移与“一侧有字一侧无字”专节（高优先级防误判）：\n"
            "26. 过磅现场常见正午直射、傍晚斜射、单侧强逆光：会导致车头大面积过曝发白、导流罩顶部文字被亮斑淹没、车门固定编号区反光看不清、前围贴纸/遮阳板小字在一张图清晰另一张图完全不可见——这些都属于成像差异，不是车辆本体没有该标识。\n"
            "27. 只要任意一侧导流罩/遮阳板/车门文字区存在强反光、过曝、发白、深阴影、字符被亮斑盖住或只能勉强辨认，就必须把该文字证据视为不可靠，不能写成“稳定标识差异”。\n"
            "28. 车门编号区在眩光、背光、侧角、离远近下经常单侧不可读；只有两侧车门对应区域都足够清晰、部位对齐，且能确认另一侧确实没有任何喷涂/贴纸编号（不是被光照洗掉）时，才可把车门文字差异作为依据。\n"
            "29. 车身主色（红/橙/深红/黄橙/银灰）在早晚巨大光线差异下会显著漂移：傍晚暖光可把红色拍成橙黄，正午强光可把深灰拍成亮银，阴影下可把红色拍成暗褐。应优先比较格栅造型、车标结构、驾驶室轮廓、保险杠形态、后视镜总成等不受单次曝光左右的硬结构。若硬结构一致，即使色号差异明显，也应判 normal，并在 reason 中写明可能为强光/时段导致的表观色偏。\n\n"
            "顶棚阴影与顶边灯：\n"
            "- 一侧因顶棚、雨棚或建筑物遮挡导致车顶/导流罩上沿深阴影，另一侧同区域明亮可见灯、字或装饰时，禁止写「图1无、图2有」。\n"
            "- 顶黑边/示廓灯/工作小圆灯（子区域=deflector_top_lamp_strip，勿用 deflector_front_center）：仅当两侧都能看见灯座或透镜外形（不是仅有光斑或无光斑）才可比较；否则写「单侧不可读，成像差异」并判 normal。\n\n"
            "挡风玻璃与雨刷：\n"
            "- 雨刷臂、雨刷片的位置与角度为临时状态，不得作为 fake_plate 依据，不得描述为「标识缺失」。\n"
            "- 玻璃上文字/标识须两侧清晰且确认为车体固定喷涂或贴附才可比较；无法确认是否为固定标识时判 normal。\n\n"

            "思考顺序：\n"
            "32. 先确认两张图里是否都存在清晰可比较的车头主体；如果有一张图主要拍到的是设备、场景或非车辆对象，不要拿这些非车辆对象做结构差异依据，而应按图片质量太差处理。\n"
            "33. 分别列出每张图清晰可见的长期标识及其部位标签（含导流罩子区域 deflector_front_center / deflector_side_model）。\n"
            "34. 求两图「同部位、同朝向面」的交集；若文字相关交集为空或无法确认对齐，则文字证据作废，不得仅凭文字判 fake_plate。\n"
            "35. 先判断是否存在单侧强光/过曝/深阴影导致的导流罩、车门、前围文字不可比；若有，直接把这些文字差异归入光照干扰，改比格栅、车标、驾驶室硬结构。\n"
            "36. 再判断车身红橙/亮暗差异是否可用时段光照解释；若能，不得单独据此判 fake_plate。\n"
            "37. 只有在同子区域文字双侧都清晰、部位对齐、且（硬结构也存在明确稳定差异，或同子区域品牌/厂家级文字明显冲突）时，才可判 fake_plate。\n"
            "38. 最后明确说明你依据的是稳定结构差异、同位对齐后的稳定标识差异，还是把跨面文字、导流罩正侧混比、车门编号、色号差异、贴纸可见性差异判定为无效证据或光照/反光/曝光干扰。\n\n"
            "分类规则：\n"
            "- fake_plate：车头存在清晰、稳定、不能用光照、反光、过曝、污渍、遮挡、投影、开灯状态、车内物品或打码状态解释的主体结构差异；或存在同部位对齐、双侧清晰可读且内容明显冲突的固定文字/品牌字样（含导流罩同子区域长期喷涂文字）、引擎盖固定标识、后视镜差异。\n"
            "- normal：车头结构与标识整体一致，或者看不出清晰稳定的异常差异；差异主要来自光照、反光、过曝、污渍、遮挡、投影、拍摄条件、开灯状态、车内物品、货物编号牌、打码框、普通装饰细节、打码状态不一致，或文字差异来自部位不对齐/跨面混比。\n\n"
            "输出要求：\n"
            "请按以下 JSON 格式输出，且只能输出一个 JSON 对象，不要输出额外解释：\n"
            "{\n"
            '  "label": "fake_plate 或 normal",\n'
            f'  "reason": "一句到两句中文说明；图片质量太差或关键证据不可靠时写输入图片质量太差、AI无法判断、{fallback_conclusion_text}，并将 label 设为 {low_similarity_fallback_label}；'
            "提到文字/标识差异时必须附带子区域、图1、图2、同部位对齐字段；"
            "同部位对齐=否或任一侧不可读时 label 必须为 normal；"
            'reason 只用中文描述，禁止出现 fake_plate、normal 等英文 label 词，可写套牌、正常、不能据此判套牌、不作为套牌依据"\n'
            "}\n"
        )

    def _build_tail_prompt(self):
        """主视角/车头方向下的车尾裁切图（tail1/tail2），非正后方尾部视角。"""
        return (
            "软限制：\n"
            "0. 先做资格审查；主体过小、过糊、强反光、过曝、深阴影遮挡、只能猜测时，尽快判 normal。\n"
            "0.1 不要为了 change_trailer 放宽标准；理由最多1到2句。\n"
            "0.2 你只判断 change_trailer 或 normal。\n\n"
            "视角（最高优先级）：\n"
            "输入是主视角/车头方向下的车尾裁切图，常见为挂车侧后段、侧挡板、轮轴、挡泥板、底盘侧挂附件。\n"
            "禁止因本视角看不到的部位猜测换挂；不得把货物、篷布、堆料轮廓当作侧挡板或车体结构。\n\n"
            "总原则：\n"
            "1. 只允许依据稳定物理结构，禁止单凭颜色深浅、明暗、反光、阴影、路灯/现场灯、夜间与白天差异判 change_trailer。\n"
            "2. 积灰、泥污、褪色、顶棚阴影、车底深阴影导致的表观色偏与附件“看不见”，默认成像差异。\n"
            "3. 车斗内货物形状/颜色不参与判断；临时篷布/苫盖差异不算车体结构换挂。\n"
            "4. 只有至少1项下方「可比对特征」存在清晰、稳定、具体差异，且不能用光照/角度/货物解释时，才可 change_trailer；否则 normal。\n\n"
            "可比对特征（仅比较本视角通常可见项，按优先级）：\n"
            "- 侧后段轮廓与侧挡板/栏板：有无栏板、高低栏、竖筋/波纹/镂空笼式、平板 vs 栏板。\n"
            "- 轴数与可见轮组布局（遮挡时写无法确认，不得猜测）。\n"
            "- 挡泥板形态、侧面反光条/下护栏分段。\n"
            "- 前挡板或侧后交界结构（若可见）。\n"
            "- 侧挂水箱/储物箱/工具箱：须两侧对应位置成像都足够清晰，且能确认无支架占位，才可判差异；一侧清晰另一侧仅阴影/沾灰/远景看不见 → normal。\n\n"
            "侧挂附件可见性（易误判）：\n"
            "浅灰箱体在一张图清晰、另一张为车底暗区，且栏板型制、轴数布局整体一致时，优先判同一附件成像差异，输出 normal。\n\n"
            "思考顺序：\n"
            "1. 先排除光照、阴影、货物、篷布、角度带来的假差异。\n"
            "2. 再比侧挡板/栏板型制、轴数、挡泥板、侧挂附件。\n"
            "3. 只有明确稳定结构差异才 output change_trailer。\n\n"
            "分类规则：\n"
            "- change_trailer：上述可比对特征中存在清晰稳定差异，且非光照/脏污/货物造成。\n"
            "- normal：可比对特征一致，或证据不足、或差异可归因光照/阴影/货物/篷布/成像。\n\n"
            "输出要求：\n"
            "1. 分两段：前1到2句写依据（点明栏型/轴数等，禁止空泛“栏板式一致”）；最后一行单独写 change_trailer 或 normal。\n"
            "2. 最后一行整行只能是 change_trailer 或 normal，无标点无中文。\n"
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
        return self._call_model(
            self._build_tail_prompt(),
            tail1_path,
            tail2_path,
            valid_keywords=["change_trailer", "normal"]
        )

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
        low_similarity_fallback_label: str = "normal"
    ) -> dict:
        return self._call_head_model_with_reason(
            self._build_head_prompt(low_similarity_fallback_label),
            head1_path,
            head2_path,
        )

    def check_tail_with_reason(self, tail1_path: str, tail2_path: str) -> dict:
        return self._call_model_with_reason(
            self._build_tail_prompt(),
            tail1_path,
            tail2_path,
            valid_keywords=["change_trailer", "normal"]
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
