import ollama
from pathlib import Path


class VehicleCheck:
    """
    使用视觉模型判断两张车辆图片是否属于 fake_plate、change_trailer 或 normal。
    """

    def __init__(self, model_name="gemma4:latest"):
        self.model_name = model_name
        self.last_error = ""

    def _build_prompt(self):
        return (
            "你是一名车辆比对复核员，需要比较两张车辆图片，并且只根据稳定的物理结构做判断。\n\n"
            "必须遵守以下规则：\n"
            "1. 忽略环境因素：白天或夜晚、灯光开关、亮度变化、反光、阴影、雨雾、轻微角度变化，都不能单独作为不同车辆或换挂车的依据。\n"
            "2. 特别注意长期运行后的积灰、泥渍、褪色、脏污覆盖，这些会掩盖车头或挂车原本颜色，颜色深浅变化不能直接作为异常依据。\n"
            "3. 特别注意建筑物遮挡造成的局部变暗、背光、阴影覆盖，以及其他外来光源照射导致的局部发亮、偏色、发白，这些都属于光照干扰，不能直接判为换挂车或套牌车。\n"
            "4. 车厢上的货物、货物多少、货物形状、货物颜色，不参与 fake_plate、change_trailer、normal 的判断，只能忽略。\n"
            "5. 优先看稳定结构：车头造型、格栅、大灯轮廓、保险杠、后视镜、车尾结构、尾灯布局、挂车栏板结构、立柱分布、轮轴位置、挡泥板形状、标识位置等。\n"
            "6. 只有在结构性差异清晰可见时，才能判为异常；如果主要差异只是光照、灰尘、脏污、遮挡、外来光源或货物变化，必须判为 normal。\n\n"
            "分类规则：\n"
            "- fake_plate：两图车牌一致，但车头主体结构明显不是同一辆车。\n"
            "- change_trailer：牵引车车头可视为同一辆，但后方挂车或车尾结构明显不同。\n"
            "- normal：整体结构一致，差异主要来自光照、灰尘、反光、阴影、外来光源、遮挡或拍摄条件。\n\n"
            "请按下面顺序思考：\n"
            "第一步，先判断两图差异是否主要由光照、灰尘、遮挡、外来光源或拍摄角度造成。\n"
            "第二步，再检查是否存在明确的车头结构差异。\n"
            "第三步，再检查是否存在明确的挂车或车尾结构差异。\n"
            "第四步，不要考虑货物差异，只有结构差异明确时才输出异常分类，否则输出 normal。\n\n"
            "输出要求：\n"
            "1. 先用1到2句话说明你依据了哪些结构特征。\n"
            "2. 最后一行必须只输出一个标签，不要带任何前缀或解释。\n"
            "可选标签只有：fake_plate、change_trailer、normal"
        )

    def _build_head_prompt(self):
        return (
            "你现在只比较两张车头裁切图，只判断：fake_plate 或 normal。\n\n"
            "最高优先级规则：\n"
            "1. 车灯亮灭、亮度高低、反光、阴影、灰尘覆盖、泥渍、轻微角度变化，不能作为 fake_plate 依据。\n"
            "2. 白天黑夜变化、建筑物遮挡导致的局部发暗、外来光源造成的局部发亮或偏色，都属于干扰项。\n"
            "3. 必须优先看车头物理结构，而不是看颜色深浅。\n"
            "4. 如不能明确看出结构不同，默认判为 normal。\n\n"
            "重点检查这些结构：\n"
            "- 格栅形状与条幅数量\n"
            "- 大灯外轮廓和内部布局\n"
            "- 保险杠造型、雾灯区、进气口\n"
            "- 后视镜形状与安装位置\n"
            "- 车标或装饰件位置\n\n"
            "分类规则：\n"
            "- fake_plate：车头主体结构明显不同，不是光照、灰尘、遮挡或外来光源造成的。\n"
            "- normal：车头结构一致，或看不出明确结构差异。\n\n"
            "输出要求：\n"
            "1. 先用1句话简要说明依据。\n"
            "2. 最后一行必须只输出 fake_plate 或 normal"
        )

    # def _build_tail_prompt(self):
    #     return (
    #         "你现在只比较两张车尾或挂车裁切图，只判断：change_trailer 或 normal。\n\n"
    #         "最高优先级规则：\n"
    #         "1. 尾灯亮灭、亮度变化、反光、阴影、灰尘、泥渍、褪色、轻微角度变化，不能作为 change_trailer 依据。\n"
    #         "2. 白天黑夜光照变化、建筑物遮挡导致的局部变暗、背光、阴影覆盖，以及其他外来光源照射造成的局部发亮、偏色、发白，都会掩盖挂车原本颜色，不能仅因颜色变化判为换挂车。\n"
    #         "3. 车厢上的货物、货物多少、货物形状、货物颜色，不参与 change_trailer 或 normal 判断。\n"
    #         "4. 必须优先看挂车或车尾的物理结构，而不是看颜色明暗。\n"
    #         "5. 如不能明确看出结构不同，默认判为 normal。\n\n"
    #         "重点检查这些结构：\n"
    #         "- 挂车整体轮廓、高度、宽度、长度比例，明显高度不一致的就是换挂\n"
    #         "- 栏板结构、立柱分布、边框形式\n"
    #         "- 尾灯数量、位置和布局\n"
    #         "- 挡泥板、反光条、下护栏布局\n"
    #         "- 车牌区域和安装位置\n"
    #         "- 箱体或挂车连接结构\n\n"
    #         "分类规则：\n"
    #         "- change_trailer：挂车或车尾主体结构明显不同，不是光照、灰尘、遮挡、外来光源或货物差异造成的。\n"
    #         "- normal：车尾或挂车结构一致，或看不出明确结构差异。\n\n"
    #         "输出要求：\n"
    #         "1. 先用1句话简要说明依据。\n"
    #         "2. 最后一行必须只输出 change_trailer 或 normal"
    #     )

    def _build_tail_prompt(self):
        return (
            "你现在只比较两张车尾或挂车裁切图，只判断：change_trailer 或 normal。\n\n"
            "本次任务的总策略：宁可多报，不能漏报。后面还有兜底检验，因此只要存在稳定的结构性疑点，就优先判为 change_trailer，不要轻易判为 normal。\n\n"
            "最高优先级规则：\n"
            "1. 仍然要忽略干扰项：尾灯亮灭、亮度变化、反光、阴影、灰尘、泥渍、褪色、轻微角度变化、局部遮挡和外来光源，这些不能单独作为 change_trailer 依据。\n"
            "2. 车斗上的货物、货物多少、货物形状、货物颜色，不参与 change_trailer 或 normal 判断。\n"
            "3. 必须优先看挂车或车尾的物理结构，而不是看颜色明暗。\n"
            "4. 只要发现以下任意一种相对稳定的结构疑点，即使图像不算特别清晰，也优先判为 change_trailer：挂车整体轮廓、高度、宽度、长度比例不同，栏板结构不同，立柱分布不同，边框形态不同，尾灯数量或布局不同，挡泥板/反光条/下护栏布局不同，车牌区域或安装位置不同，箱体或连接结构不同。\n"
            "5. 只有在两张图的车尾或挂车主体结构基本一致，且所有差异都可以被合理解释为光照、灰尘、遮挡、拍摄角度或货物干扰时，才能判为 normal。\n"
            "6. 如果你在 normal 和 change_trailer 之间摇摆不定，或者认为有疑点但视觉条件一般，统一按疑似换挂车处理，输出 change_trailer。\n\n"
            "重点检查这些结构：\n"
            "- 挂车整体轮廓、高度、宽度、长度比例，明显高度或轮廓不一致的，优先视为换挂车\n"
            "- 栏板结构、立柱分布、边框形态\n"
            "- 尾灯数量、位置和布局\n"
            "- 挡泥板、反光条、下护栏布局\n"
            "- 车牌区域和安装位置\n"
            "- 箱体或挂车连接结构\n\n"
            "分类规则：\n"
            "- change_trailer：只要挂车或车尾主体存在明确结构差异，或者存在较强的结构疑点且不能被单纯归因于光照、灰尘、遮挡、外来光源或货物干扰，都判为 change_trailer。\n"
            "- normal：只有在结构看起来基本一致，且没有可靠的结构疑点时，才判为 normal。\n\n"
            "输出要求：\n"
            "1. 先用1句话简要说明依据。\n"
            "2. 最后一行必须只输出 change_trailer 或 normal"
        )

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
        for line in reversed(lines):
            normalized = line
            for prefix in normalized_prefixes:
                normalized = normalized.replace(prefix, "")
            normalized = normalized.strip()
            if normalized in valid_keyword_set:
                return normalized

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
                print("模型结果不可判定，默认返回 normal")
                return "normal"

            return result
        except Exception as e:
            self.last_error = str(e)
            print(f"调用异常: {e}")
            print("请检查 Ollama 服务是否启动，以及模型名是否已拉取。")
            return "unknown"

    def check_vehicle(self, img1_path: str, img2_path: str) -> str:
        return self._call_model(self._build_prompt(), img1_path, img2_path)

    def check_head(self, head1_path: str, head2_path: str) -> str:
        return self._call_model(
            self._build_head_prompt(),
            head1_path,
            head2_path,
            valid_keywords=["fake_plate", "normal"]
        )

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
                print("模型结果不可判定，默认返回 normal")
                return {"label": "normal", "reason": reason}

            return {"label": result, "reason": reason}
        except Exception as e:
            self.last_error = str(e)
            print(f"调用异常: {e}")
            print("请检查 Ollama 服务是否启动，以及模型名是否已拉取。")
            return {"label": "unknown", "reason": ""}

    def check_head_with_reason(self, head1_path: str, head2_path: str) -> dict:
        return self._call_model_with_reason(
            self._build_head_prompt(),
            head1_path,
            head2_path,
            valid_keywords=["fake_plate", "normal"]
        )

    def check_tail_with_reason(self, tail1_path: str, tail2_path: str) -> dict:
        return self._call_model_with_reason(
            self._build_tail_prompt(),
            tail1_path,
            tail2_path,
            valid_keywords=["change_trailer", "normal"]
        )

    def _build_diff_analysis_prompt(self, part_type: str) -> str:
        """构建细粒度差异分析 prompt。"""
        if part_type == "head":
            return (
                "你现在只做车头差异描述，不再做分类。\n\n"
                "任务：找出两张车头图最明显、最稳定的1到2个结构差异。\n"
                "忽略这些干扰：灯光亮灭、亮度、反光、阴影、灰尘、泥渍、轻微角度变化、建筑遮挡、外来光源。\n"
                "优先观察：车牌区域、大灯轮廓、格栅形状、保险杠、后视镜、车标位置。\n\n"
                "输出要求：\n"
                "1. 只输出一句中文短句。\n"
                "2. 不要解释原因，不要输出分类标签，不要加“差异：”“结果：”这类前缀。\n"
                "3. 句子长度尽量控制在30字以内。\n\n"
                "输出示例：\n"
                "左前大灯轮廓明显不同\n"
                "格栅条幅数量和造型不同\n"
                "车牌区域位置一致但保险杠结构不同"
            )

        return (
            "你现在只做车尾或挂车差异描述，不再做分类。\n\n"
            "任务：找出两张车尾或挂车图最明显、最稳定的1到2个结构差异。\n"
            "忽略这些干扰：尾灯亮灭、亮度、反光、阴影、灰尘、泥渍、褪色、轻微角度变化、建筑遮挡、外来光源。\n"
            "不要考虑车厢上的货物、货物多少、货物形状、货物颜色。\n"
            "优先观察：挂车整体轮廓、栏板结构、立柱分布、尾灯布局、挡泥板、反光条、下护栏、车牌区域、连接结构。\n\n"
            "输出要求：\n"
            "1. 只输出一句中文短句。\n"
            "2. 不要解释原因，不要输出分类标签，不要加“差异：”“结果：”这类前缀。\n"
            "3. 句子长度尽量控制在30字以内。\n\n"
            "输出示例：\n"
            "尾灯数量和布局明显不同\n"
            "挂车栏板结构和高度不同\n"
            "挡泥板与反光条位置不同"
        )

    def analyze_differences(self, img1_path: str, img2_path: str, part_type: str = "head") -> str:
        """
        细粒度差异分析：指出具体哪个部位不一致。
        Args:
            img1_path: 图片1路径
            img2_path: 图片2路径
            part_type: "head" 或 "tail"

        Returns:
            差异描述字符串
        """
        if not Path(img1_path).exists() or not Path(img2_path).exists():
            return "无法分析（图片文件不存在）"

        try:
            prompt = self._build_diff_analysis_prompt(part_type)
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
            print(f"\n--- AI差异分析 ({part_type}) ---\n")

            for chunk in stream:
                content = chunk.get("message", {}).get("content", "")
                if content:
                    print(content, end="", flush=True)
                    full_output += content

            print("\n\n--- AI差异分析结束 ---\n")

            desc = full_output.strip().strip('"\'')
            lines = [line.strip() for line in desc.split("\n") if line.strip()]
            if lines:
                desc = lines[0]

            for prefix in ["差异：", "差异:", "结果：", "结果:", "描述：", "描述:"]:
                if desc.startswith(prefix):
                    desc = desc[len(prefix):].strip()

            if len(desc) > 100:
                desc = desc[:100] + "..."

            return desc if desc else "差异分析未完成"
        except Exception as e:
            print(f"差异分析异常: {e}")
            return f"差异分析失败 ({str(e)[:50]})"


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
