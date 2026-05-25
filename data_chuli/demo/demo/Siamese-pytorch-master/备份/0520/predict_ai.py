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



    # def _build_head_prompt(self):
    #     return (
    #         "你现在只比较两张车头裁切图，只判断：fake_plate 或 normal。\n\n"
    #         "最高优先级规则：\n"
    #         "1. 车灯亮灭、亮度变化、反光、阴影、雨雾、灰尘覆盖、泥渍、轻微角度变化，不能单独作为 fake_plate 依据。\n"
    #         "2. 建筑物遮挡、背光、局部发亮、偏色、发白、阳光过强、车头强反光、局部过曝等都属于光照干扰，不能直接作为套牌依据。\n"
    #         "3. 必须优先看车头稳定结构和稳定标识，而不是只看颜色深浅、光泽强弱或局部是否发亮。\n"
    #         "4. 重点关注：车头整体造型、格栅整体结构、大灯外轮廓、保险杠主体结构、后视镜主体形状、车标结构、引擎盖整体造型、车头文字区域、车门文字区域、导流罩文字区域是否一致。\n"
    #         "5. 普通装饰细节、细小装饰块、后加装的小配件，默认不作为车头主体结构差异依据；\n"
    #         "6. 引擎盖中央或前脸固定标识、稳定品牌字样、固定翼形标、导流罩长期喷涂文字、车门固定编号区域、后视镜总成的主体造型与配色分区。这些往往属于车辆长期稳定存在的识别特征，若清晰可见且内容/结构明显不同，可以作为 fake_plate 依据。\n"
    #         "7. 尤其注意：引擎盖上的固定标识或字样（例如 WRC 之类的稳定标记）、导流罩是否有长期喷涂文字、后视镜外壳主体颜色分区与造型，如果不是临时贴纸或反光造成，而是清晰、稳定、反复出现的差异，应视为车头稳定标识或主体部件差异，而不是简单装饰。\n"
    #         "8. 针对车标这一特征，要优先比较轮廓、外框、内部图案、安装位置等结构特征；车标发光、不发光、发蓝光、发白光、反光强弱不同，通常属于灯光或成像差异，不能仅凭车标颜色差异判为 fake_plate。\n"
    #         "9. 针对车灯这一特征，要优先比较大灯总轮廓、外框形状、安装位置和整体布局；如果一张图车灯开启、另一张图车灯关闭或不发光，灯光会严重掩盖大灯内部灯组细节，此时不能把内部灯组细节差异作为 fake_plate 依据。\n"
    #         "10. 针对文字和喷涂标识，要先判断该区域是否被强反光、过曝、发白、眩光、阴影或污渍遮盖；如果一张图文字清晰、另一张图该区域因强光或反光看不清，不能直接把“有字 vs 看不见字”判为 fake_plate。\n"
    #         "11. 后视镜不能只看反光亮暗，但要比较后视镜总成主体形状、外壳分色、安装方式、支架关系。如果后视镜主体配色和总成结构清晰可见且明显不同，这属于有效差异，不应被归为普通光照干扰。\n"
    #         "12. 不要把挡风玻璃上的阴影、反光带、雨刷阴影、车外物体投影、阳光照在玻璃上形成的暗带，误认为遮光板、额外结构件或后加装部件；这类玻璃表面明暗变化默认视为光照或投影干扰。\n"
    #         "13. 车内驾驶室物体是否存在、纸巾盒颜色、摆件、挂饰、仪表台物品、座椅套、遮阳帘、瓶子、包、人物姿态或车内物品摆放，均不能作为车头结构差异判断依据。\n"
    #         "14. 格栅条幅细节、局部纹理、表面光泽度、亮面/暗面变化，容易受阳光、反光、污渍和曝光影响；如果差异主要表现为条幅亮暗、表面反光、金属光泽强弱不同，不能直接作为结构差异依据。\n"
    #         "15. 特别注意预处理差异：如果一张图的车牌区域被黑色打码，另一张图未被打码或仍能看到车牌，这通常是预处理或识别失败造成的，不属于车头主体结构差异，不能仅凭“黑块 vs 可见车牌”判为 fake_plate。\n"
    #         "16. 如果看不出明确的稳定结构差异或稳定标识差异，默认判为 normal。\n\n"
    #         "请按下面顺序思考：\n"
    #         "第一步，先排除光照、阳光反射、局部过曝、发白、阴影、投影、玻璃反光、污渍、轻微角度变化、单边开灯造成的遮挡，以及单边打码造成的假差异。\n"
    #         "第二步，再逐项比较车头整体造型、格栅整体结构、大灯外轮廓、保险杠主体结构、后视镜主体形状与分色、车标结构、引擎盖整体造型与固定标识、车头文字区域、车门文字区域、导流罩文字区域。\n"
    #         "第三步，明确说明你看到的是稳定结构差异、稳定标识差异，还是仅仅是光照反光、玻璃投影、车内物品或普通装饰细节带来的非结构差异。\n\n"
    #         "分类规则：\n"
    #         "- fake_plate：车头存在清晰、稳定、不能用光照、反光、过曝、污渍、遮挡、投影、开灯状态、车内物品或打码状态解释的主体结构差异，或存在清晰可见且内容明显不同的固定文字、品牌字样、导流罩字样、引擎盖固定标识、后视镜总成差异。\n"
    #         "- normal：车头结构与标识整体一致，或者看不出清晰稳定的异常差异；差异主要来自光照、反光、过曝、污渍、遮挡、投影、拍摄条件、开灯状态、车内物品、装饰细节或打码状态不一致。\n\n"
    #         "输出要求：\n"
    #         "1. 先用1句话简要说明你依据了哪些稳定特征，不要笼统只说“结构一致”。\n"
    #         "2. 如果判 normal，要明确说明哪些差异被你认定为光照/反光/过曝/投影/污渍/开灯/车内物品/装饰细节/打码干扰。\n"
    #         "3. 如果涉及车标，要说明你比较的是车标结构而不是发光颜色；如果涉及大灯，要说明你比较的是外轮廓或布局，而不是被灯光遮盖的内部细节；如果涉及文字，要说明该文字区域是否清晰可见，不能把“看不见”直接当成“没有”。\n"
    #         "4. 不要把普通装饰细节、玻璃阴影或驾驶室内物品当成结构差异依据；但也不要把清晰稳定的引擎盖固定标识、导流罩长期字样、后视镜总成差异误降级为普通装饰。\n"
    #         "5. 最后一行必须只输出 fake_plate 或 normal"
    #     )

    def _build_head_prompt(self, low_similarity_fallback_label: str = "normal"):
        fallback_conclusion_text = (
            "车头相似度低于阈值，判定为套牌"
            if low_similarity_fallback_label == "fake_plate"
            else "车头相似度大于阈值，判定为正常"
        )
        return (
            "你现在只比较两张车头裁切图，只判断：fake_plate 或 normal。\n\n"
            "总原则：\n"
            "1. 只允许依据车头稳定结构和稳定标识做判断，不能只凭颜色深浅、亮暗变化、反光、阴影、局部发白、过曝、污渍、积灰、泥渍、轻微角度变化、开灯状态、打码状态或拍摄条件下结论。\n"
            "2. 建筑物遮挡、背光、玻璃反光、车头强反光、雨刷阴影、车外物体投影、夜间灯光、单边开灯、局部过曝、玻璃上的暗带，都默认属于成像或光照干扰，不是 fake_plate 依据。\n"
            "3. 判定依据只能从车体本身寻找。车内驾驶室物品、人物姿态、摆件、纸巾盒、挂饰、瓶子、包、座椅套、遮阳帘，以及单边车牌打码或未打码，均不能作为车头结构差异依据。\n"
            f"4. 如果其中一张图里没有清晰可见的车头主体，而主要是过磅自助机、建筑物、背景招牌、地磅设备、路面设施或其他非车辆对象，这本身就属于输入图片质量太差的情况；不要把这些场景设备与另一张图中的车辆做 fake_plate 比较，而要明确说明“输入图片质量太差，AI无法判断，{fallback_conclusion_text}”，并在最后输出 {low_similarity_fallback_label}。\n\n"
            "高优先级观察项：\n"
            "5. 重点比较这些稳定特征：车头整体造型、格栅整体结构、大灯外轮廓、保险杠主体结构、后视镜总成主体形状与分色、车标结构、引擎盖整体造型与固定标识、车头文字区域、车门固定文字或编号区域、导流罩长期文字区域。\n"
            "6. 以下内容不应被当作普通装饰忽略：引擎盖中央或前脸固定标识、稳定品牌字样、固定翼形标、导流罩长期喷涂文字、车门固定编号区域、后视镜总成的主体造型与配色分区。这些若清晰可见且内容或结构明显不同，可以作为 fake_plate 依据。\n"
            "7. 例如引擎盖上的固定标识或字样、导流罩是否有长期喷涂文字、后视镜外壳主体配色和总成造型，如果不是临时贴纸、反光或偶发遮挡造成，而是清晰稳定的差异，应视为有效差异，而不是普通装饰。\n\n"
            "低优先级或排除项：\n"
            "8. 细小装饰块、局部颜色块、后加装小配件，默认不作为车头主体结构差异依据，不能仅凭这些普通装饰细节判为 fake_plate。\n"
            "9. 车头区域如果出现橙色或黄色编号牌、危险品或货物标识牌、纯数字编号块，这类内容通常是运载货物或运输类别标识，不属于车辆身份标识；即使两图数字不同，也不能单独作为 fake_plate 依据。\n"
            "10. 程序可能会对真正车牌区域打上黑色矩形框，这只是预处理结果，不属于车辆本体结构或稳定标识；不能把黑色矩形框、黑块大小差异、黑块有无，当成 fake_plate 依据。\n"
            "11. 格栅条幅细节、局部纹理、表面光泽、亮面暗面变化，容易受阳光、反光、污渍和曝光影响；如果差异主要表现为条幅亮暗、表面反光或金属光泽强弱不同，不能直接作为结构差异依据。\n\n"
            "特殊判读规则：\n"
            "12. 车标要比较轮廓、外框、内部图案和安装位置，不要只看发光颜色、反光强弱或是否发白。\n"
            "13. 大灯要比较总轮廓、外框形状、安装位置和整体布局；如果一张图开灯、另一张不开灯，不能把被灯光遮住的内部灯组细节差异当成 fake_plate 依据。\n"
            "14. 导流罩、引擎盖顶部遮阳板、车头文字区域、喷涂标识区域，只有在两张图该区域都清晰可见、没有被强反光、过曝、发白、眩光、污渍或阴影遮盖时，才可以依据文字内容是否明显不同来判断 fake_plate。\n"
            "15. 对纯数字编号块、橙色编号牌、黑色车牌打码框，要优先判断它们是不是货物标识或预处理黑块，而不是车辆稳定标识；这类区域默认降权，除非你能明确确认它不是货物编号牌也不是打码框。\n"
            "16. 后视镜不能只看反光亮暗，但要比较后视镜总成主体形状、外壳分色、安装方式和支架关系；如果这些差异清晰稳定可见，属于有效差异。\n\n"
            "思考顺序：\n"
            "17. 先确认两张图里是否都存在清晰可比较的车头主体；如果有一张图主要拍到的是设备、场景或非车辆对象，不要拿这些非车辆对象做结构差异依据，而应按图片质量太差处理。\n"
            "18. 再排除光照、反光、过曝、阴影、玻璃投影、污渍、积灰、轻微角度变化、开灯状态、货物编号牌和打码框造成的假差异。\n"
            "19. 再逐项比较车头整体造型、格栅、大灯外轮廓、保险杠、后视镜总成、车标、引擎盖固定标识、车头、车门、导流罩稳定文字区域。\n"
            "20. 最后明确说明你依据的是稳定结构差异、稳定标识差异，还是把某些差异判定为光照、反光、投影、车内物品、货物编号牌、打码框、非车辆对象或普通装饰干扰。\n\n"
            "分类规则：\n"
            "- fake_plate：车头存在清晰、稳定、不能用光照、反光、过曝、污渍、遮挡、投影、开灯状态、车内物品或打码状态解释的主体结构差异，或存在清晰可见且内容明显不同的固定文字、品牌字样、导流罩字样、引擎盖固定标识、后视镜总成差异。\n"
            "- normal：车头结构与标识整体一致，或者看不出清晰稳定的异常差异；差异主要来自光照、反光、过曝、污渍、遮挡、投影、拍摄条件、开灯状态、车内物品、货物编号牌、打码框、普通装饰细节或打码状态不一致。\n\n"
            "输出要求：\n"
            "1. 先用1句话说明你依据了哪些稳定特征，不要只笼统说“结构一致”。\n"
            f"2. 如果其中一张图没有清晰车头主体，或主要是设备、建筑、背景牌子等非车辆对象，这就属于输入图片质量太差；直接说明“输入图片质量太差，AI无法判断，{fallback_conclusion_text}”，最后输出 {low_similarity_fallback_label}。\n"
            "3. 如果判 normal，要明确说明哪些差异被你认定为光照、反光、过曝、投影、污渍、开灯、车内物品、货物编号牌、打码框、非车辆对象或普通装饰细节干扰。\n"
            "4. 如果涉及车标，要说明你比较的是结构而不是发光颜色；如果涉及大灯，要说明你比较的是外轮廓或布局而不是被灯光遮住的内部细节；如果涉及文字，要说明该区域是否清晰可见，不能把“看不见”直接当成“没有”。\n"
            "5. 不要把普通装饰细节、玻璃阴影、驾驶室内物品、货物编号牌、车牌的打码框或非车辆场景设备当成结构差异依据，也不要把清晰稳定的引擎盖固定标识、导流罩长期字样、后视镜总成差异误降级为普通装饰。\n"
            "6. 最后一行必须只输出 fake_plate 或 normal"
        )

    def _build_tail_prompt(self):
        return (
            "你现在只比较两张车尾或挂车裁切图，只判断：change_trailer 或 normal。\n\n"
            "最高优先级规则：\n"
            "1. 只允许依据稳定的物理结构做判断，禁止根据颜色深浅、明暗变化、反光、阴影、灯光开关、灰尘、泥渍、褪色、轻微角度变化、局部遮挡、模糊和外来光源来判定 change_trailer。\n"
            "2. 车斗上的货物、货物多少、货物形状、货物颜色，不参与 change_trailer 或 normal 判断。\n"
            "3. 必须优先看挂车或车尾的物理结构，而不是看颜色、亮度或整体视觉感觉。\n"
            "4. 只有当你能明确看到至少1到2个稳定、具体、可重复描述的结构差异，并且这些差异不能被拍摄角度、光照、遮挡或货物干扰解释时，才能判为 change_trailer。\n"
            "5. 如果结构特征看不清、只看到一个模糊疑点、或者差异不够稳定，统一判为 normal。\n"
            "6. 如果两张图的车尾或挂车主体结构大体一致，或者你不能清楚确认结构不一致，必须判为 normal。\n\n"
            "重点检查这些结构：\n"
            "- 挂车整体轮廓、高度、宽度、长度比例\n"
            "- 栏板结构、立柱分布、边框形态\n"
            "- 尾灯数量、位置和布局\n"
            "- 挡泥板、反光条、下护栏布局\n"
            "- 车牌区域和安装位置\n"
            "- 箱体或挂车连接结构\n\n"
            "请按下面顺序思考：\n"
            "第一步，先排除光照、颜色、遮挡、货物和轻微角度变化带来的假差异。\n"
            "第二步，再逐项比较挂车轮廓、栏板立柱、尾灯、下护栏、挡泥板、车牌区、连接结构这些硬结构。\n"
            "第三步，只有在明确看到稳定结构差异时才输出 change_trailer，否则输出 normal。\n\n"
            "分类规则：\n"
            "- change_trailer：挂车或车尾主体存在清晰、稳定、明确的结构差异，且差异不止是视觉条件变化造成的。\n"
            "- normal：结构基本一致，或者证据不足以支持明确的结构不一致。\n\n"
            "输出要求：\n"
            "1. 先用1句话说明你具体依据了哪些结构特征；如果证据不足，也要明确说明是哪些结构看不出稳定差异。\n"
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
                print("模型结果不可判定，返回 unknown 交由上层回退")
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
        return self._call_model_with_reason(
            self._build_head_prompt(low_similarity_fallback_label),
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
