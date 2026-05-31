import re

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
            "1. 先用1句话说明你依据了哪些稳定特征，不要只笼统说“结构一致”。\n"
            f"2. 如果其中一张图没有清晰车头主体，或主要是设备、建筑、背景牌子等非车辆对象，这就属于输入图片质量太差；直接说明“输入图片质量太差，AI无法判断，{fallback_conclusion_text}”，最后输出 {low_similarity_fallback_label}。\n"
            f"3. 如果关键证据不可靠，例如主体模糊、反光过强、遮挡严重、只能猜字符或只能看到局部疑似差异，不要继续反复分析，要尽快说明“输入图片质量太差，AI无法判断，{fallback_conclusion_text}”，最后输出 {low_similarity_fallback_label}。\n"
            "4. 如果判 normal，要明确说明哪些差异被你认定为光照、反光、过曝、投影、污渍、开灯、车内物品、货物编号牌、打码框、非车辆对象、部位不对齐/跨面混比或普通装饰细节干扰。\n"
            "5. 输出自证（引用文字差异时强制）：reason 中必须写明子区域标签（deflector_front_center 或 deflector_side_model）、图1原文、图2原文、同部位对齐=是/否。\n"
            "5.1 若图1为 deflector_front_center、图2为 deflector_side_model（或 reason 同时含 FOTON DAIMLER 与 KX560 DONGFENG），则同部位对齐必须写「否」，最后一行只能输出 normal。\n"
            "5.2 禁止在跨面混比时写「同部位存在品牌冲突」；写了不同方位（右侧 vs 左上、正面 vs 侧面）即视为未对齐。\n"
            "6. 如果涉及车标，要说明你比较的是结构而不是发光颜色；如果涉及大灯，要说明你比较的是外轮廓或布局而不是被灯光遮住的内部细节；如果涉及导流罩/车门文字，必须说明子区域（如 deflector_front_center）及两侧是否都清晰可读，不能把“强光下看不见”写成“没有解放字样”，也不能把正面字与侧面字混比；若因光照导致色号橙红不一，应写明不能单凭颜色判套牌。\n"
            "7. 不要把普通装饰细节、玻璃阴影、驾驶室内物品、货物编号牌、车牌的打码框、跨面文字差异或非车辆场景设备当成结构差异依据；也不要把清晰稳定且同位对齐的引擎盖固定标识、导流罩同子区域长期字样、后视镜总成差异误降级为普通装饰。\n"
            "8. 理由最多使用1到2句话，不要输出长篇分步推理，不要重复比较同一处不可靠区域。\n"
            "9. 最后一行必须只输出 fake_plate 或 normal"
        )

    def _build_tail_prompt(self):
        return (
            "软限制行为规则：\n"
            "0. 先做资格审查，再做结构比较。如果关键尾部结构本身不可稳定比较，例如主体太小、严重模糊、强反光、过曝、遮挡或只能看到局部疑似差异，不要继续展开长篇分析，而应尽快收束到 normal。\n"
            "0.1 不要为了得出 change_trailer 而放宽证据标准，也不要围绕同一处不可靠区域反复比较。\n"
            "0.2 理由最多使用1到2句话，不要输出长篇分步推理。\n"
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
            "- 箱体或挂车连接结构\n"
            "- 侧挂水箱、储物箱、工具箱等底盘附件（须结合下方可见性规则）\n\n"
            "侧挂附件可见性规则（易误判）：\n"
            "- 车架侧下方常见浅灰/白色水箱、储物箱；远景、车底阴影、积灰泥污可使其在一张图中几乎不可见，另一张近景中却清晰，这通常是同一附件的成像差异，不是换挂。\n"
            "- 禁止仅凭“一侧有灰箱、另一侧对应位置看不出”就判 change_trailer；若无法排除沾灰/远近/阴影掩盖，应判 normal。\n"
            "- 仅当两侧成像都足够清晰，且能确认另一侧对应位置确实无安装支架/箱体占位时，才可把侧挂附件差异作为换挂依据。\n\n"
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
                "同位比对：只描述两张图中同一固定部位、同一朝向面上的差异；导流罩须区分正面品牌区与侧面型号区，禁止把正面字与侧面字混比。\n"
                "忽略这些干扰：灯光亮灭、亮度、反光、阴影、灰尘、泥渍、轻微角度变化、建筑遮挡、外来光源、跨面文字差异。\n"
                "优先观察：格栅形状、车标结构、大灯轮廓、保险杠、后视镜总成、引擎盖固定标识、导流罩同子区域文字。\n\n"
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
