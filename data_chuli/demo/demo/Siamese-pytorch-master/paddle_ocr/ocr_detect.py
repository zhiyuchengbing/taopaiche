import re

from paddleocr import PaddleOCR


class MaxBoxOCR:
    """
    提取 OCR 中最大检测框对应的文本，并提供容错比对。
    """

    MONITOR_OVERLAY_TEMPLATES = [
        "车头抓拍",
        "车型抓拍车头",
        "车型抓拍",
        "抓拍车头",
    ]

    def __init__(
        self,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False
    ):
        self.ocr = PaddleOCR(
            use_doc_orientation_classify=use_doc_orientation_classify,
            use_doc_unwarping=use_doc_unwarping,
            use_textline_orientation=use_textline_orientation
        )

    @staticmethod
    def normalize_text(text):
        if not text:
            return ""

        text = str(text).upper().strip()
        text = re.sub(r"[\s\-_./]+", "", text)
        text = re.sub(r"[^A-Z0-9]", "", text)
        return text

    @staticmethod
    def extract_digits(text):
        if not text:
            return ""
        return "".join(ch for ch in str(text) if ch.isdigit())

    @staticmethod
    def replace_similar_chars(text):
        if not text:
            return ""

        mapping = str.maketrans({
            "O": "0",
            "Q": "0",
            "D": "0",
            "I": "1",
            "L": "1",
            "Z": "2",
            "S": "5",
            "B": "8",
            "G": "6",
        })

        return str(text).translate(mapping)

    @staticmethod
    def extract_text_value(value):
        if isinstance(value, dict):
            return str(value.get("text") or "").strip()
        return str(value or "").strip()

    @staticmethod
    def normalize_chinese_text(text):
        if not text:
            return ""
        text = str(text).strip()
        text = re.sub(r"[\s\-_./]+", "", text)
        text = re.sub(r"[^\u4e00-\u9fffA-Z0-9]", "", text.upper())
        return text

    @classmethod
    def _is_monitor_overlay_text(cls, raw_text):
        raw = str(raw_text or "").strip()
        if not raw:
            return False

        clean = cls.normalize_chinese_text(raw)
        if not clean:
            return False

        for template in cls.MONITOR_OVERLAY_TEMPLATES:
            template_clean = cls.normalize_chinese_text(template)
            if clean == template_clean:
                return True
            # 裁剪后可能只剩角标的一部分，3 个字及以上的半截也过滤
            if len(clean) >= 3 and (clean in template_clean or template_clean in clean):
                return True

        has_date_markers = sum(marker in raw for marker in ("月", "日", "星期", ":"))
        has_many_digits = len(re.findall(r"\d", raw)) >= 4
        if has_date_markers >= 2 and has_many_digits:
            return True

        if re.search(r"\d{1,2}:\d{2}(:\d{2})?", raw):
            return True

        if re.search(r"\d{1,2}月\d{1,2}日", raw):
            return True

        if "星期" in raw:
            return True

        return False

    @classmethod
    def sanitize_ocr_text(cls, text):
        raw = str(text or "").strip()
        if not raw:
            return ""
        if cls._is_monitor_overlay_text(raw):
            return ""
        return raw

    @staticmethod
    def longest_common_contiguous_length(text1, text2):
        if not text1 or not text2:
            return 0
        max_len = 0
        len1 = len(text1)
        len2 = len(text2)
        dp = [0] * (len2 + 1)
        for i in range(1, len1 + 1):
            prev = 0
            for j in range(1, len2 + 1):
                temp = dp[j]
                if text1[i - 1] == text2[j - 1]:
                    dp[j] = prev + 1
                    if dp[j] > max_len:
                        max_len = dp[j]
                else:
                    dp[j] = 0
                prev = temp
        return max_len

    @staticmethod
    def has_any_common_char(text1, text2):
        if not text1 or not text2:
            return False
        return bool(set(text1) & set(text2))

    def compare_texts(self, text1, text2):
        raw1 = self.sanitize_ocr_text(self.extract_text_value(text1))
        raw2 = self.sanitize_ocr_text(self.extract_text_value(text2))
        clean1 = self.normalize_chinese_text(raw1)
        clean2 = self.normalize_chinese_text(raw2)
        longest_common = self.longest_common_contiguous_length(clean1, clean2)
        shortest_len = min(len(clean1), len(clean2)) if clean1 and clean2 else 0

        result = {
            "text1": raw1,
            "text2": raw2,
            "normalized_text1": clean1,
            "normalized_text2": clean2,
            "match": False,
            "strict_match": bool(raw1) and raw1 == raw2,
            "similarity": 0.0,
            "reason": "",
        }

        if raw1 and raw2 and raw1 == raw2:
            result["match"] = True
            result["similarity"] = 1.0
            result["reason"] = "raw_equal"
            return result

        if not clean1 or not clean2:
            result["reason"] = "empty_text"
            return result

        if clean1 == clean2:
            result["match"] = True
            result["similarity"] = 1.0
            result["reason"] = "normalized_equal"
            return result

        mapped1 = self.replace_similar_chars(self.normalize_text(raw1))
        mapped2 = self.replace_similar_chars(self.normalize_text(raw2))
        if mapped1 and mapped2 and mapped1 == mapped2:
            result["match"] = True
            result["similarity"] = 0.98
            result["reason"] = "confusable_chars_equal"
            return result

        digits1 = self.extract_digits(clean1)
        digits2 = self.extract_digits(clean2)
        if digits1 and digits2 and digits1 == digits2:
            result["match"] = True
            result["similarity"] = 0.95
            result["reason"] = "same_digits"
            return result

        if longest_common >= 2:
            result["match"] = True
            result["similarity"] = longest_common / max(len(clean1), len(clean2))
            result["reason"] = "common_two_char_substring"
            return result

        if shortest_len <= 2 and self.has_any_common_char(clean1, clean2):
            result["match"] = True
            result["similarity"] = 0.5
            result["reason"] = "short_text_one_char_match"
            return result

        result["similarity"] = (
            longest_common / max(len(clean1), len(clean2))
            if clean1 and clean2 else 0.0
        )
        result["reason"] = "text_mismatch"
        return result

    def get_max_text(self, img_path, min_score=0.6):
        """
        获取置信度 >= min_score 的最大检测框文字。
        """
        result = self.ocr.predict(img_path)

        max_area = 0
        max_text = ""
        max_score = 0.0

        for res in result:
            data = res.json["res"]

            texts = data["rec_texts"]
            polys = data["dt_polys"]
            scores = data["rec_scores"]

            for text, points, score in zip(texts, polys, scores):
                if score < min_score:
                    continue

                text = self.sanitize_ocr_text(text)
                if not text:
                    continue

                xs = [p[0] for p in points]
                ys = [p[1] for p in points]

                width = max(xs) - min(xs)
                height = max(ys) - min(ys)
                area = width * height

                if area > max_area:
                    max_area = area
                    max_text = text
                    max_score = score

        return {
            "text": max_text,
            "score": float(max_score),
            "area": float(max_area),
        }

    def compare_images(self, img1_path, img2_path, min_score=0.8):
        result = {
            "img1_path": str(img1_path),
            "img2_path": str(img2_path),
            "text1": "",
            "text2": "",
            "normalized_text1": "",
            "normalized_text2": "",
            "match": None,
            "strict_match": False,
            "similarity": 0.0,
            "reason": "",
            "error": None,
        }

        try:
            ocr1 = self.get_max_text(img1_path, min_score)
            ocr2 = self.get_max_text(img2_path, min_score)

            text1 = ocr1["text"]
            text2 = ocr2["text"]

            print("图片1最大框文字:", text1)
            print("图片2最大框文字:", text2)
            print("图片1置信度:", ocr1["score"])
            print("图片2置信度:", ocr2["score"])

            compare_result = self.compare_texts(text1, text2)

            result.update(compare_result)
            result["ocr1"] = ocr1
            result["ocr2"] = ocr2
            result["error"] = None
            return result
        except Exception as e:
            result["error"] = str(e)
            result["reason"] = "ocr_exception"
            result["match"] = None
            return result


if __name__ == "__main__":
    img1 = r"D:\project\data_chuli\demo\demo\Siamese-pytorch-master\exports\export_20260512_174610\fake_plate\20260512_174532_1e80450e_fake_plate\head1.jpg"
    img2 = r"D:\project\data_chuli\demo\demo\Siamese-pytorch-master\exports\export_20260512_174610\fake_plate\20260512_174532_1e80450e_fake_plate\head2.jpg"

    ocr_model = MaxBoxOCR()

    single_result = ocr_model.get_max_text(
        img2,
        min_score=0.4
    )

    print("单张图片最大OCR结果:")
    print(single_result)
