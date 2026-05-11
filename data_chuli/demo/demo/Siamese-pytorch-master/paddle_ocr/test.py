# 使用 PaddlePaddle 进行推理
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)
result = ocr.predict(
    r"C:\\\\Users\\\\ADMINI~1\\\\AppData\\\\Local\\\\Temp\\\\ocr_head1__yg7lap1.jpg"
)
for res in result:
    res.print()
    res.save_to_img(r"D:\project\data_chuli\demo\demo\Siamese-pytorch-master\paddle_ocr\output")
    res.save_to_json(r"D:\project\data_chuli\demo\demo\Siamese-pytorch-master\paddle_ocr\output")
