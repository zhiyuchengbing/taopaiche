import matplotlib.pyplot as plt
import numpy as np
import onnxruntime
import torch
import cv2
from PIL import Image
from nets.siamese import Siamese as siamese
from torch.autograd import Variable
from onnxruntime.datasets import get_example
from PIL import Image
import time

from utils.utils_aug import center_crop, resize, crop

start = time.perf_counter()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def letterbox_image(image, size, letterbox_image):
    w, h = size
    iw, ih = image.size
    if letterbox_image:
        '''resize image with unchanged aspect ratio using padding'''
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)

        image = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        if h == w:
            new_image = resize(image, h)
        else:
            new_image = resize(image, [h ,w])
        new_image = center_crop(new_image, [h ,w])
    return new_image

def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image
def preprocess_input(x):
    x /= 255.0
    return x

def onnx_runtime(image_1, image_2):
    image_1_ = cvtColor(image_1)
    image_2_ = cvtColor(image_2)

    # ---------------------------------------------------#
    #   对输入图像进行不失真的resize
    # ---------------------------------------------------#

    image_1 = letterbox_image(image_1_, [105, 105], False)
    image_2 = letterbox_image(image_2_, [105, 105], False)

    # ---------------------------------------------------------#
    #   归一化+添加上batch_size维度
    # ---------------------------------------------------------#
    photo_1 = preprocess_input(np.array(image_1, np.float32))
    photo_2 = preprocess_input(np.array(image_2, np.float32))
    photo_1 = torch.from_numpy(np.expand_dims(np.transpose(photo_1, (2, 0, 1)), 0)).type(torch.FloatTensor).to(device)
    photo_2 = torch.from_numpy(np.expand_dims(np.transpose(photo_2, (2, 0, 1)), 0)).type(torch.FloatTensor).to(device)

    dummy_input = [photo_1, photo_2]

    example_model = get_example("D:\code\Vehicle_Re-identification\Siamese-pytorch-master\Siamese.onnx")
    session = onnxruntime.InferenceSession(example_model)
    # get the name of the first input of the model
    input_name0 = session.get_inputs()[0].name
    input_name1 = session.get_inputs()[1].name
    # print('onnx Input Name:', input_name)
    output = session.run([], {input_name0: dummy_input[0].data.cpu().numpy(), input_name1: dummy_input[1].data.cpu().numpy()})
    output = torch.nn.Sigmoid()(torch.Tensor(np.array(output)))
    #output = torch.nn.Sigmoid()(output)
    score = output[0].tolist()[0][0]

    plt.subplot(1, 2, 1)
    plt.imshow(np.array(image_1_))

    plt.subplot(1, 2, 2)
    plt.imshow(np.array(image_2_))
    plt.text(-12, -12, 'Similarity:%.3f' %score, ha='center', va='bottom', fontsize=11)
    plt.savefig("./my_test.jpeg", dpi=300)
    # time.sleep(4)
    # plt.show()
    #
    print('score:', score)

    if score > 0.35:
        print("为同一辆车")
    else:
        print("疑似为套牌车")



if __name__ == "__main__":

    start = time.perf_counter()

    image1 = Image.open(r"img/20230505043842_2.jpeg")
    image2 = Image.open(r"img/20230505043909_2.jpeg")

    image1 = crop(image1, 1440, 0, 1440, 2560)
    image2 = crop(image2, 1440, 0, 1440, 2560)

    # image1.show()
    # image2.show()

    onnx_runtime(image1, image2)

    end = time.perf_counter()
    print(f'Execution time: {end - start:.2f} seconds')