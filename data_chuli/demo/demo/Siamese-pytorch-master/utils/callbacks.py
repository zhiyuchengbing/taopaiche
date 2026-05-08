import datetime
import os

import torch
import matplotlib
matplotlib.use('Agg')
import scipy.signal
from matplotlib import pyplot as plt

# 兼容性修复：处理numpy版本兼容性问题
try:
    from torch.utils.tensorboard import SummaryWriter
except AttributeError as e:
    if "bool8" in str(e):
        print("警告: 检测到numpy版本兼容性问题，正在尝试修复...")
        import numpy as np
        # 为旧版本tensorboard添加bool8别名
        if not hasattr(np, 'bool8'):
            np.bool8 = np.bool_
        from torch.utils.tensorboard import SummaryWriter
    else:
        raise e


class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        self.log_dir    = os.path.join(log_dir, "loss_" + str(time_str))
        self.losses     = []
        self.val_loss   = []
        self.accs       = []
        self.val_accs   = []
        
        # 确保目录创建成功，exist_ok=True避免并发问题
        os.makedirs(self.log_dir, exist_ok=True)
        # 转换为绝对路径，避免Windows路径问题
        self.log_dir    = os.path.abspath(self.log_dir)
        self.writer     = SummaryWriter(self.log_dir)
        try:
            dummy_input     = torch.randn(2, 2, 3, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except:
            pass

    def append_loss(self, epoch, loss, val_loss, acc=None, val_acc=None):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)

        self.losses.append(loss)
        self.val_loss.append(val_loss)

        if acc is not None and val_acc is not None:
            self.accs.append(acc)
            self.val_accs.append(val_acc)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")

        if acc is not None and val_acc is not None:
            with open(os.path.join(self.log_dir, "epoch_acc.txt"), 'a') as f:
                f.write(str(acc))
                f.write("\n")
            with open(os.path.join(self.log_dir, "epoch_val_acc.txt"), 'a') as f:
                f.write(str(val_acc))
                f.write("\n")

        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        if acc is not None and val_acc is not None:
            self.writer.add_scalar('acc', acc, epoch)
            self.writer.add_scalar('val_acc', val_acc, epoch)
        self.loss_plot()
        if self.accs and self.val_accs:
            self.acc_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")

    def acc_plot(self):
        iters = range(len(self.accs))

        plt.figure()
        plt.plot(iters, self.accs, 'blue', linewidth = 2, label='train acc')
        plt.plot(iters, self.val_accs, 'cyan', linewidth = 2, label='val acc')
        try:
            if len(self.accs) < 25:
                num = 5
            else:
                num = 15

            plt.plot(iters, scipy.signal.savgol_filter(self.accs, num, 3), 'navy', linestyle = '--', linewidth = 2, label='smooth train acc')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_accs, num, 3), '#1E90FF', linestyle = '--', linewidth = 2, label='smooth val acc')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc="lower right")

        plt.savefig(os.path.join(self.log_dir, "epoch_acc.png"))

        plt.cla()
        plt.close("all")
