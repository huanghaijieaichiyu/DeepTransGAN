#这是一个实现由INR图像生成RGB图像的神经网络，思路来源于知乎大佬的ai上色示例，链接: https://zhuanlan.zhihu.com/p/30493746
#大佬的训练思路是由RGB图像转为LAB图像，通过A，B通道的色彩信息作为标签，来对L通道的明度图像进行训练，最后融合训练后的色彩信息与明度，实现对图像上色
##这个思路恰好可以用来实现由红外图：INR ，来转化成普通色彩图像：RGB 的目的：本人研究方向为自动驾驶目标检测，此模块作为图像增强部分使用，实现自动驾驶汽车夜视识别功能
