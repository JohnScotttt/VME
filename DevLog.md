# DeveloperLog

## 2024/03/04

目前先把core部分写完，再考虑UI的问题。人物抠像选用了PaddleSeg的PP-Mattingv2，目前正在部署Paddle Inference使其体量尽量减小，不排除要把整个PaddleSeg搬上来的可能。其他算法部分可能沿用原有的代码，不过也会有相应的跟进，不排除重写的可能。

已知的模块：

- [x] 人物抠像
- [x] 全局运动估计
- [x] 水印处理
- [x] 带Alpha通道图像叠加处理
- [x] 视频合成

鲁莽了，还是要把PaddleSeg全部搬上来的，我看着要用的模块一点点删吧。

## 2024/03/05

现在在考虑是使用流式处理还是full-load。full-load对代码维护性来说更强一点，但是实在是太占用存储空间，可能会生成好几份视频帧文件，如果视频又大又长就很麻烦。流式处理相对就没这么多文件，都是在内存中解决，当然对代码的要求也更高一些。如果要实现多平台使用，还是尽量选择流式处理。

## 2024/03/08

Matting写完了，我真的谢谢ppmatting，不过还好终于是重写完了，这个paddle模块又多，分的又散，删都删不好，只能把整个包扔工程里，把要改的拎出来重写一下了。

最后只有全局运动估计还没写，写完之后看一下有没有什么保护容灾没做的，做完就准备整合了。

## 2024/03/09

有点遗憾，Matting白写了，我仔细思考了一番，发现流处理是根本行不通的，单帧预测速度过慢让流处理不具备实现可能，还是需要保存后批量处理。

## 2024/03/10

四个字：近乎完美！！！我最喜欢的一版！