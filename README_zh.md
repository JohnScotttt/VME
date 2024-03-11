# Video-based Multiple Exposure(VME)

[EN](README.md) | [中文](README_zh.md)

## 简介

VME是一种将视频处理成多重曝光轨迹视频的工具，因为其核心技术为基于视频的多重曝光所以称之为VME（Video-based Multiple Exposure）。

实现效果如下图：

![demo](demo.gif)

该工具不需要前期有较好的拍摄设备作为支撑，可以手机拍摄甚至录屏，不需要固定相机视角，在普通场景下只需要一个视频就可以实现上图效果的视频多重曝光。该工具支持自定义选取时间范围，自定义多重曝光数量，自定义输出类型，自由度高，限制小，应用场景广。接下来有计划开发UI界面，交互更方便、直观。

## 使用教程

#### 1. 环境安装

本地环境：

``` 
python==3.9.18
opencv_python==4.5.5.64
paddlepaddle_gpu==2.6.0.post120
paddleseg==2.8.0
cuda==12.0
```

推荐环境：

```
python>=3.8
opencv_python>=4.5.5
paddlepaddle_gpu>=2.0.2
paddleseg>=2.5
cuda>=10.0
```

由于图像抠图模型计算开销大，推荐在GPU版本的PaddlePaddle下使用。

#### 2. 运行

```
python core.py
	--input_video /path/to/your_video.mp4
	[--device {cpu,gpu}, default=cpu]
	--output_dir /path/to/output_dir
	[--output_type {video,image}, default=video]
	[--output_name name_of_file, default=demo]
	--capture_type {time,frame}
	--capture_range start_time(frame) end_time(frame)
	[--capture_numbers number_of_capture, default=5]
	[--watermark_numbers number_of_watermark, default=0]
```

demo参数：

```
python core.py --input_video demo.mp4 --output_dir output --capture_type time --capture_range 1 3 --watermark_numbers 2
```

注意，当画面中存在水印时，务必使用watermark_numbers参数设置水印数量，当watermark_numbers参数大于0时，会弹出一个窗口，选择画面中一个水印的任意两对角就可以框选水印。
