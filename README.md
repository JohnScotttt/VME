# Video-based Multiple Exposure(VME)

[EN](README.md) | [中文](README_zh.md)

## Introduction

VME is a tool for processing video into multi-exposure track video, so called because its core technology is based on multiple exposures of video.

The realization is shown below:

![demo](demo.gif)

The tool does not need to have a better shooting equipment as a support in the early stage, you can use the video shot by cell phone or even use the video recorded screen, do not need to fix the camera view, in ordinary scenes only need a video to achieve the above effect of video multiple exposure. The tool supports custom selection of the time range, number of multiple exposures and output type, which has a high degree of freedom, small limitations, the characteristics of a wide range of application scenarios. Next there are plans to develop the UI interface, the interaction is more convenient and intuitive.

## Usage

#### 1. Installation

Local environment:

```
python==3.9.18
opencv_python==4.5.5.64
paddlepaddle_gpu==2.6.0.post120
paddleseg==2.8.0
cuda==12.0
```

Recommended environment:

```
python>=3.8
opencv_python>=4.5.5
paddlepaddle_gpu>=2.0.2
paddleseg>=2.5
cuda>=10.0
```

Due to the high computational cost of model, it is recommended for GPU version PaddlePaddle.

#### 2. Run

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

Demo parameters:

```
python core.py --input_video demo.mp4 --device gpu --output_dir output --capture_type time --capture_range 1 3 --watermark_numbers 2
```

Note that when there is a watermark in the screen, be sure to use the **watermark_numbers** parameter to set the number of watermarks, when the **watermark_numbers** parameter is greater than 0, a pop-up window will appear, select any two opposite corners of a watermark in the screen to frame the watermark.
