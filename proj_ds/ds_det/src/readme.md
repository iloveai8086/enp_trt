## build
编译plugin library
```shell
/usr/local/cuda/bin/nvcc -Xcompiler -fPIC -shared -o config/yolov5_decode.so ./yoloForward_nc.cu ./yoloPlugins.cpp ./nvdsparsebbox_Yolo.cpp -isystem /usr/include/x86_64-linux-gnu/ -L /usr/lib/x86_64-linux-gnu/ -I /opt/nvidia/deepstream/deepstream/sources/includes -lnvinfer 
```
把build好的yolov5 trt engine文件放到config目录下，并在yolov5的配置文件中填入其对应的名字, 如果使用docker请使用pytrt_build编译的engine，如果在jetson中，请使用tensorrt_cpp编译的engine。deepstream的配置文件信息在下面介绍。

## 配置
利用Deepstream-app，配置相应的参数实现yolov5推理和推流。
```
[application]
enable-perf-measurement=1
perf-measurement-interval-sec=5


[tiled-display]
enable=0
rows=1
columns=1
width=1280
height=720
gpu-id=0
nvbuf-memory-type=0

[source0]
enable=1
type=3
uri=file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4
num-sources=1
gpu-id=0
cudadec-memtype=0

[sink0]
enable=1
type=4
sync=0
gpu-id=0
codec=1
bitrate=1000
rtsp-port=9001
udp-port=5400
nvbuf-memory-type=0

[osd]
enable=1
gpu-id=0
border-width=5
border-color=0;1;0;1
text-size=15
text-color=1;1;1;1;
text-bg-color=0.3;0.3;0.3;1
font=Serif
show-clock=1
clock-x-offset=800
clock-y-offset=820
clock-text-size=12
clock-color=1;0;0;0
nvbuf-memory-type=0

[streammux]
gpu-id=0
live-source=0
batch-size=1
batched-push-timeout=40000
width=1920
height=1080
enable-padding=0
nvbuf-memory-type=0

[primary-gie]
enable=1
gpu-id=0
gie-unique-id=1
nvbuf-memory-type=0
config-file=config_infer_primary_yoloV5.txt

[tests]
file-loop=1
```
以下是每个部分的详细解释：

* [application]：启用性能测量，并设置性能测量时间间隔。
* [tiled-display]：指定平铺显示的设置，例如行数、列数、显示尺寸和GPU ID。
* [source0]：指定输入视频源为一个H.264编码的文件，并设置GPU ID和CUDA解码内存类型。
* [sink0]：指定输出视频流的设置，包括输出类型为RTSP流、编解码器类型为H.264、比特率为1000 Kbps、RTSP端口为9001、UDP端口为5400以及NVBUF内存类型。
* [osd]：启用On-Screen Display (OSD)，并设置边框宽度、边框颜色、文字大小、文字颜色、文字背景颜色、字体、时钟偏移量和时钟颜色。
* [streammux]：设置流多路复用器的GPU ID、批次大小、批处理推送超时、视频宽度和高度、是否启用填充以及NVBUF内存类型。
* [primary-gie]：启用一个基础的图像推理引擎（GIE），使用YOLOv5进行目标检测，并设置GPU ID、唯一标识符以及NVBUF内存类型。
* [tests]：循环播放文件。
其中如果需要在本地窗口显示请将sink0中的type改为2，如果需要rtsp推流则改成4，rtsp推流的链接为rtsp://127.0.0.1:9001/ds-test。可以通过vlc捕获并查看。推流的比特率以及编码格式可以在sink0中进行配置。如果需要查看bbox的绘制，需要在osd配置中将enable改为1，不绘制则改为0。需要注意显示和绘制都会占用计算资源，如果要查看峰值性能和FPS，则需要将sink0，osd中enable改为0。
在jetson中，需要将设备调为最高性能已查看峰值性能：
```shell
sudo nvpmodel -m 8
```
不同的型号的设备其峰值的模式不同，可以通过`sudo nvpmodel -q`查看当前的模式，并根据设备信息修改模式。

对应我们也需要对gie进行配置，配置文件为config_infer_primary_yoloV5.txt
```
[property]
gpu-id=0
net-scale-factor=0.0039215697906911373
model-color-format=0
onnx-file=yolov5s.onnx
model-engine-file=yolov5.engine
infer-dims=3;640;640
labelfile-path=labels.txt
batch-size=1
workspace-size=1024
network-mode=2
num-detected-classes=80
interval=0
gie-unique-id=1
process-mode=1
network-type=0
cluster-mode=2
maintain-aspect-ratio=1
parse-bbox-func-name=NvDsInferParseYolo
custom-lib-path=yolov5_decode.so

[class-attrs-all]
nms-iou-threshold=0.45
pre-cluster-threshold=0.25
topk=300
```
每个部分的详细解释：

* [property]：指定GIE的各种属性和参数，包括GPU ID、输入图像的缩放因子、模型颜色格式、ONNX文件路径、推理引擎文件路径、输入图像维度、标签文件路径、批次大小、工作空间大小、网络模式、检测类别数、GIE唯一标识符、处理模式、网络类型、集群模式、是否维持宽高比、解析边界框函数名以及自定义库路径。

* [class-attrs-all]：指定目标检测的属性和参数，包括非极大值抑制（NMS）的IoU阈值、预聚类阈值和每个类别的最大检测数量。

其中，这个GIE使用了YOLOv5模型进行目标检测，模型文件为yolov5s.onnx，推理引擎文件为yolov5.engine，使用INT8量化和最大批次大小为1。标签文件labels.txt中包含了80个目标类别。输入图像的尺寸为640x640，解析边界框函数为NvDsInferParseYolo，使用了一个自定义库文件yolov5_decode.so来解码推理输出。

### 运行
#### Docker 
```shell
sudo docker run --gpus all -v `pwd`:/app -p 9001:9001  --rm -it nvcr.io/nvidia/deepstream:6.1.1-devel bash
cp /app
deepstream-app -c config/deepstream_app_config.txt
```
#### Jetson
```shell
deepstream-app -c config/deepstream_app_config.txt
```
运行后，deepstream会自动进行模型构建并保存engine文件，如果engine文件已经存在，则会直接加载engine文件。需要注意，这里deepstream只能进行fp16和fp32的模型构建。int8模型构建时需要输入calib文件目录，但是ds并没有提供这个参数输入接口。因此如果需要使用int8 engine文件，需要通过pytrt_buil或者tensorrt_cpp中构建得到的模型文件来进行推理，将文件拷贝到config目录中，并将文件名放到对应的配置项。