/*
演示opencv + RMTP 推流
*/
#include <iostream>
#include <opencv2/opencv.hpp>
#include "streamer/streamer.hpp"

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <video_file> <bitrate>" << std::endl;
        return -1;
    }
    int bitrate = atoi(argv[2]);

    // 读取视频文件
    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened())
    {
        std::cerr << "Could not open video file " << argv[1] << std::endl;
        return -1;
    }
    // 获取画面尺寸
    cv::Size frameSize(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    // 获取帧率
    double fps = cap.get(cv::CAP_PROP_FPS);

    // 实例化推流器
    streamer::Streamer streamer;
    streamer::StreamerConfig streamer_config(frameSize.width, frameSize.height,
                                             frameSize.width, frameSize.height,
                                             fps, bitrate, "main", "rtmp://127.0.0.1:1935/live/mystream");
    streamer.init(streamer_config);

    cv::Mat frame;
    while (true)
    {
        // 读取视频帧
        cap >> frame;
        if (frame.empty())
        {
            break;
        }
        // 简单绘制文字
        cv::putText(frame, "Hello World!", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
        // 画面中心绘制一个矩形
        cv::rectangle(frame, cv::Point(frame.cols / 2 - 50, frame.rows / 2 - 50), cv::Point(frame.cols / 2 + 50, frame.rows / 2 + 50), cv::Scalar(0, 255, 0), 2);
        // 推流
        streamer.stream_frame(frame.data);
    }
    // 释放资源
    cap.release();
    return 0;
}
