#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "logger.h"
#include "common.h"
#include "buffers.h"
#include "utils/preprocess.h"
#include "utils/postprocess.h"
#include "utils/types.h"
#include "streamer/streamer.hpp"

#include <fstream>
#include "task/border_cross.h"
#include "task/gather.h"

// 用于测试
#define PRINT_TIME 1

// 加载模型文件
std::vector<unsigned char> load_engine_file(const std::string &file_name)
{
    std::vector<unsigned char> engine_data;
    std::ifstream engine_file(file_name, std::ios::binary);
    assert(engine_file.is_open() && "Unable to load engine file.");
    engine_file.seekg(0, engine_file.end);
    int length = engine_file.tellg();
    engine_data.resize(length);
    engine_file.seekg(0, engine_file.beg);
    engine_file.read(reinterpret_cast<char *>(engine_data.data()), length);
    return engine_data;
}

// 从文件中恢复多边形定点
void readPoints(std::string filename, Polygon &g_ploygon, int width, int height)
{
    std::ifstream file(filename);
    std::string str;
    while (std::getline(file, str))
    {
        std::stringstream ss(str);
        std::string x, y;
        std::getline(ss, x, ',');
        std::getline(ss, y, ',');

        // recover to original size
        x = std::to_string(std::stof(x) * width);
        y = std::to_string(std::stof(y) * height);

        g_ploygon.push_back({std::stoi(x), std::stoi(y)});
    }
}
void blender_overlay(int x, int y, int radius, cv::Mat &image, float alpha, int height, int width)
{
    // initial
    int rect_l = x - radius;
    int rect_t = y - radius;
    int rect_w = radius * 2;
    int rect_h = radius * 2;

    int point_x = radius;
    int point_y = radius;

    // check if out of range
    if (x + radius > width)
    {
        rect_w = radius + (width - x);
    }
    if (y + radius > height)
    {
        rect_h = radius + (height - y);
    }
    if (x - radius < 0)
    {
        rect_l = 0;
        rect_w = radius + x;
        point_x = x;
    }
    if (y - radius < 0)
    {
        rect_t = 0;
        rect_h = radius + y;
        point_y = y;
    }
    // get roi
    cv::Mat roi = image(cv::Rect(rect_l, rect_t, rect_w, rect_h));
    cv::Mat color;
    roi.copyTo(color);
    // draw circle
    cv::circle(color, cv::Point(point_x, point_y), radius, cv::Scalar(255, 0, 255), -1);
    // blend
    cv::addWeighted(color, alpha, roi, 1.0 - alpha, 0.0, roi);
}

int main(int argc, char **argv)
{

    if (argc < 7)
    {
        std::cerr << "用法: " << argv[0] << " <engine_file> <input_path_path> <preprocess_mode> <dist_threshold> <stream> <bitrate> " << std::endl;
        return -1;
    }

    auto engine_file = argv[1];                // 模型文件
    std::string input_video_path = argv[2];    // 输入视频文件
    auto mode = std::stoi(argv[3]);            // 预处理模式
    float dist_threshold = std::stof(argv[4]); // 距离阈值
    auto do_stream = std::stoi(argv[5]);       // 是否推流
    auto bitrate = std::stoi(argv[6]);         // 码率

    // ========= 1. 创建推理运行时runtime =========
    auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
    if (!runtime)
    {
        std::cout << "runtime create failed" << std::endl;
        return -1;
    }
    // ======== 2. 反序列化生成engine =========
    // 加载模型文件
    auto plan = load_engine_file(engine_file);
    // 反序列化生成engine
    auto mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan.data(), plan.size()));
    if (!mEngine)
    {
        return -1;
    }

    // ======== 3. 创建执行上下文context =========
    auto context = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        std::cout << "context create failed" << std::endl;
        return -1;
    }

    // ========== 4. 创建输入输出缓冲区 =========
    samplesCommon::BufferManager buffers(mEngine);

    // 如果input_video_path是rtsp，则读取rtsp流
    cv::VideoCapture cap;
    if (input_video_path == "rtsp")
    {
        auto rtsp = "rtsp://192.168.1.241:8556/live1.sdp";
        std::cout << "当前使用的是RTSP流" << std::endl;
        cap = cv::VideoCapture(rtsp, cv::CAP_FFMPEG);
    }
    else
    {
        std::cout << "当前使用的是视频文件" << std::endl;
        cap = cv::VideoCapture(input_video_path);
    }

    // 获取画面尺寸
    cv::Size frameSize(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    // 获取帧率
    double video_fps = cap.get(cv::CAP_PROP_FPS);
    std::cout << "width: " << frameSize.width << " height: " << frameSize.height << " fps: " << video_fps << std::endl;

    // 实例化推流器
    streamer::Streamer streamer;
    if (do_stream)
    {
        streamer::StreamerConfig streamer_config(frameSize.width, frameSize.height,
                                                 frameSize.width, frameSize.height,
                                                 video_fps, bitrate, "main", "rtmp://localhost/live/mystream");
        streamer.init(streamer_config);
    }
    // // 写入MP4文件，参数分别是：文件名，编码格式，帧率，帧大小
    // cv::VideoWriter writer("./output/record.mp4", cv::VideoWriter::fourcc('H', '2', '6', '4'), video_fps, frameSize);

    cv::Mat frame;

    int img_size = frameSize.width * frameSize.height;
    cuda_preprocess_init(img_size); // 申请cuda内存

    Polygon g_ploygon;
    // 检查多边形定点配置文件是否存在
    std::ifstream infile("./config/polygon.txt");
    if (infile)
    {
        std::cout << "检测到多边形顶点配置文件，从文件中恢复多边形定点" << std::endl;
        readPoints("./config/polygon.txt", g_ploygon, frameSize.width, frameSize.height);
    }
    else
    {
        std::cout << "未检测到多边形顶点配置文件" << std::endl;
        return -1;
    }

    // all start
    auto start_all = std::chrono::high_resolution_clock::now();
    int frame_count = 0;

    while (cap.isOpened())
    {

        // step1 start
        auto start_1 = std::chrono::high_resolution_clock::now();
        cap >> frame;
        if (frame.empty())
        {
            std::cout << "文件处理完毕" << std::endl;
            break;
        }
        // step1 end
        auto end_1 = std::chrono::high_resolution_clock::now();
        auto elapsed_1 = std::chrono::duration_cast<std::chrono::microseconds>(end_1 - start_1).count() / 1000.f;

        // step2 start
        auto start_2 = std::chrono::high_resolution_clock::now();
        // 选择预处理方式
        if (mode == 0)
        {
            // 使用CPU做letterbox、归一化、BGR2RGB、NHWC to NCHW
            process_input_cpu(frame, (float *)buffers.getDeviceBuffer(kInputTensorName));
        }
        else if (mode == 1)
        {
            // 使用CPU做letterbox，GPU做归一化、BGR2RGB、NHWC to NCHW
            process_input_cv_affine(frame, (float *)buffers.getDeviceBuffer(kInputTensorName));
        }
        else if (mode == 2)
        {
            // 使用cuda预处理所有步骤
            process_input_gpu(frame, (float *)buffers.getDeviceBuffer(kInputTensorName));
        }
        // step2 end
        auto end_2 = std::chrono::high_resolution_clock::now();
        auto elapsed_2 = std::chrono::duration_cast<std::chrono::microseconds>(end_2 - start_2).count() / 1000.f;

        // ========== 5. 执行推理 =========
        // step3 start
        auto start_3 = std::chrono::high_resolution_clock::now();
        context->executeV2(buffers.getDeviceBindings().data());
        // 拷贝回host
        buffers.copyOutputToHost();

        // 从buffer manager中获取模型输出
        int32_t *num_det = (int32_t *)buffers.getHostBuffer(kOutNumDet); // 检测到的目标个数
        int32_t *cls = (int32_t *)buffers.getHostBuffer(kOutDetCls);     // 检测到的目标类别
        float *conf = (float *)buffers.getHostBuffer(kOutDetScores);     // 检测到的目标置信度
        float *bbox = (float *)buffers.getHostBuffer(kOutDetBBoxes);     // 检测到的目标框
        // step3 end
        auto end_3 = std::chrono::high_resolution_clock::now();
        auto elapsed_3 = std::chrono::duration_cast<std::chrono::microseconds>(end_3 - start_3).count() / 1000.f;

        // 执行nms（非极大值抑制），得到最后的检测框
        // step4 start
        auto start_4 = std::chrono::high_resolution_clock::now();
        std::vector<Detection> bboxs;
        yolo_nms(bboxs, num_det, cls, conf, bbox, kConfThresh, kNmsThresh);
        // step4 end
        auto end_4 = std::chrono::high_resolution_clock::now();
        auto elapsed_4 = std::chrono::duration_cast<std::chrono::microseconds>(end_4 - start_4).count() / 1000.f;

        //  计算 ： 输入数据的预处理 -->推理-->NMS后处理 耗时及fps
        auto three_step_time = std::chrono::duration_cast<std::chrono::microseconds>(end_4 - start_2).count() / 1000.f;
        std::string time_str = "time: " + std::to_string(three_step_time);
        std::string fps_str = "fps: " + std::to_string(1000.f / three_step_time);
        cv::putText(frame, time_str, cv::Point(50, 50), cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar(255, 255, 255), 2);
        cv::putText(frame, fps_str, cv::Point(50, 100), cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar(255, 255, 255), 2);

        // step5 start
        auto start_5 = std::chrono::high_resolution_clock::now();
        // 记录所有的检测框中心点
        std::vector<Point> all_points;
        // 遍历检测结果
        for (size_t j = 0; j < bboxs.size(); j++)
        {

            cv::Rect r = get_rect(frame, bboxs[j].bbox);
            // 获取检测框中心点
            Point p_center = {r.x + int(r.width / 2), r.y + int(r.height / 2)};
            // 筛选labelid为0的检测框
            if (bboxs[j].class_id == 0)
            {
                all_points.push_back(p_center);
            }
            // 检测框中心点是否在多边形内，在则画红框，不在则画绿框
            if (isInside(g_ploygon, p_center))
            {
                cv::rectangle(frame, r, cv::Scalar(0x00, 0x00, 0xFF), 2);
            }
            else
            {
                cv::rectangle(frame, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
            }
            // 绘制labelid
            // cv::putText(frame, std::to_string((int)bboxs[j].class_id), cv::Point(r.x, r.y - 10), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0x27, 0xC1, 0x36), 2);
        }
        // 获取聚集点
        auto gather_points = gather_rule(all_points, dist_threshold);
        for (size_t i = 0; i < gather_points.size(); i++)
        {

            if (gather_points[i].size() < 3)
                continue;
            for (size_t j = 0; j < gather_points[i].size(); j++)
            {
                // std::cout << "聚集点：" << gather_points[i][j].x << "," << gather_points[i][j].y << std::endl;
                // 绘制聚集点
                blender_overlay(gather_points[i][j].x, gather_points[i][j].y, 80, frame, 0.3, frameSize.height, frameSize.width);
                // cv::circle(frame, cv::Point(gather_points[i][j].x, gather_points[i][j].y), 10, cv::Scalar(0, 0, 255), -1);
            }
        }

        // 绘制多边形
        std::vector<cv::Point> polygon;
        for (size_t i = 0; i < g_ploygon.size(); i++)
        {
            polygon.push_back(cv::Point(g_ploygon[i].x, g_ploygon[i].y));
        }
        cv::polylines(frame, polygon, true, cv::Scalar(0, 0, 255), 2);
        // step5 end
        auto end_5 = std::chrono::high_resolution_clock::now();
        auto elapsed_5 = std::chrono::duration_cast<std::chrono::microseconds>(end_5 - start_5).count() / 1000.f;

        // cv::imshow("frame", frame);
        // 写入视频文件
        // writer.write(frame);

        // step6 start
        auto start_6 = std::chrono::high_resolution_clock::now();
        if (do_stream)
            streamer.stream_frame(frame.data); // 推流
        // step6 end
        auto end_6 = std::chrono::high_resolution_clock::now();
        auto elapsed_6 = std::chrono::duration_cast<std::chrono::microseconds>(end_6 - start_6).count() / 1000.f;

#if PRINT_TIME
        // microseconds 微秒，milliseconds 毫秒，seconds 秒，1微妙=0.001毫秒 = 0.000001秒
        std::cout << "each step time(ms): " << elapsed_1 << "," << elapsed_2 << "," << elapsed_3 << "," << elapsed_4 << "," << elapsed_5 << "," << elapsed_6 << std::endl;

        std::cout << "each step fps: " << 1000.0f / elapsed_1 << "," << 1000.0f / elapsed_2 << "," << 1000.0f / elapsed_3 << "," << 1000.0f / elapsed_4 << "," << 1000.0f / elapsed_5 << "," << 1000.0f / elapsed_6 << std::endl;

        // 所有步骤总耗时
        // 算法1：计算单张图片的总耗时
        auto elapsed_all = std::chrono::duration_cast<std::chrono::microseconds>(end_6 - start_1).count() / 1000.f;
        std::cout << "method 1 all steps time(ms): " << elapsed_all << ", fps: " << 1000.0f / elapsed_all << std::endl;

        // 算法2：计算超过 1s 一共处理了多少张图片
        frame_count++;
        // all end
        auto end_all = std::chrono::high_resolution_clock::now();
        auto elapsed_all_2 = std::chrono::duration_cast<std::chrono::microseconds>(end_all - start_all).count() / 1000.f;
        // 每隔1秒打印一次
        if (elapsed_all_2 > 1000)
        {
            std::cout << "method 2 all steps time(ms): " << elapsed_all_2 << ", fps: " << frame_count / (elapsed_all_2 / 1000.0f) << ",frame count: " << frame_count << std::endl;
            frame_count = 0;
            start_all = std::chrono::high_resolution_clock::now();
        }

        std::cout << "==============================" << std::endl;
#endif
        // if (cv::waitKey(1) == 27)
        //     break;
    }
    // ========== 6. 释放资源 =========
    // 因为使用了unique_ptr，所以不需要手动释放

    return 0;
}
