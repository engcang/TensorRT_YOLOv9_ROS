#pragma once

/// c++ headers
#include <fstream> // std::ifstream
#include <random> // std::random_device, std::mt19937, std::uniform_int_distribution
#include <vector> // std::vector
/// this package headers
#include "macros.h"
#include "logging.h" // Logger
#include "cuda_utils.h" // CUDA_CHECK
#include "preprocess.h" // cuda_preprocess_init, cuda_preprocess, cuda_preprocess_destroy
/// third party headers
#include "NvInfer.h" // TensorRT
#include <opencv2/opencv.hpp> // OpenCV

static Logger gLogger;
using namespace nvinfer1;

struct Detection
{
    float conf;
    int class_id;
    cv::Rect bbox;
};

class Yolov9
{

public:
    Yolov9(){};
    Yolov9(const std::string& engine_file_path_in, 
           const float& confidence_thresh_in,
           const float& nms_thresh_in,
           const std::vector<std::string>& classes_in);
    ~Yolov9();
   
    void predict(cv::Mat& image, std::vector<Detection>& output);
    void draw(cv::Mat& image, std::vector<Detection>& output);
    std::string getClassName(const int& class_id) const { return classes[class_id]; }

private:
    void postprocess(std::vector<Detection>& output);

private:
    float* gpu_buffers[2];               //!< The std::vector of device buffers needed for engine execution
    float* cpu_output_buffer;

    cudaStream_t cuda_stream;
    IRuntime* runtime;                   //!< The TensorRT runtime used to deserialize the engine
    ICudaEngine* engine;                 //!< The TensorRT engine used to run the network
    IExecutionContext* context;          //!< The context for executing inference using an ICudaEngine

    // Model parameters
    int model_input_w;
    int model_input_h;
    int num_detections;
    int detection_attribute_size;
    int num_classes = 2;
    const int MAX_IMAGE_SIZE = 4096 * 4096;
    float conf_threshold = 0.3f;
    float nms_threshold = 0.4f;
    std::vector<std::string> classes;
    std::vector<cv::Scalar> colors;
};

Yolov9::Yolov9(const std::string& engine_file_path_in, 
               const float& confidence_thresh_in,
               const float& nms_thresh_in,
               const std::vector<std::string>& classes_in)
              : conf_threshold(confidence_thresh_in), nms_threshold(nms_thresh_in), classes(classes_in)
{
    // Read the engine file
    std::ifstream engineStream(engine_file_path_in, std::ios::binary);
    engineStream.seekg(0, std::ios::end);
    const size_t modelSize = engineStream.tellg();
    engineStream.seekg(0, std::ios::beg);
    std::unique_ptr<char[]> engineData(new char[modelSize]);
    engineStream.read(engineData.get(), modelSize);
    engineStream.close();

    // Deserialize the tensorrt engine
    runtime = createInferRuntime(gLogger);
    engine = runtime->deserializeCudaEngine(engineData.get(), modelSize);
    context = engine->createExecutionContext();

    // Get input and output sizes of the model
#if NV_TENSORRT_MAJOR >= 10
    auto const name_input = engine->getIOTensorName(0);
    auto const name_output = engine->getIOTensorName(1);
    model_input_h = engine->getTensorShape(name_input).d[2];
    model_input_w = engine->getTensorShape(name_input).d[3];
    detection_attribute_size = engine->getTensorShape(name_output).d[1];
    num_detections = engine->getTensorShape(name_output).d[2];
#else
    model_input_h = engine->getBindingDimensions(0).d[2];
    model_input_w = engine->getBindingDimensions(0).d[3];
    detection_attribute_size = engine->getBindingDimensions(1).d[1];
    num_detections = engine->getBindingDimensions(1).d[2];
#endif
    num_classes = detection_attribute_size - 4;

    // Initialize input buffers
    cpu_output_buffer = new float[detection_attribute_size * num_detections];
    CUDA_CHECK(cudaMalloc(&gpu_buffers[0], 3 * model_input_w * model_input_h * sizeof(float)));
    // Initialize output buffer
    CUDA_CHECK(cudaMalloc(&gpu_buffers[1], detection_attribute_size * num_detections * sizeof(float)));

    cuda_preprocess_init(MAX_IMAGE_SIZE);

    CUDA_CHECK(cudaStreamCreate(&cuda_stream));

    // Create random colors
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(70, 255);
    for (size_t i = 0; i < classes.size(); i++)
    {
        cv::Scalar color = cv::Scalar(dis(gen), dis(gen), dis(gen));
        colors.push_back(color);
    }
}

Yolov9::~Yolov9()
{
    // Release stream and buffers
    CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
    CUDA_CHECK(cudaStreamDestroy(cuda_stream));
    for (int i = 0; i < 2; i++)
    {
        CUDA_CHECK(cudaFree(gpu_buffers[i]));
    }
    delete[] cpu_output_buffer;

    // Destroy the engine
    cuda_preprocess_destroy();
    delete context;
    delete engine;
    delete runtime;
}

void Yolov9::predict(cv::Mat& image, std::vector<Detection> &output)
{
    // Preprocessing data on gpu
    cuda_preprocess(image.ptr(), image.cols, image.rows, gpu_buffers[0], model_input_w, model_input_h, cuda_stream);
    CUDA_CHECK(cudaStreamSynchronize(cuda_stream));

    // Perform inference
#if NV_TENSORRT_MAJOR >= 10
    for (int32_t i = 0, e = engine->getNbIOTensors(); i < e; i++)
    {
        auto const name = engine->getIOTensorName(i);
        context->setTensorAddress(name, gpu_buffers[i]);
    }
    context->enqueueV3(cuda_stream);
#else
    context->enqueueV2((void**)gpu_buffers, cuda_stream, nullptr);
#endif

    // Memcpy from device output buffer to host output buffer
    CUDA_CHECK(cudaMemcpyAsync(cpu_output_buffer, gpu_buffers[1], num_detections * detection_attribute_size * sizeof(float), cudaMemcpyDeviceToHost, cuda_stream));
    CUDA_CHECK(cudaStreamSynchronize(cuda_stream));

    // Perform postprocessing
    postprocess(output);
}

void Yolov9::postprocess(std::vector<Detection>& output)
{
    std::vector<cv::Rect> boxes;
    std::vector<int> class_ids;
    std::vector<float> confidences;

    const cv::Mat det_output(detection_attribute_size, num_detections, CV_32F, cpu_output_buffer);

    for (int i = 0; i < det_output.cols; ++i)
    {
        const cv::Mat classes_scores = det_output.col(i).rowRange(4, 4 + num_classes);
        cv::Point class_id_point;
        double score;
        cv::minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);

        if (score > conf_threshold) {
            const float cx = det_output.at<float>(0, i);
            const float cy = det_output.at<float>(1, i);
            const float ow = det_output.at<float>(2, i);
            const float oh = det_output.at<float>(3, i);
            cv::Rect box;
            box.x = static_cast<int>((cx - 0.5 * ow));
            box.y = static_cast<int>((cy - 0.5 * oh));
            box.width = static_cast<int>(ow);
            box.height = static_cast<int>(oh);

            boxes.push_back(box);
            class_ids.push_back(class_id_point.y);
            confidences.push_back(score);
        }
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, nms_result);

    for (size_t i = 0; i < nms_result.size(); i++)
    {
        Detection result;
        size_t idx = nms_result[i];
        result.class_id = class_ids[idx];
        result.conf = confidences[idx];
        result.bbox = boxes[idx];
        output.push_back(result);
    }
}

void Yolov9::draw(cv::Mat& image, std::vector<Detection>& output)
{
    const float ratio_h = model_input_h / (float)image.rows;
    const float ratio_w = model_input_w / (float)image.cols;

    for (size_t i = 0; i < output.size(); i++)
    {
        auto& detection = output[i]; // Use auto& to get a reference to each Detection object
        auto& box = detection.bbox; // Use auto& to directly modify the bbox of the Detection object
        auto class_id = detection.class_id;
        auto conf = detection.conf;

        if (ratio_h > ratio_w) 
        {
            box.x = static_cast<int>(box.x / ratio_w);
            box.y = static_cast<int>((box.y - (model_input_h - ratio_w * image.rows) / 2) / ratio_w);
            box.width = static_cast<int>(box.width / ratio_w);
            box.height = static_cast<int>(box.height / ratio_w);
        }
        else 
        {
            box.x = static_cast<int>((box.x - (model_input_w - ratio_h * image.cols) / 2) / ratio_h);
            box.y = static_cast<int>(box.y / ratio_h);
            box.width = static_cast<int>(box.width / ratio_h);
            box.height = static_cast<int>(box.height / ratio_h);
        }
        
        // Detection box
        cv::rectangle(image, cv::Point(box.x, box.y), cv::Point(box.x + box.width, box.y + box.height), colors[class_id], 2);
        // Detection box text
        std::string class_string = classes[class_id] + ' ' + std::to_string(conf).substr(0, 4);
        cv::Size text_size = getTextSize(class_string, cv::FONT_HERSHEY_SIMPLEX, 0.5, 2, 0);
        cv::rectangle(image, cv::Rect(box.x - 1, box.y - text_size.height - 9, text_size.width + 2, text_size.height + 8), colors[class_id], cv::FILLED);
        cv::putText(image, class_string, cv::Point(box.x, box.y - 8), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2, 0);
    }
}
