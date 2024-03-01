#ifndef TENSORRT_YOLOV9_ROS_H
#define TENSORRT_YOLOV9_ROS_H

/// C++ standard headers
#include <string>
#include <vector>
#include <chrono>
#include <ctime>
/// Main TensorRT YOLO header
#include "yolov9.hpp"
/// ROS headers
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
/// this package headers
#include <tensorrt_yolov9_ros/bbox.h>
#include <tensorrt_yolov9_ros/bboxes.h>


///////////////////////////////////////////////////////////////////
class TensorrtYoloRos
{
private:
    ///// YOLO
    bool m_image_compressed;
    double m_confidence_thresh;
    double m_nms_thresh;
    int m_downsampling_infer;
    int m_counter = 0;
    std::string m_engine_file_path;
    std::string m_image_topic;
    std::vector<std::string> m_classes;
    std::shared_ptr<Yolov9> m_yolo = nullptr;
    ///// ros and tf
    ros::NodeHandle m_nh;
    ros::Subscriber m_img_sub;
    ros::Publisher m_detected_img_pub, m_bounding_box_pub;
    ///// Functions
    void rawImageCallback(const sensor_msgs::Image::ConstPtr& msg);
    void compressedImageCallback(const sensor_msgs::CompressedImage::ConstPtr& msg);
    void processImage(cv::Mat& img_in, const double& time, const bool& is_compressed);
public:
    TensorrtYoloRos(const ros::NodeHandle& n); // constructor
    ~TensorrtYoloRos(){}; // destructor
};

TensorrtYoloRos::TensorrtYoloRos(const ros::NodeHandle& n) : m_nh(n)
{
    ///// params
    m_nh.param<bool>("/tensorrt_yolov9_ros/image_compressed", m_image_compressed, true);
    m_nh.param<std::string>("/tensorrt_yolov9_ros/image_topic", m_image_topic, "/camera/image_raw");
    m_nh.param<std::string>("/tensorrt_yolov9_ros/engine_file_path", m_engine_file_path, "yolov9.engine");
    m_nh.param<double>("/tensorrt_yolov9_ros/confidence_thres", m_confidence_thresh, 0.3);
    m_nh.param<double>("/tensorrt_yolov9_ros/nms_thres", m_nms_thresh, 0.4);
    m_nh.param<int>("/tensorrt_yolov9_ros/downsampling_infer", m_downsampling_infer, 1);
    m_nh.param<std::vector<std::string>>("/tensorrt_yolov9_ros/classes", m_classes, {"person", "bicycle"});

    ///// sub pub
    if (m_image_compressed)
    {
        m_img_sub = m_nh.subscribe<sensor_msgs::CompressedImage>(m_image_topic, 10, &TensorrtYoloRos::compressedImageCallback, this);
        m_detected_img_pub = m_nh.advertise<sensor_msgs::CompressedImage>("/detected_output"+m_image_topic, 10);
    }
    else
    {
        m_img_sub = m_nh.subscribe<sensor_msgs::Image>(m_image_topic, 10, &TensorrtYoloRos::rawImageCallback, this);
        m_detected_img_pub = m_nh.advertise<sensor_msgs::Image>("/detected_output"+m_image_topic, 10); 
    }
    m_bounding_box_pub = m_nh.advertise<tensorrt_yolov9_ros::bboxes>("/detected_bounding_boxes", 10);

    ROS_WARN("class heritated, starting node...");
}; // constructor

void TensorrtYoloRos::processImage(cv::Mat& img_in, const double& time, const bool& is_compressed)
{
    // Note: init yolo only once, if initialized in constructor, it will not work
    if (m_yolo == nullptr)
    {
        m_yolo = std::make_shared<Yolov9>(m_engine_file_path, m_confidence_thresh, m_nms_thresh, m_classes);
    }
    // infer and draw
    std::chrono::high_resolution_clock::time_point start_time_ = std::chrono::high_resolution_clock::now();
    std::vector<Detection> bboxes_out_;
    m_yolo->predict(img_in, bboxes_out_);
    m_yolo->draw(img_in, bboxes_out_);
    std::chrono::high_resolution_clock::time_point end_time_ = std::chrono::high_resolution_clock::now();

    // handle output
    tensorrt_yolov9_ros::bboxes out_boxes_;
    out_boxes_.header.stamp = ros::Time().fromSec(time);
    for (size_t i = 0; i < bboxes_out_.size(); ++i)
    {
        tensorrt_yolov9_ros::bbox out_box_;
        auto detected_ = bboxes_out_[i];
        out_box_.score = detected_.conf;
        out_box_.x = detected_.bbox.x;
        out_box_.y = detected_.bbox.y;
        out_box_.width = detected_.bbox.width;
        out_box_.height = detected_.bbox.height;
        out_box_.id = detected_.class_id;
        out_box_.Class = m_yolo->getClassName(detected_.class_id);
        out_boxes_.bboxes.push_back(out_box_);
    }

    // publish
    if (out_boxes_.bboxes.size() > 0)
    {
        m_bounding_box_pub.publish(out_boxes_);
    }

    // draw fps and date time
    char fps_[40], date_time_[40];
    std::sprintf(fps_, "%.2f ms infer + draw", std::chrono::duration_cast<std::chrono::microseconds>(end_time_ - start_time_).count()/1e3);
    cv::putText(img_in, std::string(fps_), cv::Point(5, 20), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 0), 2);
    
    std::time_t timer_ = std::time(NULL);
    struct std::tm* t_;
    t_ = std::localtime(&timer_);
    if (t_) // not NULL
    {
        std::sprintf(date_time_, "%d-%d-%d_%d:%d:%d__%d", t_->tm_year+1900, t_->tm_mon+1, t_->tm_mday, t_->tm_hour, t_->tm_min, t_->tm_sec, m_counter);
        cv::putText(img_in, std::string(date_time_), cv::Point(5, 40), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(255, 50, 50), 2);
    }

    // publish image
    cv_bridge::CvImage bridge_img_ = cv_bridge::CvImage(out_boxes_.header, sensor_msgs::image_encodings::BGR8, img_in);
    if (is_compressed)
    {
        sensor_msgs::CompressedImage _comp_img_msg;
        bridge_img_.toCompressedImageMsg(_comp_img_msg);
        m_detected_img_pub.publish(_comp_img_msg);
    }
    else
    {
        sensor_msgs::Image _raw_img_msg;
        bridge_img_.toImageMsg(_raw_img_msg);
        m_detected_img_pub.publish(_raw_img_msg);
    }

    return;
}
void TensorrtYoloRos::compressedImageCallback(const sensor_msgs::CompressedImage::ConstPtr& msg)
{
    m_counter++;
    if (m_counter % m_downsampling_infer==0)
    {
        cv::Mat img_in_ = cv_bridge::toCvCopy(*msg, sensor_msgs::image_encodings::BGR8)->image;
        processImage(img_in_, msg->header.stamp.toSec() , true);
    }
    return;
}
void TensorrtYoloRos::rawImageCallback(const sensor_msgs::Image::ConstPtr& msg)
{
    m_counter++;
    if (m_counter % m_downsampling_infer==0)
    {
        cv::Mat img_in_ = cv_bridge::toCvCopy(*msg, sensor_msgs::image_encodings::BGR8)->image;
        processImage(img_in_, msg->header.stamp.toSec(), false);
    }
    return;
}



#endif