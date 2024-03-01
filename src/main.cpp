#include "main.h"
#include <signal.h>
void signal_handler(sig_atomic_t s)
{
  std::cout << "You pressed Ctrl + C, exiting" << std::endl;
  exit(1);
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "tensorrt_yolov9_ros_node");
  ros::NodeHandle n("~");

  TensorrtYoloRos tensorrtyoloros_(n);

  signal(SIGINT, signal_handler); // to exit program when ctrl+c

  ros::AsyncSpinner spinner(2); // Use 2 threads
  spinner.start();
  ros::waitForShutdown();

  return 0;
}