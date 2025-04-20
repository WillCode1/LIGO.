#pragma once
#include <ros/ros.h>
#include "Header.h"

extern bool save_pgm;
extern double pgm_resolution;
extern float min_z, max_z;

void init_pgo_system(ros::NodeHandle &nh);

void pgo_handle(PointXYZIRPYT &this_pose6d, PointCloudType::Ptr &feats_undistort, PointCloudType::Ptr &submap_fix);
