/**
 * @file
 * @brief Keypoint class
 */

#ifndef KEYPOINT_H
#define KEYPOINT_H

#include <ros/ros.h>

// Generic pcl
#include <pcl/common/common.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/range_image/range_image_planar.h>
#include <pcl/visualization/range_image_visualizer.h>

// pcl keypoints
#include <pcl/impl/point_types.hpp>
#include <pcl/keypoints/agast_2d.h>
#include <pcl/keypoints/harris_3d.h>
#include <pcl/keypoints/harris_6d.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/keypoints/narf_keypoint.h>
#include <pcl/keypoints/susan.h>
#include <pcl/keypoints/uniform_sampling.h>

// pcl features
#include <pcl/features/normal_3d.h>
#include <pcl/features/range_image_border_extractor.h>

using namespace std;
using namespace pcl;

// pcl definition
typedef PointXYZRGB           PointRGB;
typedef PointCloud<PointXYZ>  PointCloudXYZ;
typedef PointCloud<PointXYZI> PointCloudXYZI;
typedef PointCloud<PointRGB>  PointCloudRGB;

// List the available keypoints
static const string KP_AGAST_DETECTOR_7_12s  = "AgastDetector7_12s";
static const string KP_AGAST_DETECTOR_5_8    = "AgastDetector5_8";
static const string KP_OAST_DETECTOR_9_16    = "OastDetector9_16";
static const string KP_HARRIS_3D             = "Harris3D";
static const string KP_HARRIS_6D             = "Harris6D";
static const string KP_ISS                   = "ISS";
static const string KP_NARF                  = "Narf";
static const string KP_SIFT                  = "Sift";
static const string KP_SUSAN                 = "Susan";
static const string KP_UNIFORM_SAMPLING      = "UniformSampling";

class Keypoints
{

public:

  // Constructor
  explicit Keypoints(string kp_type);

  // Detect
  void compute(const PointCloudRGB::Ptr& cloud, PointCloudRGB::Ptr& cloud_keypoints);

  // Normal estimation
  void normals(const PointCloudRGB::Ptr& cloud, PointCloud<PointNormal>::Ptr& cloud_normals);

  // Get keypoint cloud indices from pointcloud of PointUVs
  void getKeypointsCloud( const PointCloudRGB::Ptr& cloud,
                          const PointCloud<PointUV>::Ptr& keypoints,
                          PointCloudRGB::Ptr& cloud_keypoints);
  void getKeypointsCloud( const PointCloudRGB::Ptr& cloud,
                          const PointCloud<PointXYZI>::Ptr& keypoints,
                          PointCloudRGB::Ptr& cloud_keypoints);
  void getKeypointsCloud( const PointCloudRGB::Ptr& cloud,
                          const PointCloud<PointWithScale>::Ptr& keypoints,
                          PointCloudRGB::Ptr& cloud_keypoints);

  // Get the cloud index of a give point
  int getCloudIdx(pcl::KdTreeFLANN<PointRGB> kdtree,
                  PointRGB pt);

  // Computes the cloud resolution
  double computeCloudResolution(const PointCloudRGB::Ptr& cloud);

private:

  string kp_type_;  //!> Stores the keypoint type

};


#endif // KEYPOINT_H