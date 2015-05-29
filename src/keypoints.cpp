#include "pcl_feature_extraction/keypoints.h"

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

// More code: http://robotica.unileon.es/~victorm/

/** \brief Class constructor. Initialize the class
  */
Keypoints::Keypoints(string kp_type)
{
  kp_type_ = kp_type;
}


/** \brief Detects the keypoints of the input cloud
  * @return
  * \param Input cloud
  * \param Stores the indices of the detected keypoints on the input cloud
  */
void Keypoints::compute(const PointCloudRGB::Ptr& cloud, PointCloudRGB::Ptr& cloud_keypoints)
{
  // AGAST
  if (kp_type_ == KP_AGAST_DETECTOR_7_12s || kp_type_ == KP_AGAST_DETECTOR_5_8 || kp_type_ == KP_OAST_DETECTOR_9_16)
  {
    // https://code.google.com/p/kfls2/source/browse/trunk/apps/src/ni_agast.cpp?r=2

    // Keypoints
    PointCloud<PointUV>::Ptr keypoints(new PointCloud<PointUV>);

    // Parameters
    double bmax = 255;
    double threshold = 30;

    // The generic agast
    AgastKeypoint2D<PointRGB> agast;
    agast.setNonMaxSuppression(true);
    agast.setThreshold(threshold);
    agast.setMaxDataValue(bmax);
    agast.setInputCloud(cloud);

    // Detector
    if (kp_type_ == KP_AGAST_DETECTOR_7_12s)
    {
      keypoints::agast::AgastDetector7_12s::Ptr detector(new keypoints::agast::AgastDetector7_12s(cloud->width, cloud->height, threshold, bmax));
      agast.setAgastDetector(detector);
    }
    else if (kp_type_ == KP_AGAST_DETECTOR_5_8)
    {
      keypoints::agast::AgastDetector5_8::Ptr detector(new keypoints::agast::AgastDetector5_8 (cloud->width, cloud->height, threshold, bmax));
      agast.setAgastDetector(detector);

    }
    else if (kp_type_ == KP_OAST_DETECTOR_9_16)
    {
      keypoints::agast::OastDetector9_16::Ptr detector(new keypoints::agast::OastDetector9_16 (cloud->width, cloud->height, threshold, bmax));
      agast.setAgastDetector(detector);
    }

    // Compute the keypoints
    agast.compute(*keypoints);

    // Convert keypoints to pointcloud indices
    getKeypointsCloud(cloud, keypoints, cloud_keypoints);
    return;
  }

  // HARRIS 3D
  else if(kp_type_ == KP_HARRIS_3D)
  {
    // https://github.com/PointCloudLibrary/pcl/blob/master/examples/keypoints/example_get_keypoints_indices.cpp
    // http://docs.ros.org/hydro/api/pcl/html/tutorial_8cpp_source.html
    HarrisKeypoint3D<PointRGB, PointXYZI> harris3d;
    PointCloudXYZI::Ptr keypoints(new PointCloudXYZI);
    harris3d.setNonMaxSupression(true);
    harris3d.setInputCloud(cloud);
    harris3d.setThreshold(1e-6);
    harris3d.compute(*keypoints);

    // Extract the indices
    getKeypointsCloud(cloud, keypoints, cloud_keypoints);
    return;
  }

  // HARRIS 6D
  else if(kp_type_ == KP_HARRIS_6D)
  {
    HarrisKeypoint6D<PointRGB, PointXYZI> harris6d;
    PointCloudXYZI::Ptr keypoints(new PointCloudXYZI);
    harris6d.setNonMaxSupression(true);
    harris6d.setInputCloud(cloud);
    harris6d.setThreshold(1e-6);
    harris6d.compute(*keypoints);

    // Extract the indices
    getKeypointsCloud(cloud, keypoints, cloud_keypoints);
    return;
  }

  // ISS
  else if(kp_type_ == KP_ISS)
  {
    ISSKeypoint3D<PointRGB, PointRGB> detector;
    detector.setInputCloud(cloud);
    search::KdTree<PointRGB>::Ptr kdtree(new search::KdTree<PointRGB>);
    detector.setSearchMethod(kdtree);
    double resolution = computeCloudResolution(cloud);
    detector.setSalientRadius(6 * resolution);
    detector.setNonMaxRadius(4 * resolution);
    detector.setMinNeighbors(5);
    detector.setThreshold21(0.975);
    detector.setThreshold32(0.975);
    detector.compute(*cloud_keypoints);
    return;
  }

  // NARF
  else if(kp_type_ == KP_NARF)
  {
    // http://pointclouds.org/documentation/tutorials/narf_keypoint_extraction.php#narf-keypoint-extraction

    // Convert the cloud to range image.
    int image_size_x = 640, image_size_y = 480;
    float center_x = (640.0f / 2.0f), center_y = (480.0f / 2.0f);
    float focal_length_x = 525.0f, focal_length_y = focal_length_x;
    Eigen::Affine3f sensor_pose = Eigen::Affine3f(Eigen::Translation3f(cloud->sensor_origin_[0],
                   cloud->sensor_origin_[1],
                   cloud->sensor_origin_[2])) *
                   Eigen::Affine3f(cloud->sensor_orientation_);
    float noise_level = 0.0f, minimum_range = 0.0f;
    RangeImagePlanar range_image;
    range_image.createFromPointCloudWithFixedSize(*cloud, image_size_x, image_size_y,
        center_x, center_y, focal_length_x, focal_length_x,
        sensor_pose, RangeImage::CAMERA_FRAME,
        noise_level, minimum_range);

    // Extract keypoints
    PointCloud<int>::Ptr keypoints(new PointCloud<int>);
    RangeImageBorderExtractor border_extractor;
    NarfKeypoint detector(&border_extractor);
    detector.setRangeImage(&range_image);
    detector.getParameters().support_size = 0.2f;
    detector.compute(*keypoints);

    // Get the cloud indices
    cloud_keypoints.reset(new PointCloudRGB);
    for (size_t i=0; i<keypoints->points.size (); ++i)
      cloud_keypoints->points.push_back(cloud->points[keypoints->points[i]]);
    return;
  }

  // SIFT
  else if (kp_type_ == KP_SIFT)
  {
    // https://github.com/otherlab/pcl/blob/master/examples/keypoints/example_sift_normal_keypoint_estimation.cpp

    // Parameters for sift computation
    const float min_scale = 0.01;
    const int n_octaves = 3;
    const int n_scales_per_octave = 4;
    const float min_contrast = 0.001;

    // Estimate the normals of the input cloud
    PointCloud<PointNormal>::Ptr cloud_normals;
    normals(cloud, cloud_normals);

    // Estimate the sift interest points using normals values from xyz as the Intensity variants
    SIFTKeypoint<PointNormal, PointXYZI> sift;
    PointCloud<PointXYZI>::Ptr keypoints(new PointCloud<PointXYZI>);
    search::KdTree<PointNormal>::Ptr tree(new search::KdTree<PointNormal> ());
    sift.setSearchMethod(tree);
    sift.setScales(min_scale, n_octaves, n_scales_per_octave);
    sift.setMinimumContrast(min_contrast);
    sift.setInputCloud(cloud_normals);
    sift.compute(*keypoints);

    // Extract the indices
    getKeypointsCloud(cloud, keypoints, cloud_keypoints);
    return;
  }

  // SUSAN
  else if (kp_type_ == KP_SUSAN)
  {
    // Detect
    SUSANKeypoint<PointRGB, PointRGB>* susan3D = new  SUSANKeypoint<PointRGB, PointRGB>;
    susan3D->setInputCloud(cloud);
    susan3D->setNonMaxSupression(true);
    susan3D->compute(*cloud_keypoints);
    return;
  }

  // UNIFORM_SAMPLING
  else if (kp_type_ == KP_UNIFORM_SAMPLING)
  {
    // https://searchcode.com/codesearch/view/19993937/

    PointCloud<int>::Ptr keypoints(new PointCloud<int>);
    UniformSampling<PointRGB> uniform;
    uniform.setRadiusSearch(0.05);
    uniform.setInputCloud(cloud);
    uniform.compute(*keypoints);

    // Get the cloud indices
    cloud_keypoints.reset(new PointCloudRGB);
    for (size_t i=0; i<keypoints->points.size (); ++i)
      cloud_keypoints->points.push_back(cloud->points[keypoints->points[i]]);
    return;
  }
}

/** \brief Normal estimation
  * @return
  * \param Input cloud
  * \param Output cloud with normals
  */
void Keypoints::normals(const PointCloudRGB::Ptr& cloud, PointCloud<PointNormal>::Ptr& cloud_normals)
{
  // Init
  cloud_normals.reset(new PointCloud<PointNormal>);
  NormalEstimation<PointRGB, PointNormal> ne;
  search::KdTree<PointRGB>::Ptr tree_n(new search::KdTree<PointRGB>());

  ne.setInputCloud(cloud);
  ne.setSearchMethod(tree_n);
  ne.setRadiusSearch(0.05);
  ne.compute(*cloud_normals);

  // Copy the xyz info from input cloud and add it to cloud_normals as the xyz field in PointNormals estimation is zero
  for(size_t i = 0; i<cloud_normals->points.size(); ++i)
  {
    cloud_normals->points[i].x = cloud->points[i].x;
    cloud_normals->points[i].y = cloud->points[i].y;
    cloud_normals->points[i].z = cloud->points[i].z;
  }
}


/** \brief Get keypoint cloud indices from pointcloud of PointUVs
  * @return
  * \param Input cloud
  * \param Input keypoints of type PointUV
  * \param The output cloud indices corresponding to the detected keypoints
  */
void Keypoints::getKeypointsCloud(const PointCloudRGB::Ptr& cloud,
                                  const PointCloud<PointUV>::Ptr& keypoints,
                                  PointCloudRGB::Ptr& cloud_keypoints)
{
  // Reset output
  cloud_keypoints.reset(new PointCloudRGB);

  // Sanity check
  if (!cloud || !keypoints || cloud->points.empty() || keypoints->points.empty())
    return;

  // Init the kdtree
  KdTreeFLANN<PointRGB> kdtree;
  kdtree.setInputCloud(cloud);

  for (size_t i=0; i<keypoints->size(); ++i)
  {
    // Get the point in the pointcloud
    PointRGB &pt = (*cloud)(static_cast<long unsigned int>(keypoints->points[i].u),
                            static_cast<long unsigned int>(keypoints->points[i].v));

    if (!pcl_isfinite(pt.x) || !pcl_isfinite(pt.y) || !pcl_isfinite(pt.z))
      continue;

    cloud_keypoints->points.push_back(pt);
  }
}

/** \brief Get keypoint cloud indices from pointcloud of PointUVs
  * @return
  * \param Input cloud
  * \param Input keypoints of type PointXYZI
  * \param The output cloud indices corresponding to the detected keypoints
  */
void Keypoints::getKeypointsCloud(const PointCloudRGB::Ptr& cloud,
                                  const PointCloudXYZI::Ptr& keypoints,
                                  PointCloudRGB::Ptr& cloud_keypoints)
{
  // Reset output
  cloud_keypoints.reset(new PointCloudRGB);

  // Sanity check
  if (!cloud || !keypoints || cloud->points.empty() || keypoints->points.empty())
    return;

  KdTreeFLANN<PointRGB> kdtree;
  kdtree.setInputCloud(cloud);

  for (size_t i=0; i<keypoints->size(); ++i)
  {
    // Get the point in the pointcloud
    PointXYZI pt_tmp = keypoints->points[i];
    PointRGB pt;
    pt.x = pt_tmp.x;
    pt.y = pt_tmp.y;
    pt.z = pt_tmp.z;

    if (!pcl_isfinite(pt.x) || !pcl_isfinite(pt.y) || !pcl_isfinite(pt.z))
      continue;

    // Search this point into the cloud
    vector<int> idx_vec;
    vector<float> dist;
    if (kdtree.nearestKSearch(pt, 1, idx_vec, dist) > 0)
    {
      if (dist[0] < 0.0001)
        cloud_keypoints->points.push_back(cloud->points[idx_vec[0]]);
    }
  }
}

/** \brief Computes the cloud resolution. From http://pointclouds.org/documentation/tutorials/correspondence_grouping.php
  * @return the cloud resolution
  * \param Input cloud
  */
double Keypoints::computeCloudResolution(const PointCloudRGB::Ptr& cloud)
{
  double resolution = 0.0;
  int numberOfPoints = 0;
  int nres;
  vector<int> indices(2);
  vector<float> squaredDistances(2);
  search::KdTree<PointRGB> tree;
  tree.setInputCloud(cloud);

  for (size_t i = 0; i < cloud->size(); ++i)
  {
    if (! pcl_isfinite((*cloud)[i].x))
      continue;

    // Considering the second neighbor since the first is the point itself.
    nres = tree.nearestKSearch(i, 2, indices, squaredDistances);
    if (nres == 2)
    {
      resolution += sqrt(squaredDistances[1]);
      ++numberOfPoints;
    }
  }
  if (numberOfPoints != 0)
    resolution /= numberOfPoints;

  return resolution;
}
