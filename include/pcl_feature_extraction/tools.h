/// Copyright 2015 Miquel Massot Campos
/// Systems, Robotics and Vision
/// University of the Balearic Islands
/// All rights reserved.

#ifndef TOOLS_H
#define TOOLS_H

#include <pcl/common/common.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/intensity_gradient.h>

class Tools {
 public:
  /** \brief Normal estimation
    * @return
    * \param Cloud where normals will be estimated
    * \param Cloud surface with additional information to estimate the features for every point in the input dataset
    * \param Output cloud with normals
    */
  static void estimateNormals(const PointCloudRGB::Ptr& cloud,
                              PointCloud<Normal>::Ptr& normals,
                              double radius_search)
  {
    NormalEstimation<PointRGB, Normal> normal_estimation;
    normal_estimation.setInputCloud(cloud);
    normal_estimation.setRadiusSearch(radius_search);
    search::KdTree<PointRGB>::Ptr kdtree(new search::KdTree<PointRGB>);
    normal_estimation.setSearchMethod(kdtree);
    normal_estimation.compute(*normals);
  }

  /** \brief Compute the intensity gradients
    * @return
    * \param Input intensity cloud
    * \param Input cloud with normals
    * \param Output cloud with gradients
    */
  static void computeGradient(const PointCloud<PointXYZI>::Ptr& intensity,
                              const PointCloud<Normal>::Ptr& normals,
                              PointCloud<IntensityGradient>::Ptr& gradients,
                              double radius_search)
  {
    // Compute the intensity gradients.
    IntensityGradientEstimation<PointXYZI,
                                Normal,
                                IntensityGradient,
                                common::IntensityFieldAccessor<PointXYZI> > ge;
    ge.setInputCloud(intensity);
    ge.setInputNormals(normals);
    ge.setRadiusSearch(radius_search);
    ge.compute(*gradients);
  }

  /** \brief Converts a pointcloud to a range image
    * @return
    * \param Input cloud
    * \param Output range image
    */
  static void convertToRangeImage(const PointCloudRGB::Ptr& cloud,
                           RangeImagePlanar& range_image)
  {
    // Convert the cloud to range image.
    int image_size_x = 640, image_size_y = 480;
    float center_x = (640.0f / 2.0f), center_y = (480.0f / 2.0f);
    float focal_length_x = 525.0f, focal_length_y = focal_length_x;
    Eigen::Affine3f sensor_pose = Eigen::Affine3f(Eigen::Translation3f(cloud->sensor_origin_[0],
                   cloud->sensor_origin_[1],
                   cloud->sensor_origin_[2])) *
                   Eigen::Affine3f(cloud->sensor_orientation_);
    float noise_level = 0.0f, minimum_range = 0.0f;
    range_image.createFromPointCloudWithFixedSize(*cloud, image_size_x, image_size_y,
        center_x, center_y, focal_length_x, focal_length_x,
        sensor_pose, RangeImage::CAMERA_FRAME,
        noise_level, minimum_range);
  }

  static bool areEqual(const PointRGB& p, const PointRGB& q) {
    return (abs(p.x - q.x) < 0.001) && (abs(p.y - q.y) < 0.001) && (abs(p.z - q.z) < 0.001);
  }

  /**
   * @brief Extracts the inliers vector from two pointclouds
   *
   * @param cloud Input clout
   * @param inliers Input cloud of inliers.
   *
   * @return Vector whose index matches cloud inliers
   */
  static void getIndices(const PointCloudRGB::Ptr& cloud,
                         const PointCloudRGB::Ptr& inliers,
                         vector<int>& inliers_indexes)
  {
    for (size_t i = 0; i < inliers->points.size(); i++) {
      for (size_t j = 0; j < cloud->points.size(); j++) {
        if (areEqual(inliers->points[i], cloud->points[j])) {
          inliers_indexes.push_back(j);
        }
      }
    }
  }
};
#endif  // TOOLS_H
