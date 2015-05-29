/// Copyright 2015 Miquel Massot Campos
/// Systems, Robotics and Vision
/// University of the Balearic Islands
/// All rights reserved.

#include <pcl_feature_extraction/features.h>

#include <pcl/registration/correspondence_rejection_sample_consensus.h>

// pcl keypoints
#include <pcl/features/normal_3d.h>
#include <pcl/features/3dsc.h>
#include <pcl/features/board.h>
#include <pcl/features/boundary.h>
#include <pcl/features/don.h>
#include <pcl/features/cvfh.h>
#include <pcl/features/esf.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/intensity_gradient.h>
#include <pcl/features/intensity_spin.h>
#include <pcl/features/moment_invariants.h>
#include <pcl/features/narf_descriptor.h>
#include <pcl/features/narf.h>
#include <pcl/features/pfh.h>
#include <pcl/features/principal_curvatures.h>
#include <pcl/features/range_image_border_extractor.h>
#include <pcl/features/rift.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/shot_lrf_omp.h>
#include <pcl/features/spin_image.h>
#include <pcl/features/usc.h>
#include <pcl/features/vfh.h>

using namespace pcl;
using namespace pcl::registration;

/** \brief Class constructor. Initialize the class
  */
Features::Features(const string& descriptor_type)
  : descriptor_type_(descriptor_type) {}


/** \brief Normal estimation
  * @return
  * \param Input cloud
  * \param Output cloud with normals
  */
void Features::estimateNormals(const PointCloudRGB::Ptr& cloud,
                               PointCloud<Normal>::Ptr& normals)
{
  NormalEstimation<PointRGB, Normal> normalEstimation;
  normalEstimation.setInputCloud(cloud);
  normalEstimation.setRadiusSearch(0.05);
  search::KdTree<PointRGB>::Ptr kdtree(new search::KdTree<PointRGB>);
  normalEstimation.setSearchMethod(kdtree);
  normalEstimation.compute(*normals);
}

/** \brief Compute the intensity gradients
  * @return
  * \param Input intensity cloud
  * \param Input cloud with normals
  * \param Output cloud with gradients
  */
void Features::computeGradient(const PointCloud<PointXYZI>::Ptr& intensity,
                               const PointCloud<Normal>::Ptr& normals,
                               PointCloud<IntensityGradient>::Ptr& gradients)
{
  // Compute the intensity gradients.
  IntensityGradientEstimation<PointXYZI,
                              Normal,
                              IntensityGradient,
                              common::IntensityFieldAccessor<PointXYZI> > ge;
  ge.setInputCloud(intensity);
  ge.setInputNormals(normals);
  ge.setRadiusSearch(0.08);
  ge.compute(*gradients);
}

/** \brief Converts a pointcloud to a range image
  * @return
  * \param Input cloud
  * \param Output range image
  */
void Features::convertToRangeImage(const PointCloudRGB::Ptr& cloud,
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

/** \brief Compute descriptors
  * @return
  * \param Input cloud
  * \param Input keypoints, where features will be computed
  * \param Output descriptors
  */
template<typename FeatureType>
void Features<FeatureType>::compute(const PointCloudRGB& cloud,
                                    const PointCloudRGB& keypoints,
                                    typename PointCloud<FeatureType>::Ptr descriptors)
{
  // Normals are required for many features
  PointCloud<Normal>::Ptr keypoint_normals(new PointCloud<Normal>);

  if (descriptor_type_ == DESC_SHAPE_CONTEXT)
  {
    ShapeContext3DEstimation<PointCloudRGB, Normal, ShapeContext1980> impl;
    impl.setSearchSurface(cloud);
    impl.setInputCloud(keypoints);
    estimateNormals(keypoints, *keypoint_normals);
    impl.setInputNormals(keypoint_normals);
    search::KdTree<PointCloudRGB>::Ptr kdtree(new search::KdTree<PointCloudRGB>);
    impl.setSearchMethod(kdtree);
    impl.setRadiusSearch(0.05);
    impl.setMinimalRadius(0.05 / 10.0);
    impl.setPointDensityRadius(0.05 / 5.0);
    impl.compute(*descriptors);
  }
  else if (descriptor_type_ == DESC_USC)
  {

  }
  else if (descriptor_type_ == DESC_BOARD)
  {

  }
  else if (descriptor_type_ == DESC_BOUNDARY)
  {

  }
  else if (descriptor_type_ == DESC_INT_GRAD)
  {

  }
  else if (descriptor_type_ == DESC_INT_SPIN)
  {

  }
  else if (descriptor_type_ == DESC_RIB)
  {

  }
  else if (descriptor_type_ == DESC_SPIN_IMAGE)
  {

  }
  else if (descriptor_type_ == DESC_MOMENT_INV)
  {

  }
  else if (descriptor_type_ == DESC_CRH)
  {

  }
  else if (descriptor_type_ == DESC_DIFF_OF_NORM)
  {


  }
  else if (descriptor_type_ == DESC_ESF)
  {
    ESFEstimation<PointRGB, ESFSignature640> impl;
    impl.setSearchSurface(cloud);
    impl.setInputCloud(keypoints);
    search::KdTree<PointCloudRGB>::Ptr kdtree(new search::KdTree<PointCloudRGB>);
    impl.setSearchMethod(kdtree);
    impl.setRadiusSearch(0.08);
    impl.compute(*descriptors);
  }
  else if (descriptor_type_ == DESC_FPFH)
  {
    FPFHEstimation<PointRGB, Normal, FPFHSignature33> impl;
    impl.setSearchSurface(cloud);
    impl.setInputCloud(keypoints);
    estimateNormals(keypoints, *keypoint_normals);
    impl.setInputNormals(keypoint_normals);
    search::KdTree<PointCloudRGB>::Ptr kdtree(new search::KdTree<PointCloudRGB>);
    impl.setSearchMethod(kdtree);
    impl.setRadiusSearch(0.08);
    impl.compute(*descriptors);
  }
  else if (descriptor_type_ == DESC_NARF)
  {
    PointCloud<Narf36>::Ptr descriptors(new PointCloud<Narf36>);
    RangeImagePlanar range_image;
    convertToRangeImage(cloud, range_image)
    NarfDescriptor impl(&range_image, &keypoints);
    impl.getParameters().support_size = 0.2f;
    impl.getParameters().rotation_invariant = true;
    impl.compute(*descriptors);
  }
  else if (descriptor_type_ == DESC_VFH)
  {
    VFHEstimation<PointRGB, Normal, VFHSignature308> impl;
    impl.setSearchSurface(cloud);
    impl.setInputCloud(keypoints);
    estimateNormals(keypoints, *keypoint_normals);
    impl.setInputNormals(keypoint_normals);
    search::KdTree<PointCloudRGB>::Ptr kdtree(new search::KdTree<PointCloudRGB>);
    impl.setSearchMethod(kdtree);
    impl.setRadiusSearch(0.08);
    impl.compute(*descriptors);
  }
  else if (descriptor_type_ == DESC_CVFH)
  {
    CVFHEstimation<PointRGB, Normal, VFHSignature308> impl;
    impl.setSearchSurface(cloud);
    impl.setInputCloud(keypoints);
    estimateNormals(keypoints, *keypoint_normals);
    impl.setInputNormals(keypoint_normals);
    search::KdTree<PointCloudRGB>::Ptr kdtree(new search::KdTree<PointCloudRGB>);
    impl.setSearchMethod(kdtree);
    impl.setRadiusSearch(0.08);
    impl.compute(*descriptors);
  }
  else if (descriptor_type_ == DESC_PFH)
  {
    PFHEstimation<PointCloudRGB, Normal, PFHSignature125> impl;
    impl.setSearchSurface(cloud);
    impl.setInputCloud(keypoints);
    estimateNormals(keypoints, *keypoint_normals);
    impl.setInputNormals(keypoint_normals);
    search::KdTree<PointCloudRGB>::Ptr kdtree(new search::KdTree<PointCloudRGB>);
    impl.setSearchMethod(kdtree);
    impl.setRadiusSearch(0.08);
    impl.compute(*descriptors);
  }
  else if (descriptor_type_ == DESC_PPAL_CURV)
  {
    PointCloudXYZ::Ptr keypoints_xyz
    copyPointCloud(*keypoints, *keypoints_xyz);
    PrincipalCurvaturesEstimation<PointXYZ, Normal, PrincipalCurvatures> impl;
    impl.setSearchSurface(cloud);
    impl.setInputCloud(keypoints_xyz);
    estimateNormals(keypoints, *keypoint_normals);
    impl.setInputNormals(keypoint_normals);
    search::KdTree<PointCloudRGB>::Ptr kdtree(new search::KdTree<PointCloudRGB>);
    impl.setSearchMethod(kdtree);
    impl.setRadiusSearch(0.08);
    impl.compute(*descriptors);
  }
  else if (descriptor_type_ == DESC_RIFT)
  {
    PointCloud<PointXYZI>::Ptr cloud_intensities(new PointCloud<PointXYZI>);
    PointCloud<IntensityGradient>::Ptr keypoint_gradients(new PointCloud<IntensityGradient>);
    estimateNormals(keypoints, *keypoint_normals);
    PointCloudXYZRGBtoXYZI(keypoints, *keypoint_intensities);
    PointCloudXYZRGBtoXYZI(cloud, *cloud_intensities);
    computeGradient(keypoint_intensities, keypoint_normals, *keypoint_gradients)

    RIFTEstimation<PointXYZI, IntensityGradient, RIFT32> impl;
    impl.setInputCloud(keypoint_intensities);
    impl.setSearchSurface(cloud_intensities);
    search::KdTree<PointCloudRGB>::Ptr kdtree(new search::KdTree<PointCloudRGB>);
    impl.setSearchMethod(kdtree);
    impl.setInputGradient(keypoint_gradients);
    impl.setRadiusSearch(0.08);
    impl.setNrDistanceBins(4);
    impl.setNrGradientBins(8);
    impl.compute(*descriptors);
  }
  else if (descriptor_type_ == DESC_RSD)
  {
    RSDEstimation<PointRGB, Normal, PrincipalRadiiRSD> impl;
    impl.setSearchSurface(cloud);
    impl.setInputCloud(keypoints);
    estimateNormals(keypoints, *keypoint_normals);
    impl.setInputNormals(keypoint_normals);
    search::KdTree<PointCloudRGB>::Ptr kdtree(new search::KdTree<PointCloudRGB>);
    impl.setSearchMethod(kdtree);
    impl.setRadiusSearch(0.08);
    impl.setPlaneRadius(0.1);
    impl.setSaveHistograms(false);
    impl.compute(*descriptors);
  }
  else if (descriptor_type_ == DESC_SHOT)
  {
    PointCloudXYZ::Ptr keypoints_xyz
    copyPointCloud(*keypoints, *keypoints_xyz);
    SHOTEstimationOMP<PointXYZ, Normal, SHOT352> impl;
    impl.setSearchSurface(cloud);
    impl.setInputCloud(keypoints);
    estimateNormals(keypoints, *keypoint_normals);
    impl.setInputNormals(keypoint_normals);
    search::KdTree<PointCloudRGB>::Ptr kdtree(new search::KdTree<PointCloudRGB>);
    impl.setSearchMethod(kdtree);
    impl.setRadiusSearch(0.05);
    impl.compute(*descriptors);
  }
  else if (descriptor_type_ == DESC_SHOT_COLOR)
  {
    SHOTColorEstimationOMP<PointRGB, Normal, SHOT1344> impl;
    impl.setSearchSurface(cloud);
    impl.setInputCloud(keypoints);
    estimateNormals(keypoints, *keypoint_normals);
    impl.setInputNormals(keypoint_normals);
    search::KdTree<PointCloudRGB>::Ptr kdtree(new search::KdTree<PointCloudRGB>);
    impl.setSearchMethod(kdtree);
    impl.setRadiusSearch(0.05);
    impl.compute(*descriptors);
  }
  else if (descriptor_type_ == DESC_SHOT_LRF)
  {
    PointCloudXYZ::Ptr keypoints_xyz
    copyPointCloud(*keypoints, *keypoints_xyz);
    SHOTLocalReferenceFrameEstimationOMP<PointXYZ, Normal, SHOT352> impl;
    impl.setSearchSurface(cloud);
    impl.setInputCloud(keypoints);
    estimateNormals(keypoints, *keypoint_normals);
    impl.setInputNormals(keypoint_normals);
    search::KdTree<PointCloudRGB>::Ptr kdtree(new search::KdTree<PointCloudRGB>);
    impl.setSearchMethod(kdtree);
    impl.setRadiusSearch(0.05);
    impl.compute(*descriptors);
  }
  else
  {
    ROS_WARN("Descriptor type unavailable");
  }
}

template<typename FeatureType>
void Features<FeatureType>::findCorrespondences(typename PointCloud<FeatureType>::Ptr source,
                                                typename PointCloud<FeatureType>::Ptr target,
                                                CorrespondencesPtr correspondences) const
{
  vector<int>& source2target;
  vector<int>& target2source;
  const int k = 1;
  vector<int> k_indices(k);
  vector<float> k_dist(k);
  KdTreeFLANN<FeatureType> descriptor_kdtree;

  // Find the index of the best match for each keypoint
  // From source to target
  descriptor_kdtree.setInputCloud(target);
  source2target.resize(source->size());
  for (size_t i = 0; i < source->size(); ++i)
  {
    descriptor_kdtree.nearestKSearch(*source, i, k, k_indices, k_dist);
    source2target[i] = k_indices[0];
  }
  // and from target to source
  descriptor_kdtree.setInputCloud(source);
  target2source.resize(target->size());
  for (size_t i = 0; i < target->size(); ++i)
  {
    descriptor_kdtree.nearestKSearch(*target, i, k, k_indices, k_dist);
    target2source[i] = k_indices[0];
  }

  // now populate the correspondences vector
  vector<pair<unsigned, unsigned> > c;
  for (unsigned c_idx = 0; c_idx < source2target.size (); ++c_idx)
    if (target2source[source2target[c_idx]] == c_idx)
      c.push_back(make_pair(c_idx, source2target[c_idx]));

  correspondences->resize(c.size());
  for (unsigned c_idx = 0; c_idx < c.size(); ++c_idx)
  {
    (*correspondences)[c_idx].index_query = c[c_idx].first;
    (*correspondences)[c_idx].index_match = c[c_idx].second;
  }
}

template<typename FeatureType>
void Features<FeatureType>::filterCorrespondences(typename PointCloud<FeatureType>::Ptr source,
                                                  typename PointCloud<FeatureType>::Ptr target,
                                                  CorrespondencesPtr correspondences,
                                                  CorrespondencesPtr filtered_correspondences)
{
  registration::CorrespondenceRejectorSampleConsensus<PointXYZI> rejector;
  rejector.setInputCloud(source);
  rejector.setTargetCloud(target);
  rejector.setInputCorrespondences(correspondences);
  rejector.getCorrespondences(*filtered_correspondences);
}
