/// Copyright 2015 Miquel Massot Campos
/// Systems, Robotics and Vision
/// University of the Balearic Islands
/// All rights reserved.

#ifndef FEATURES_H
#define FEATURES_H

#include <ros/ros.h>
#include <vector>
#include <string>

// Generic pcl
#include <pcl/common/common.h>
#include <pcl/point_types.h>
#include <pcl/registration/correspondence_types.h>
#include <pcl/range_image/range_image_planar.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/point_types_conversion.h>

// pcl features
#include <pcl_feature_extraction/features.h>
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

using namespace std;
using namespace pcl;
using namespace pcl::registration;

///  Types of 3D features:
///  - Shape Context 3D (SC)
///  - Unique Shape Context (USC)
///  - BOrder Aware Repeatable Direction (BOARD)
///  - Boundary
///  - Intensity Gradient (IG)
///  - Intensity Spin (IS)
///  - Range Image Border (RIB)
///  - Spin Image (SI)
///  - Moment Invariants (MI)
///  - Camera Roll Histogram (CRH)
///  - Difference of normals (DoN)
///  - Ensemble of Shape Functions (ESF)
///  - Fast Point Feature Histogram (FPFH) using OpenMP
///  - Normal Aligned Radial Features (NARF)
///  - Viewpoint Feature Histogram (VFH)
///  - Clustered Viewpoint Feature Histogram (CVFH)
///  - Point Feature Histogram (PFH)
///  - Principal Curvatures (PC)
///  - Rotation Invariant Feature Transform (RIFT)
///  - Signature of Histograms of OrienTations (SHOT) using OpenMP
///  - SHOT with colour using OpenMP (SHOTColor) using OpenMP
///  - SHOT Local Reference Frame using OpenMP (SHOTLRF)

// List the available descriptors
static const string DESC_SHAPE_CONTEXT = "ShapeContext";
static const string DESC_USC           = "USC";
static const string DESC_BOARD         = "BOARD";
static const string DESC_BOUNDARY      = "Boundary";
static const string DESC_INT_GRAD      = "IntensityGradient";
static const string DESC_INT_SPIN      = "IntensitySpin";
static const string DESC_RIB           = "RIB";
static const string DESC_SPIN_IMAGE    = "SpinImage";
static const string DESC_MOMENT_INV    = "MomentInvariants";
static const string DESC_CRH           = "CRH";
static const string DESC_DIFF_OF_NORM  = "DifferenceOfNormals";
static const string DESC_ESF           = "ESF";
static const string DESC_FPFH          = "FPFH";
static const string DESC_NARF          = "NARF";
static const string DESC_VFH           = "VFH";
static const string DESC_CVFH          = "CVFH";
static const string DESC_PFH           = "PFH";
static const string DESC_PPAL_CURV     = "PrincipalCurvatures";
static const string DESC_RIFT          = "RIFT";
static const string DESC_SHOT          = "SHOT";
static const string DESC_SHOT_COLOR    = "SHOTColore";
static const string DESC_SHOT_LRF      = "SHOTLocalReferenceFrame";

typedef PointXYZRGB PointRGB;
typedef PointCloud<PointXYZ> PointCloudXYZ;
typedef PointCloud<PointRGB> PointCloudRGB;

template<typename FeatureType>
class Features
{
 public:

  // Class constructor
  explicit Features(const string desc_type);

  // Feature computation
  void compute(const PointCloudRGB::Ptr& cloud,
               const PointCloudRGB::Ptr& keypoints,
               typename PointCloud<FeatureType>::Ptr& descriptors);

  // Search for correspondences
  void findCorrespondences(typename PointCloud<FeatureType>::Ptr source,
                           typename PointCloud<FeatureType>::Ptr target,
                           CorrespondencesPtr correspondences);

  // Correspondence filtering
  void filterCorrespondences(typename PointCloud<FeatureType>::Ptr source,
                             typename PointCloud<FeatureType>::Ptr target,
                             CorrespondencesPtr correspondences,
                             CorrespondencesPtr filtered_correspondences);

 protected:

  // Normal estimation
  void estimateNormals(const PointCloudRGB::Ptr& cloud,
                       const PointCloudRGB::Ptr& surface,
                       PointCloud<Normal>::Ptr& normals);

  // Compute the intensity gradients
  void computeGradient(const PointCloud<PointXYZI>::Ptr& intensity,
                       const PointCloud<Normal>::Ptr& normals,
                       PointCloud<IntensityGradient>::Ptr& gradients);

  // Converts a pointcloud to a range image
  void convertToRangeImage(const PointCloudRGB::Ptr& cloud,
                           RangeImagePlanar& range_image);

 private:

  string desc_type_;  //!> Stores the keypoint type
};


/** \brief Class constructor. Initialize the class
  */
template<typename FeatureType>
Features<FeatureType>::Features(const string desc_type)
  : desc_type_(desc_type) {}


/** \brief Compute descriptors
  * @return
  * \param Input cloud
  * \param Input keypoints, where features will be computed
  * \param Output descriptors
  */
template<typename FeatureType>
void Features<FeatureType>::compute(const PointCloudRGB::Ptr& cloud,
                                    const PointCloudRGB::Ptr& keypoints,
                                    typename PointCloud<FeatureType>::Ptr& descriptors)
{
  // Normals are required for many features
  PointCloud<Normal>::Ptr keypoint_normals(new PointCloud<Normal>);

  if (desc_type_ == DESC_SHAPE_CONTEXT)
  {
    PointCloud<ShapeContext1980>::Ptr features(new PointCloud<ShapeContext1980>);
    ShapeContext3DEstimation<PointRGB, Normal, ShapeContext1980> desc;
    desc.setSearchSurface(cloud);
    desc.setInputCloud(keypoints);
    estimateNormals(keypoints, cloud, keypoint_normals);
    desc.setInputNormals(keypoint_normals);
    search::KdTree<PointRGB>::Ptr kdtree(new search::KdTree<PointRGB>);
    desc.setSearchMethod(kdtree);
    desc.setRadiusSearch(0.05);
    desc.setMinimalRadius(0.05 / 10.0);
    desc.setPointDensityRadius(0.05 / 5.0);
    desc.compute(*features);
  }
  else if (desc_type_ == DESC_USC)
  {

  }
  else if (desc_type_ == DESC_BOARD)
  {

  }
  else if (desc_type_ == DESC_BOUNDARY)
  {

  }
  else if (desc_type_ == DESC_INT_GRAD)
  {

  }
  else if (desc_type_ == DESC_INT_SPIN)
  {

  }
  else if (desc_type_ == DESC_RIB)
  {

  }
  else if (desc_type_ == DESC_SPIN_IMAGE)
  {

  }
  else if (desc_type_ == DESC_MOMENT_INV)
  {

  }
  else if (desc_type_ == DESC_CRH)
  {

  }
  else if (desc_type_ == DESC_DIFF_OF_NORM)
  {


  }
  else if (desc_type_ == DESC_ESF)
  {
    PointCloud<ESFSignature640>::Ptr features(new PointCloud<ESFSignature640>);
    ESFEstimation<PointRGB, ESFSignature640> desc;
    desc.setSearchSurface(cloud);
    desc.setInputCloud(keypoints);
    search::KdTree<PointRGB>::Ptr kdtree(new search::KdTree<PointRGB>);
    desc.setSearchMethod(kdtree);
    desc.setRadiusSearch(0.08);
    desc.compute(*features);
  }
  else if (desc_type_ == DESC_FPFH)
  {
    PointCloud<FPFHSignature33>::Ptr features(new PointCloud<FPFHSignature33>);
    FPFHEstimation<PointRGB, Normal, FPFHSignature33> desc;
    desc.setSearchSurface(cloud);
    desc.setInputCloud(keypoints);
    estimateNormals(keypoints, cloud, keypoint_normals);
    desc.setInputNormals(keypoint_normals);
    search::KdTree<PointRGB>::Ptr kdtree(new search::KdTree<PointRGB>);
    desc.setSearchMethod(kdtree);
    desc.setRadiusSearch(0.08);
    desc.compute(*features);
  }
  else if (desc_type_ == DESC_NARF)
  {
    PointCloud<Narf36>::Ptr features(new PointCloud<Narf36>);
    PointCloud<Narf36>::Ptr descriptors(new PointCloud<Narf36>);
    RangeImagePlanar range_image;
    convertToRangeImage(cloud, range_image);
    NarfDescriptor desc(&range_image, &keypoints);
    desc.getParameters().support_size = 0.2f;
    desc.getParameters().rotation_invariant = true;
    desc.compute(*features);
  }
  else if (desc_type_ == DESC_VFH)
  {
    PointCloud<VFHSignature308>::Ptr features(new PointCloud<VFHSignature308>);
    VFHEstimation<PointRGB, Normal, VFHSignature308> desc;
    desc.setSearchSurface(cloud);
    desc.setInputCloud(keypoints);
    estimateNormals(keypoints, cloud, keypoint_normals);
    desc.setInputNormals(keypoint_normals);
    search::KdTree<PointRGB>::Ptr kdtree(new search::KdTree<PointRGB>);
    desc.setSearchMethod(kdtree);
    desc.setRadiusSearch(0.08);
    desc.compute(*features);
  }
  else if (desc_type_ == DESC_CVFH)
  {
    PointCloud<VFHSignature308>::Ptr features(new PointCloud<VFHSignature308>);
    CVFHEstimation<PointRGB, Normal, VFHSignature308> desc;
    desc.setSearchSurface(cloud);
    desc.setInputCloud(keypoints);
    estimateNormals(keypoints, cloud, keypoint_normals);
    desc.setInputNormals(keypoint_normals);
    search::KdTree<PointRGB>::Ptr kdtree(new search::KdTree<PointRGB>);
    desc.setSearchMethod(kdtree);
    desc.setRadiusSearch(0.08);
    desc.compute(*features);
  }
  else if (desc_type_ == DESC_PFH)
  {
    PointCloud<PFHSignature125>::Ptr features(new PointCloud<PFHSignature125>);
    PFHEstimation<PointRGB, Normal, PFHSignature125> desc;
    desc.setSearchSurface(cloud);
    desc.setInputCloud(keypoints);
    estimateNormals(keypoints, cloud, keypoint_normals);
    desc.setInputNormals(keypoint_normals);
    search::KdTree<PointRGB>::Ptr kdtree(new search::KdTree<PointRGB>);
    desc.setSearchMethod(kdtree);
    desc.setRadiusSearch(0.08);
    desc.compute(*features);
  }
  else if (desc_type_ == DESC_PPAL_CURV)
  {
    // Convert to xyz
    PointCloudXYZ::Ptr cloud_xyz(new PointCloudXYZ);
    copyPointCloud(*cloud, *cloud_xyz);
    PointCloudXYZ::Ptr keypoints_xyz(new PointCloudXYZ);
    copyPointCloud(*keypoints, *keypoints_xyz);

    // Estimate features
    PointCloud<PrincipalCurvatures>::Ptr features(new PointCloud<PrincipalCurvatures>);
    PrincipalCurvaturesEstimation<PointXYZ, Normal, PrincipalCurvatures> desc;
    desc.setSearchSurface(cloud_xyz);
    desc.setInputCloud(keypoints_xyz);
    estimateNormals(keypoints, cloud, keypoint_normals);
    desc.setInputNormals(keypoint_normals);
    search::KdTree<PointXYZ>::Ptr kdtree(new search::KdTree<PointXYZ>);
    desc.setSearchMethod(kdtree);
    desc.setRadiusSearch(0.08);
    desc.compute(*features);
  }
  else if (desc_type_ == DESC_RIFT)
  {
    // Compute gradients
    PointCloud<PointXYZI>::Ptr keypoint_intensities(new PointCloud<PointXYZI>);
    PointCloud<PointXYZI>::Ptr cloud_intensities(new PointCloud<PointXYZI>);
    PointCloud<IntensityGradient>::Ptr keypoint_gradients(new PointCloud<IntensityGradient>);
    estimateNormals(keypoints, cloud, keypoint_normals);
    PointCloudXYZRGBtoXYZI(*keypoints, *keypoint_intensities);
    PointCloudXYZRGBtoXYZI(*cloud, *cloud_intensities);
    computeGradient(keypoint_intensities, keypoint_normals, keypoint_gradients);

    // Estimate features
    PointCloud< Histogram<32> >::Ptr features(new PointCloud< Histogram<32> >);
    RIFTEstimation< PointXYZI, IntensityGradient, Histogram<32> > desc;
    desc.setInputCloud(keypoint_intensities);
    desc.setSearchSurface(cloud_intensities);
    search::KdTree<PointXYZI>::Ptr kdtree(new search::KdTree<PointXYZI>);
    desc.setSearchMethod(kdtree);
    desc.setInputGradient(keypoint_gradients);
    desc.setRadiusSearch(0.08);
    desc.setNrDistanceBins(4);
    desc.setNrGradientBins(8);
    desc.compute(*features);
  }
  else if (desc_type_ == DESC_SHOT)
  {
    // Convert to xyz
    PointCloudXYZ::Ptr cloud_xyz(new PointCloudXYZ);
    copyPointCloud(*cloud, *cloud_xyz);
    PointCloudXYZ::Ptr keypoints_xyz(new PointCloudXYZ);
    copyPointCloud(*keypoints, *keypoints_xyz);

    // Estimate features
    PointCloud<SHOT352>::Ptr features(new PointCloud<SHOT352>);
    SHOTEstimationOMP<PointXYZ, Normal, SHOT352> desc;
    desc.setSearchSurface(cloud_xyz);
    desc.setInputCloud(keypoints_xyz);
    estimateNormals(keypoints, cloud, keypoint_normals);
    desc.setInputNormals(keypoint_normals);
    search::KdTree<PointXYZ>::Ptr kdtree(new search::KdTree<PointXYZ>);
    desc.setSearchMethod(kdtree);
    desc.setRadiusSearch(0.08);
    desc.compute(*features);
  }
  else if (desc_type_ == DESC_SHOT_COLOR)
  {
    PointCloud<SHOT1344>::Ptr features(new PointCloud<SHOT1344>);
    SHOTColorEstimationOMP<PointRGB, Normal, SHOT1344> desc;
    desc.setSearchSurface(cloud);
    desc.setInputCloud(keypoints);
    estimateNormals(keypoints, cloud, keypoint_normals);
    desc.setInputNormals(keypoint_normals);
    search::KdTree<PointRGB>::Ptr kdtree(new search::KdTree<PointRGB>);
    desc.setSearchMethod(kdtree);
    desc.setRadiusSearch(0.08);
    desc.compute(*features);
  }
  else if (desc_type_ == DESC_SHOT_LRF)
  {
    // Convert to xyz
    PointCloudXYZ::Ptr cloud_xyz(new PointCloudXYZ);
    copyPointCloud(*cloud, *cloud_xyz);
    PointCloudXYZ::Ptr keypoints_xyz(new PointCloudXYZ);
    copyPointCloud(*keypoints, *keypoints_xyz);

    // Estimate features
    PointCloud<SHOT352>::Ptr features(new PointCloud<SHOT352>);
    SHOTLocalReferenceFrameEstimationOMP<PointXYZ, SHOT352> desc;
    desc.setSearchSurface(cloud_xyz);
    desc.setInputCloud(keypoints_xyz);
    search::KdTree<PointXYZ>::Ptr kdtree(new search::KdTree<PointXYZ>);
    desc.setSearchMethod(kdtree);
    desc.setRadiusSearch(0.08);
    desc.compute(*features);
  }
  else
  {
    ROS_WARN("Descriptor type unavailable");
  }
}

/** \brief Normal estimation
  * @return
  * \param Cloud where normals will be estimated
  * \param Cloud surface with additional information to estimate the features for every point in the input dataset
  * \param Output cloud with normals
  */
template<typename FeatureType>
void Features<FeatureType>::estimateNormals(const PointCloudRGB::Ptr& cloud,
                                            const PointCloudRGB::Ptr& surface,
                                            PointCloud<Normal>::Ptr& normals)
{
  NormalEstimation<PointRGB, Normal> normalEstimation;
  normalEstimation.setInputCloud(cloud);
  normalEstimation.setSearchSurface(surface);
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
template<typename FeatureType>
void Features<FeatureType>::computeGradient(const PointCloud<PointXYZI>::Ptr& intensity,
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
template<typename FeatureType>
void Features<FeatureType>::convertToRangeImage(const PointCloudRGB::Ptr& cloud,
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

/** \brief Find correspondences between features
  * @return
  * \param Source cloud features
  * \param Target cloud features
  * \param Vector of correspondences
  */
template<typename FeatureType>
void Features<FeatureType>::findCorrespondences(typename PointCloud<FeatureType>::Ptr source,
                                                typename PointCloud<FeatureType>::Ptr target,
                                                CorrespondencesPtr correspondences)
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

/** \brief Filter the correspondences by RANSAC
  * @return
  * \param Source cloud features
  * \param Target cloud features
  * \param Vector of correspondences
  * \param Output filtered vector of correspondences
  */
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


#endif  // FEATURES_H
