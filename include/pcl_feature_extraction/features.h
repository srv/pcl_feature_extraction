/// Copyright 2015 Miquel Massot Campos
/// Systems, Robotics and Vision
/// University of the Balearic Islands
/// All rights reserved.

#ifndef FEATURES_H
#define FEATURES_H

#include <pcl_feature_extraction/tools.h>

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
#include <pcl/kdtree/impl/kdtree_flann.hpp>

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
using namespace registration;

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
  Features() {}
  Features(typename Feature<PointXYZRGB, FeatureType>::Ptr feature_extractor);
  Features(typename Feature<PointXYZRGB, FeatureType>::Ptr feature_extractor,
           const double feat_radius_search,
           const double normal_radius_search);

  // Feature computation
  void compute(const PointCloudRGB::Ptr cloud,
               const PointCloudRGB::Ptr keypoints,
               typename PointCloud<FeatureType>::Ptr& descriptors);

  // Search for correspondences
  void findCorrespondences(typename PointCloud<FeatureType>::Ptr source,
                           typename PointCloud<FeatureType>::Ptr target,
                           CorrespondencesPtr& correspondences);

  // Correspondence filtering
  void filterCorrespondences(const PointCloudRGB::Ptr source,
                             const PointCloudRGB::Ptr target,
                             const CorrespondencesPtr correspondences,
                             CorrespondencesPtr& filtered_correspondences);

  // Set common feature properties
  void setFeatureRadiusSearch(double radius_search);
  void setNormalRadiusSearch(double radius_search);

 private:
  typename Feature<PointXYZRGB, FeatureType>::Ptr feature_extractor_;
  double feat_radius_search_;
  double normal_radius_search_;
};


/** \brief Class constructor. Initialize the class
  */
template<typename FeatureType>
Features<FeatureType>::Features(typename Feature<PointXYZRGB, FeatureType>::Ptr feature_extractor)
  : feature_extractor_(feature_extractor)
{
  feat_radius_search_ = 0.08;
  normal_radius_search_ = 0.05;
}
  template<typename FeatureType>
Features<FeatureType>::Features(typename Feature<PointXYZRGB, FeatureType>::Ptr feature_extractor,
                                const double feat_radius_search,
                                const double normal_radius_search) :
    feature_extractor_(feature_extractor),
    feat_radius_search_(feat_radius_search),
    normal_radius_search_(normal_radius_search) {}


/** \brief Compute descriptors
  * @return
  * \param Input cloud
  * \param Input keypoints, where features will be computed
  * \param Output descriptors
  */
template<typename FeatureType>
void Features<FeatureType>::compute(const PointCloudRGB::Ptr cloud,
                                    const PointCloudRGB::Ptr keypoints,
                                    typename PointCloud<FeatureType>::Ptr& descriptors)
{

  typename FeatureFromNormals<PointXYZRGB, Normal, FeatureType>::Ptr feature_from_normals =
    boost::dynamic_pointer_cast<FeatureFromNormals<PointXYZRGB, Normal, FeatureType> >(feature_extractor_);

  if(feature_from_normals)
  {
    typename PointCloud<Normal>::Ptr normals (new PointCloud<Normal>);
    Tools::estimateNormals(cloud, normals, normal_radius_search_);
    feature_from_normals->setInputNormals(normals);
  }
  feature_extractor_->setSearchSurface(cloud);
  feature_extractor_->setInputCloud(keypoints);
  search::KdTree<PointRGB>::Ptr kdtree(new search::KdTree<PointRGB>);
  feature_extractor_->setSearchMethod(kdtree);
  feature_extractor_->setRadiusSearch(feat_radius_search_);
  feature_extractor_->compute(*descriptors);
}

/** \brief Sets the feature radius search
  * @return
  * \param Radius search
  */
template<typename FeatureType>
void Features<FeatureType>::setFeatureRadiusSearch(double radius_search)
{
  feat_radius_search_ = radius_search;
}

/** \brief Sets the normals radius search
  * @return
  * \param Radius search
  */
template<typename FeatureType>
void Features<FeatureType>::setNormalRadiusSearch(double radius_search)
{
  normal_radius_search_ = radius_search;
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
                                                CorrespondencesPtr& correspondences)
{
  vector<int> source2target;
  vector<int> target2source;
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
void Features<FeatureType>::filterCorrespondences(const PointCloudRGB::Ptr source,
                                                  const PointCloudRGB::Ptr target,
                                                  CorrespondencesPtr correspondences,
                                                  CorrespondencesPtr& filtered_correspondences)
{
  registration::CorrespondenceRejectorSampleConsensus<PointRGB> rejector;
  rejector.setInputSource(source);
  rejector.setInputTarget(target);
  rejector.setInputCorrespondences(correspondences);
  rejector.getCorrespondences(*filtered_correspondences);
}

#endif  // FEATURES_H
