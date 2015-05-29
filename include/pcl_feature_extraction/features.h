/// Copyright 2015 Miquel Massot Campos
/// Systems, Robotics and Vision
/// University of the Balearic Islands
/// All rights reserved.

#ifndef FEATURES_H
#define FEATURES_H

// Generic pcl
#include <pcl/common/common.h>
#include <pcl/point_types.h>
#include <pcl/registration/correspondence_types.h>
#include <pcl/range_image/range_image_planar.h>

#include <vector>
#include <string>

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
///  - Radius-based Surface Descriptor (RSD)
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
static const string DESC_RSD           = "RSD";
static const string DESC_SHOT          = "SHOT";
static const string DESC_SHOT_COLOR    = "SHOTColore";
static const string DESC_SHOT_LRF      = "SHOTLocalReferenceFrame";

typedef PointXYZRGB PointRGB;
typedef PointCloud<PointXYZ> PointCloudXYZ;
typedef PointCloud<PointRGB> PointCloudRGB;

class Features
{
 public:

  // Class constructor
  explicit Features(const std::string& descriptor_type);

  // Feature computation
  void compute(const PointCloudRGB& cloud, const PointCloudRGB& keypoints) const;

  // Search for correspondences
  template<typename FeatureType>
  void findCorrespondences(typename PointCloud<FeatureType>::Ptr source,
                           typename PointCloud<FeatureType>::Ptr target,
                           CorrespondencesPtr correspondences);

  // Correspondence filtering
  template<typename FeatureType>
  void filterCorrespondences(typename PointCloud<FeatureType>::Ptr source,
                             typename PointCloud<FeatureType>::Ptr target,
                             CorrespondencesPtr correspondences,
                             CorrespondencesPtr filtered_correspondences);

 protected:

  // Normal estimation
  void estimateNormals(const PointCloudRGB::Ptr& cloud,
                       PointCloud<Normal>::Ptr& normals);

  // Compute the intensity gradients
  void computeGradient(const PointCloud<PointXYZI>::Ptr& intensity,
                       const PointCloud<Normal>::Ptr& normals,
                       PointCloud<IntensityGradient>::Ptr& gradients);

  // Converts a pointcloud to a range image
  void convertToRangeImage(const PointCloudRGB::Ptr& cloud,
                           RangeImagePlanar& range_image);

 private:

  string descriptor_type_;  //!> Stores the keypoint type
};

#endif  // FEATURES_H
