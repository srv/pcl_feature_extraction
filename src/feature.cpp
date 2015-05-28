/// Copyright 2015 Miquel Massot Campos
/// Systems, Robotics and Vision
/// University of the Balearic Islands
/// All rights reserved.

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
///  - Feature with local reference frames (LRF)
///  - Fast Point Feature Histogram (FPFH) using OpenMP
///  - Global Fast Point Feature Histogram (GFPFH)
///  - Normal Aligned Radial Features (NARF)
///  - Normal Estimation using OpenMP (NE)
///  - Viewpoint Feature Histogram (VFH)
///  - Clustered Viewpoint Feature Histogram (CVFH)
///  - Oriented, Unique and Repeatable CVFH (OURCVFHE)
///  - Point Feature Histogram (PFH)
///  - Principal Curvatures (PC)
///  - Rotation Invariant Feature Transform (RIFT)
///  - Radius-based Surface Descriptor (RSD)
///  - Signature of Histograms of OrienTations (SHOT) using OpenMP
///  - SHOT with colour using OpenMP (SHOTColor)
///  - SHOT Local Reference Frame using OpenMP (SHOTLRF)

#include <3d_feature_extraction/feature.h>

namespace 3d_feature_extraction {

Feature::Feature(const std::string& descriptor_type)
  : descriptor_type_(descriptor_type) {
}

void Feature::compute(const PointCloudRGB& points,
                      const std::vector<int> indexes) {
  if (descriptor_type_ == "SC") {
  } else if (descriptor_type_ == "USC") {
  } else if (descriptor_type_ == "BOARD") {
  } else if (descriptor_type_ == "Boundary") {
  } else if (descriptor_type_ == "IG") {
  } else if (descriptor_type_ == "IS") {
  } else if (descriptor_type_ == "RIB") {
  } else if (descriptor_type_ == "SI") {
  } else if (descriptor_type_ == "MI") {
  } else if (descriptor_type_ == "CRH") {
  } else if (descriptor_type_ == "DoN") {
  } else if (descriptor_type_ == "ESF") {
  } else if (descriptor_type_ == "LRF") {
  } else if (descriptor_type_ == "FPFH") {
  } else if (descriptor_type_ == "GFPFH") {
  } else if (descriptor_type_ == "NARF") {
  } else if (descriptor_type_ == "NE") {
  } else if (descriptor_type_ == "VFH") {
  } else if (descriptor_type_ == "CVFH") {
  } else if (descriptor_type_ == "OURCVFHE") {
  } else if (descriptor_type_ == "PFH") {
  } else if (descriptor_type_ == "PC") {
  } else if (descriptor_type_ == "RIFT") {
  } else if (descriptor_type_ == "RSD") {
  } else if (descriptor_type_ == "SHOT") {
  } else if (descriptor_type_ == "SHOTColor") {
  } else if (descriptor_type_ == "SHOTLRF") {
  } else {
    std::cerr << "Descriptor Type Unavailable" << std::endl;
  }
}
}
