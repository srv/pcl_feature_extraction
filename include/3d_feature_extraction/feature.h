/// Copyright 2015 Miquel Massot Campos
/// Systems, Robotics and Vision
/// University of the Balearic Islands
/// All rights reserved.

#ifndef FEATURE_H
#define FEATURE_H

#include <vector>
#include <string>

namespace 3d_feature_extraction {
class Feature {
 public:
  explicit Feature(const std::string& descriptor_type);
  void compute(const PointCloudRGB& points, const std::vector<int> indexes);
 private:
  std::string descriptor_type_;
};
}

#endif  // FEATURE_H
