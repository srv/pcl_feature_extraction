#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <ros/ros.h>
#include <boost/filesystem.hpp>
#include <boost/assign/list_inserter.hpp>

// Custom
#include "pcl_feature_extraction/keypoints.h"
#include "pcl_feature_extraction/features.h"

// Generic pcl
#include <pcl/common/common.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>

using namespace pcl;
using namespace std;
using namespace boost;
namespace fs=filesystem;

// pcl definition
typedef PointXYZRGB                                PointRGB;
typedef PointCloud<PointRGB>                       PointCloudRGB;
typedef IterativeClosestPoint<PointRGB, PointRGB>  IterativeClosestPoint;

// Stop handler binding
boost::function<void(int)> stopHandlerCb;

/** \brief Catches the Ctrl+C signal.
  */
void stopHandler(int s)
{
  printf("Caught signal %d\n",s);
  stopHandlerCb(s);
  ros::shutdown();
}

// Define the list of keypoints and descriptors
// string keypoints_list[] = {KP_AGAST_DETECTOR_7_12s,
//                            KP_AGAST_DETECTOR_5_8,
//                            KP_OAST_DETECTOR_9_16,
//                            KP_HARRIS_3D,
//                            KP_HARRIS_6D,
//                            KP_ISS,
//                            KP_NARF,
//                            KP_SIFT,
//                            KP_SUSAN,
//                            KP_UNIFORM_SAMPLING};

string keypoints_list[] = {KP_HARRIS_3D,
                           KP_HARRIS_6D,
                           KP_ISS,
                           KP_NARF,
                           KP_SIFT,
                           KP_SUSAN,
                           KP_UNIFORM_SAMPLING};

// string descriptors_list[] = {DESC_SHAPE_CONTEXT,
//                              DESC_USC,
//                              DESC_BOARD,
//                              DESC_BOUNDARY,
//                              DESC_INT_GRAD,
//                              DESC_INT_SPIN,
//                              DESC_RIB,
//                              DESC_SPIN_IMAGE,
//                              DESC_MOMENT_INV,
//                              DESC_CRH,
//                              DESC_DIFF_OF_NORM,
//                              DESC_ESF,
//                              DESC_FPFH,
//                              DESC_NARF,
//                              DESC_VFH,
//                              DESC_CVFH,
//                              DESC_PFH,
//                              DESC_PPAL_CURV,
//                              DESC_RIFT,
//                              DESC_SHOT,
//                              DESC_SHOT_COLOR,
//                              DESC_SHOT_LRF};

string descriptors_list[] = {DESC_SHAPE_CONTEXT,
                             DESC_ESF,
                             DESC_FPFH,
                             DESC_VFH,
                             DESC_CVFH,
                             DESC_PFH,
                             DESC_PPAL_CURV,
                             DESC_SHOT,
                             DESC_SHOT_COLOR};

class PclFeaturesEvaluation
{

private:

  // Node handles
  ros::NodeHandle nh_;
  ros::NodeHandle nhp_;

  // PointCloud files
  string cloud_filename_1_;
  string cloud_filename_2_;

  // PointClouds
  PointCloudRGB::Ptr cloud_1_;
  PointCloudRGB::Ptr cloud_2_;

  // Common parameters
  double feat_radius_search_;
  double normal_radius_search_;

  // Keypoints and descriptors combinations
  vector< pair<string,string> > comb_;


public:

  /** \brief Class constructor
    */
  PclFeaturesEvaluation() : nh_(), nhp_("~"), cloud_1_(new PointCloudRGB), cloud_2_(new PointCloudRGB)
  {
    // Bind the finalize member to stopHandler signal
    stopHandlerCb = std::bind1st(std::mem_fun(&PclFeaturesEvaluation::finalize), this);

    // Read parameters
    readParameters();

    // Read clouds
    readClouds();
  }

  /** \brief Finalizes the node
    */
  void finalize(int s)
  {
    ROS_INFO("[PclFeaturesEvaluation:] Finalizing...");
  }

  /** \brief Read the parameters
    */
  void readParameters()
  {
    // Directories
    nhp_.param("cloud_filename_1", cloud_filename_1_, string(""));
    nhp_.param("cloud_filename_2", cloud_filename_2_, string(""));

    // Parameters
    nhp_.param("feat_radius_search", feat_radius_search_, 0.08);
    nhp_.param("normal_radius_search", normal_radius_search_, 0.05);
  }

  /** \brief Read the PointClouds
    */
  void readClouds()
  {
    if (pcl::io::loadPCDFile<PointRGB>(cloud_filename_1_, *cloud_1_) == -1)
    {
      ROS_WARN_STREAM("[PclFeaturesEvaluation:] Couldn't read the file: " << cloud_filename_1_);
      return;
    }
    if (pcl::io::loadPCDFile<PointRGB>(cloud_filename_2_, *cloud_2_) == -1)
    {
      ROS_WARN_STREAM("[PclFeaturesEvaluation:] Couldn't read the file: " << cloud_filename_2_);
      return;
    }
  }

  /** \brief Creates a list keypoint/descriptor combinations
    */
  void createCombinations()
  {
    uint kp_size = sizeof(keypoints_list)/sizeof(keypoints_list[0]);
    uint desc_size = sizeof(descriptors_list)/sizeof(descriptors_list[0]);
    for (uint i=0; i<kp_size; i++)
    {
      for (uint j=0; j<desc_size; j++)
        comb_.push_back(make_pair(keypoints_list[i], descriptors_list[j]));
    }
  }

  /** \brief Performs the evaluation
    */
  void evaluate()
  {
    // Step 1: create the keypoints/descriptors combinations
    createCombinations();

    // Step 2: loop over the combinations
    for (uint i=0; i<comb_.size(); i++)
    {
      string kp_type = comb_[i].first;
      string desc_type = comb_[i].second;

      ROS_INFO("#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#");
      ROS_INFO_STREAM("Evaluating: " << kp_type << " / " << desc_type);

      // Extract the keypoints
      Keypoints kp(kp_type, normal_radius_search_);
      ros::WallTime kp_start = ros::WallTime::now();
      PointCloudRGB::Ptr keypoints_1(new PointCloudRGB);
      PointCloudRGB::Ptr keypoints_2(new PointCloudRGB);
      kp.compute(cloud_1_, keypoints_1);
      kp.compute(cloud_2_, keypoints_2);

      // Sanity check
      if (keypoints_1->points.size() == 0 || keypoints_2->points.size() == 0 )
      {
        ROS_WARN("No keypoints, skipping...");
        continue;
      }

      // Log
      ros::WallDuration kp_runtime = ros::WallTime::now() - kp_start;
      ROS_INFO_STREAM("Total cloud points: " << cloud_1_->points.size() <<
                      ".    Number of keypoints: " << keypoints_1->points.size() <<
                      ".    Runtime: " << kp_runtime.toSec() << "\n");

      // Extract the features
      ros::WallTime desc_start = ros::WallTime::now();
      if (desc_type == DESC_SHAPE_CONTEXT)
      {
        ShapeContext3DEstimation<PointXYZRGB, Normal, ShapeContext1980>::Ptr feature_extractor_orig(
          new ShapeContext3DEstimation<PointXYZRGB, Normal, ShapeContext1980>);

        // Set properties
        feature_extractor_orig->setMinimalRadius(feat_radius_search_ / 10.0);
        feature_extractor_orig->setPointDensityRadius(feat_radius_search_ / 5.0);

        // Compute features
        Feature<PointXYZRGB, ShapeContext1980>::Ptr feature_extractor(feature_extractor_orig);
        PointCloud<ShapeContext1980>::Ptr features_1(new PointCloud<ShapeContext1980>);
        PointCloud<ShapeContext1980>::Ptr features_2(new PointCloud<ShapeContext1980>);
        Features<ShapeContext1980> feat(feature_extractor, feat_radius_search_, normal_radius_search_);
        feat.compute(cloud_1_, keypoints_1, features_1);
        feat.compute(cloud_2_, keypoints_2, features_2);

        ROS_INFO_STREAM(".    Number of features: " << features_1->points.size());
      }
      else if (desc_type == DESC_USC)
      {
      }
      else if (desc_type == DESC_BOARD)
      {
      }
      else if (desc_type == DESC_BOUNDARY)
      {
      }
      else if (desc_type == DESC_INT_GRAD)
      {
      }
      else if (desc_type == DESC_INT_SPIN)
      {
      }
      else if (desc_type == DESC_RIB)
      {
      }
      else if (desc_type == DESC_SPIN_IMAGE)
      {
      }
      else if (desc_type == DESC_MOMENT_INV)
      {
      }
      else if (desc_type == DESC_CRH)
      {
      }
      else if (desc_type == DESC_DIFF_OF_NORM)
      {
      }
      else if (desc_type == DESC_ESF)
      {
        Feature<PointXYZRGB, ESFSignature640>::Ptr feature_extractor(new ESFEstimation<PointXYZRGB, ESFSignature640>);
        PointCloud<ESFSignature640>::Ptr features_1(new PointCloud<ESFSignature640>);
        PointCloud<ESFSignature640>::Ptr features_2(new PointCloud<ESFSignature640>);
        Features<ESFSignature640> feat(feature_extractor, feat_radius_search_, normal_radius_search_);
        feat.compute(cloud_1_, keypoints_1, features_1);
        feat.compute(cloud_2_, keypoints_2, features_2);

        ROS_INFO_STREAM(".    Number of features: " << features_1->points.size());
      }
      else if (desc_type == DESC_FPFH)
      {
        Feature<PointXYZRGB, FPFHSignature33>::Ptr feature_extractor(new FPFHEstimation<PointXYZRGB, Normal, FPFHSignature33>);
        PointCloud<FPFHSignature33>::Ptr features_1(new PointCloud<FPFHSignature33>);
        PointCloud<FPFHSignature33>::Ptr features_2(new PointCloud<FPFHSignature33>);
        Features<FPFHSignature33> feat(feature_extractor, feat_radius_search_, normal_radius_search_);
        feat.compute(cloud_1_, keypoints_1, features_1);
        feat.compute(cloud_2_, keypoints_2, features_2);

        ROS_INFO_STREAM(".    Number of features: " << features_1->points.size());
      }
      else if (desc_type == DESC_NARF)
      {
        // Feature<PointXYZRGB, Narf36>::Ptr feature_extractor(new NarfDescriptor<PointXYZRGB, Narf36>);
        // PointCloud<Narf36>::Ptr features_1(new PointCloud<Narf36>);
        // PointCloud<Narf36>::Ptr features_2(new PointCloud<Narf36>);
        // Features<Narf36> feat(feature_extractor, feat_radius_search_, normal_radius_search_);
        // feat.compute(cloud_1_, keypoints_1, features_1);
        // feat.compute(cloud_2_, keypoints_2, features_2);
      }
      else if (desc_type == DESC_VFH)
      {
        Feature<PointXYZRGB, VFHSignature308>::Ptr feature_extractor(new VFHEstimation<PointXYZRGB, Normal, VFHSignature308>);
        PointCloud<VFHSignature308>::Ptr features_1(new PointCloud<VFHSignature308>);
        PointCloud<VFHSignature308>::Ptr features_2(new PointCloud<VFHSignature308>);
        Features<VFHSignature308> feat(feature_extractor, feat_radius_search_, normal_radius_search_);
        feat.compute(cloud_1_, keypoints_1, features_1);
        feat.compute(cloud_2_, keypoints_2, features_2);

        ROS_INFO_STREAM(".    Number of features: " << features_1->points.size());
      }
      else if (desc_type == DESC_CVFH)
      {
        Feature<PointXYZRGB, VFHSignature308>::Ptr feature_extractor(new CVFHEstimation<PointXYZRGB, Normal, VFHSignature308>);
        PointCloud<VFHSignature308>::Ptr features_1(new PointCloud<VFHSignature308>);
        PointCloud<VFHSignature308>::Ptr features_2(new PointCloud<VFHSignature308>);
        Features<VFHSignature308> feat(feature_extractor, feat_radius_search_, normal_radius_search_);
        feat.compute(cloud_1_, keypoints_1, features_1);
        feat.compute(cloud_2_, keypoints_2, features_2);

        ROS_INFO_STREAM(".    Number of features: " << features_1->points.size());
      }
      else if (desc_type == DESC_PFH)
      {
        Feature<PointXYZRGB, PFHSignature125>::Ptr feature_extractor(new PFHEstimation<PointXYZRGB, Normal, PFHSignature125>);
        PointCloud<PFHSignature125>::Ptr features_1(new PointCloud<PFHSignature125>);
        PointCloud<PFHSignature125>::Ptr features_2(new PointCloud<PFHSignature125>);
        Features<PFHSignature125> feat(feature_extractor, feat_radius_search_, normal_radius_search_);
        feat.compute(cloud_1_, keypoints_1, features_1);
        feat.compute(cloud_2_, keypoints_2, features_2);

        ROS_INFO_STREAM(".    Number of features: " << features_1->points.size());
      }
      else if (desc_type == DESC_PPAL_CURV)
      {
        Feature<PointXYZRGB, PrincipalCurvatures>::Ptr feature_extractor(new PrincipalCurvaturesEstimation<PointXYZRGB, Normal, PrincipalCurvatures>);
        PointCloud<PrincipalCurvatures>::Ptr features_1(new PointCloud<PrincipalCurvatures>);
        PointCloud<PrincipalCurvatures>::Ptr features_2(new PointCloud<PrincipalCurvatures>);
        Features<PrincipalCurvatures> feat(feature_extractor, feat_radius_search_, normal_radius_search_);
        feat.compute(cloud_1_, keypoints_1, features_1);
        feat.compute(cloud_2_, keypoints_2, features_2);

        ROS_INFO_STREAM(".    Number of features: " << features_1->points.size());
      }
      else if (desc_type == DESC_RIFT)
      {
        // Feature<PointXYZRGB, Histogram<32> >::Ptr feature_extractor(new RIFTEstimation<PointXYZRGB, Normal, Histogram<32> >);
        // PointCloud< Histogram<32> >::Ptr features_1(new PointCloud< Histogram<32> >);
        // PointCloud< Histogram<32> >::Ptr features_2(new PointCloud< Histogram<32> >);
        // Features< Histogram<32> > feat(feature_extractor, feat_radius_search_, normal_radius_search_);
        // feat.compute(cloud_1_, keypoints_1, features_1);
        // feat.compute(cloud_2_, keypoints_2, features_2);
      }
      else if (desc_type == DESC_SHOT)
      {
        Feature<PointXYZRGB, SHOT352>::Ptr feature_extractor(new SHOTEstimationOMP<PointXYZRGB, Normal, SHOT352>);
        PointCloud<SHOT352>::Ptr features_1(new PointCloud<SHOT352>);
        PointCloud<SHOT352>::Ptr features_2(new PointCloud<SHOT352>);
        Features<SHOT352> feat(feature_extractor, feat_radius_search_, normal_radius_search_);
        feat.compute(cloud_1_, keypoints_1, features_1);
        feat.compute(cloud_2_, keypoints_2, features_2);

        ROS_INFO_STREAM(".    Number of features: " << features_1->points.size());
      }
      else if (desc_type == DESC_SHOT_COLOR)
      {
        Feature<PointXYZRGB, SHOT1344>::Ptr feature_extractor(new SHOTColorEstimationOMP<PointXYZRGB, Normal, SHOT1344>);
        PointCloud<SHOT1344>::Ptr features_1(new PointCloud<SHOT1344>);
        PointCloud<SHOT1344>::Ptr features_2(new PointCloud<SHOT1344>);
        Features<SHOT1344> feat(feature_extractor, feat_radius_search_, normal_radius_search_);
        feat.compute(cloud_1_, keypoints_1, features_1);
        feat.compute(cloud_2_, keypoints_2, features_2);

        ROS_INFO_STREAM(".    Number of features: " << features_1->points.size());
      }
      else if (desc_type == DESC_SHOT_LRF)
      {
        // Feature<PointXYZRGB, SHOT352>::Ptr feature_extractor(new SHOTLocalReferenceFrameEstimationOMP<PointXYZRGB, SHOT352>);
        // PointCloud<SHOT352>::Ptr features_1(new PointCloud<SHOT352>);
        // PointCloud<SHOT352>::Ptr features_2(new PointCloud<SHOT352>);
        // Features<SHOT352> feat(feature_extractor, feat_radius_search_, normal_radius_search_);
        // feat.compute(cloud_1_, keypoints_1, features_1);
        // feat.compute(cloud_2_, keypoints_2, features_2);
      }

      ros::WallDuration desc_runtime = ros::WallTime::now() - desc_start;
      ROS_INFO_STREAM(".    Runtime: " << desc_runtime.toSec() << "\n");

    }
  }

};


int main(int argc, char** argv)
{
  // Start the node
  ros::init(argc, argv, "evaluation");
  PclFeaturesEvaluation node;

  // Start evaluation
  node.evaluate();

  // Exit
  return 0;
}

