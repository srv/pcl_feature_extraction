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
string keypoints_list[] = {KP_AGAST_DETECTOR_7_12s,
                           KP_AGAST_DETECTOR_5_8,
                           KP_OAST_DETECTOR_9_16,
                           KP_HARRIS_3D,
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
                             DESC_NARF,
                             DESC_VFH,
                             DESC_CVFH,
                             DESC_PFH,
                             DESC_PPAL_CURV,
                             DESC_RIFT,
                             DESC_SHOT,
                             DESC_SHOT_COLOR,
                             DESC_SHOT_LRF};

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
      Keypoints kp(kp_type);
      ros::WallTime kp_start = ros::WallTime::now();
      PointCloudRGB::Ptr keypoints(new PointCloudRGB);
      kp.compute(cloud_1_, keypoints);
      ros::WallDuration kp_runtime = ros::WallTime::now() - kp_start;

      // Extract the features
      ros::WallTime desc_start = ros::WallTime::now();
      if (desc_type == DESC_SHAPE_CONTEXT)
      {
        PointCloud<ShapeContext1980>::Ptr features(new PointCloud<ShapeContext1980>);
        Features<ShapeContext1980> feat(desc_type);
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
        PointCloud<ESFSignature640>::Ptr features(new PointCloud<ESFSignature640>);
        Features<ESFSignature640> feat(desc_type);
      }
      else if (desc_type == DESC_FPFH)
      {
        PointCloud<FPFHSignature33>::Ptr features(new PointCloud<FPFHSignature33>);
        Features<FPFHSignature33> feat(desc_type);
      }
      else if (desc_type == DESC_NARF)
      {
        PointCloud<Narf36>::Ptr features(new PointCloud<Narf36>);
        Features<Narf36> feat(desc_type);
      }
      else if (desc_type == DESC_VFH)
      {
        PointCloud<VFHSignature308>::Ptr features(new PointCloud<VFHSignature308>);
        Features<VFHSignature308> feat(desc_type);
      }
      else if (desc_type == DESC_CVFH)
      {
        PointCloud<VFHSignature308>::Ptr features(new PointCloud<VFHSignature308>);
        Features<VFHSignature308> feat(desc_type);
      }
      else if (desc_type == DESC_PFH)
      {
        PointCloud<PFHSignature125>::Ptr features(new PointCloud<PFHSignature125>);
        Features<PFHSignature125> feat(desc_type);
      }
      else if (desc_type == DESC_PPAL_CURV)
      {
        PointCloud<PrincipalCurvatures>::Ptr features(new PointCloud<PrincipalCurvatures>);
        Features<PrincipalCurvatures> feat(desc_type);
      }
      else if (desc_type == DESC_RIFT)
      {
        PointCloud< Histogram<32> >::Ptr features(new PointCloud< Histogram<32> >);
        Features< Histogram<32> > feat(desc_type);
      }
      else if (desc_type == DESC_SHOT)
      {
        PointCloud<SHOT352>::Ptr features(new PointCloud<SHOT352>);
        Features<SHOT352> feat(desc_type);
      }
      else if (desc_type == DESC_SHOT_COLOR)
      {
        PointCloud<SHOT1344>::Ptr features(new PointCloud<SHOT1344>);
        Features<SHOT1344> feat(desc_type);
      }
      else if (desc_type == DESC_SHOT_LRF)
      {
        PointCloud<SHOT352>::Ptr features(new PointCloud<SHOT352>);
        Features<SHOT352> feat(desc_type);
      }
      ros::WallDuration desc_runtime = ros::WallTime::now() - desc_start;

      ROS_INFO_STREAM("Done! Total cloud points: " << cloud_1_->points.size() <<
                      ".    Number of keypoints: " << keypoints->points.size() <<
                      ".    Runtime: " << kp_runtime.toSec() << "\n");
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

