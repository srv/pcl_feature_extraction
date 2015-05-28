#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <ros/ros.h>
#include <boost/filesystem.hpp>
#include <boost/assign/list_inserter.hpp>

// Custom
#include "pcl_feature_extraction/keypoints.h"

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


class PclFeaturesEvaluation
{

private:

  // Node handles
  ros::NodeHandle nh_;
  ros::NodeHandle nhp_;

  // Pointcloud files
  string cloud_filename_1_;
  string cloud_filename_2_;

  // Pointclouds
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

  /** \brief Read the pointclouds
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

  /** \brief Performs the evaluation
    */
  void evaluate()
  {
    // Step 1: create the keypoints<->descriptors combinations


    // Step 2: loop over the combinations
    uint size = sizeof(keypoints_list)/sizeof(keypoints_list[0]);
    for (uint i=0; i<size; i++)
    {
      // string kp_type = comb_[i].first;
      // string desc_type = comb_[i].second;
      ROS_INFO("#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#");
      ROS_INFO_STREAM("Evaluating keypoint: " << keypoints_list[i]);

      // Extract the keypoints
      ros::WallTime start = ros::WallTime::now();
      Keypoints kp(keypoints_list[i]);
      PointCloudRGB::Ptr keypoints(new PointCloudRGB);
      kp.compute(cloud_1_, keypoints);
      ros::WallDuration runtime = ros::WallTime::now() - start;

      ROS_INFO_STREAM("Done! Total cloud points: " << cloud_1_->points.size() <<
                      ".    Number of keypoints: " << keypoints->points.size() <<
                      ".    Runtime: " << runtime.toSec() << "\n");
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
