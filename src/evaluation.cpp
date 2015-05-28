#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <ros/ros.h>
#include <boost/filesystem.hpp>

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
    for (uint i=0; i<comb_.size(); i++)
    {
      string kp_type = comb_[i].first;
      string desc_type = comb_[i].second;

      Keypoints kp(kp_type);
      PointCloudRGB::Ptr keypoints(new PointCloudRGB);
      kp.compute(cloud_1_, keypoints);
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

