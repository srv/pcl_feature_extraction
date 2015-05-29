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

// Features pcl
#include <pcl/features/3dsc.h>

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
      if (desc_type == DESC_SHAPE_CONTEXT)
      {
        PointCloud<ShapeContext1980>::Ptr features(new PointCloud<ShapeContext1980>);
        Features<ShapeContext1980> feat(desc_type);
      }
      // else if (desc_type == DESC_USC)
      // {

      // }
      // else if (desc_type == DESC_BOARD)
      // {

      // }
      // else if (desc_type == DESC_BOUNDARY)
      // {

      // }
      // else if (desc_type == DESC_INT_GRAD)
      // {

      // }
      // else if (desc_type == DESC_INT_SPIN)
      // {

      // }
      // else if (desc_type == DESC_RIB)
      // {

      // }
      // else if (desc_type == DESC_SPIN_IMAGE)
      // {

      // }
      // else if (desc_type == DESC_MOMENT_INV)
      // {

      // }
      // else if (desc_type == DESC_CRH)
      // {

      // }
      // else if (desc_type == DESC_DIFF_OF_NORM)
      // {


      // }
      // else if (desc_type == DESC_ESF)
      // {
      //   ESFEstimation<PointRGB, ESFSignature640> desc;
      //   desc.setSearchSurface(cloud);
      //   desc.setInputCloud(keypoints);
      //   search::KdTree<PointRGB>::Ptr kdtree(new search::KdTree<PointRGB>);
      //   desc.setSearchMethod(kdtree);
      //   desc.setRadiusSearch(0.08);
      //   desc.compute(*descriptors);
      // }
      // else if (desc_type == DESC_FPFH)
      // {
      //   FPFHEstimation<PointRGB, Normal, FPFHSignature33> desc;
      //   desc.setSearchSurface(cloud);
      //   desc.setInputCloud(keypoints);
      //   estimateNormals(keypoints, cloud, keypoint_normals);
      //   desc.setInputNormals(keypoint_normals);
      //   search::KdTree<PointRGB>::Ptr kdtree(new search::KdTree<PointRGB>);
      //   desc.setSearchMethod(kdtree);
      //   desc.setRadiusSearch(0.08);
      //   desc.compute(*descriptors);
      // }
      // else if (desc_type == DESC_NARF)
      // {
      //   PointCloud<Narf36>::Ptr descriptors(new PointCloud<Narf36>);
      //   RangeImagePlanar range_image;
      //   convertToRangeImage(cloud, range_image);
      //   NarfDescriptor desc(&range_image, &keypoints);
      //   desc.getParameters().support_size = 0.2f;
      //   desc.getParameters().rotation_invariant = true;
      //   desc.compute(*descriptors);
      // }
      // else if (desc_type == DESC_VFH)
      // {
      //   VFHEstimation<PointRGB, Normal, VFHSignature308> desc;
      //   desc.setSearchSurface(cloud);
      //   desc.setInputCloud(keypoints);
      //   estimateNormals(keypoints, cloud, keypoint_normals);
      //   desc.setInputNormals(keypoint_normals);
      //   search::KdTree<PointRGB>::Ptr kdtree(new search::KdTree<PointRGB>);
      //   desc.setSearchMethod(kdtree);
      //   desc.setRadiusSearch(0.08);
      //   desc.compute(*descriptors);
      // }
      // else if (desc_type == DESC_CVFH)
      // {
      //   CVFHEstimation<PointRGB, Normal, VFHSignature308> desc;
      //   desc.setSearchSurface(cloud);
      //   desc.setInputCloud(keypoints);
      //   estimateNormals(keypoints, cloud, keypoint_normals);
      //   desc.setInputNormals(keypoint_normals);
      //   search::KdTree<PointRGB>::Ptr kdtree(new search::KdTree<PointRGB>);
      //   desc.setSearchMethod(kdtree);
      //   desc.setRadiusSearch(0.08);
      //   desc.compute(*descriptors);
      // }
      // else if (desc_type == DESC_PFH)
      // {
      //   PFHEstimation<PointRGB, Normal, PFHSignature125> desc;
      //   desc.setSearchSurface(cloud);
      //   desc.setInputCloud(keypoints);
      //   estimateNormals(keypoints, cloud, keypoint_normals);
      //   desc.setInputNormals(keypoint_normals);
      //   search::KdTree<PointRGB>::Ptr kdtree(new search::KdTree<PointRGB>);
      //   desc.setSearchMethod(kdtree);
      //   desc.setRadiusSearch(0.08);
      //   desc.compute(*descriptors);
      // }
      // else if (desc_type == DESC_PPAL_CURV)
      // {
      //   // Convert to xyz
      //   PointCloudXYZ::Ptr cloud_xyz(new PointCloudXYZ);
      //   copyPointCloud(*cloud, *cloud_xyz);
      //   PointCloudXYZ::Ptr keypoints_xyz(new PointCloudXYZ);
      //   copyPointCloud(*keypoints, *keypoints_xyz);

      //   // Estimate features
      //   PrincipalCurvaturesEstimation<PointXYZ, Normal, PrincipalCurvatures> desc;
      //   desc.setSearchSurface(cloud_xyz);
      //   desc.setInputCloud(keypoints_xyz);
      //   estimateNormals(keypoints, cloud, keypoint_normals);
      //   desc.setInputNormals(keypoint_normals);
      //   search::KdTree<PointXYZ>::Ptr kdtree(new search::KdTree<PointXYZ>);
      //   desc.setSearchMethod(kdtree);
      //   desc.setRadiusSearch(0.08);
      //   desc.compute(*descriptors);
      // }
      // else if (desc_type == DESC_RIFT)
      // {
      //   // Compute gradients
      //   PointCloud<PointXYZI>::Ptr keypoint_intensities(new PointCloud<PointXYZI>);
      //   PointCloud<PointXYZI>::Ptr cloud_intensities(new PointCloud<PointXYZI>);
      //   PointCloud<IntensityGradient>::Ptr keypoint_gradients(new PointCloud<IntensityGradient>);
      //   estimateNormals(keypoints, cloud, keypoint_normals);
      //   PointCloudXYZRGBtoXYZI(*keypoints, *keypoint_intensities);
      //   PointCloudXYZRGBtoXYZI(*cloud, *cloud_intensities);
      //   computeGradient(keypoint_intensities, keypoint_normals, *keypoint_gradients);

      //   // Estimate features
      //   RIFTEstimation< PointXYZI, IntensityGradient, Histogram<32> > desc;
      //   desc.setInputCloud(keypoint_intensities);
      //   desc.setSearchSurface(cloud_intensities);
      //   search::KdTree<PointXYZI>::Ptr kdtree(new search::KdTree<PointXYZI>);
      //   desc.setSearchMethod(kdtree);
      //   desc.setInputGradient(keypoint_gradients);
      //   desc.setRadiusSearch(0.08);
      //   desc.setNrDistanceBins(4);
      //   desc.setNrGradientBins(8);
      //   desc.compute(*descriptors);
      // }
      // else if (desc_type == DESC_SHOT)
      // {
      //   // Convert to xyz
      //   PointCloudXYZ::Ptr cloud_xyz(new PointCloudXYZ);
      //   copyPointCloud(*cloud, *cloud_xyz);
      //   PointCloudXYZ::Ptr keypoints_xyz(new PointCloudXYZ);
      //   copyPointCloud(*keypoints, *keypoints_xyz);

      //   // Estimate features
      //   SHOTEstimationOMP<PointXYZ, Normal, SHOT352> desc;
      //   desc.setSearchSurface(cloud_xyz);
      //   desc.setInputCloud(keypoints_xyz);
      //   estimateNormals(keypoints, cloud, keypoint_normals);
      //   desc.setInputNormals(keypoint_normals);
      //   search::KdTree<PointXYZ>::Ptr kdtree(new search::KdTree<PointXYZ>);
      //   desc.setSearchMethod(kdtree);
      //   desc.setRadiusSearch(0.08);
      //   desc.compute(*descriptors);
      // }
      // else if (desc_type == DESC_SHOT_COLOR)
      // {
      //   SHOTColorEstimationOMP<PointRGB, Normal, SHOT1344> desc;
      //   desc.setSearchSurface(cloud);
      //   desc.setInputCloud(keypoints);
      //   estimateNormals(keypoints, cloud, keypoint_normals);
      //   desc.setInputNormals(keypoint_normals);
      //   search::KdTree<PointRGB>::Ptr kdtree(new search::KdTree<PointRGB>);
      //   desc.setSearchMethod(kdtree);
      //   desc.setRadiusSearch(0.08);
      //   desc.compute(*descriptors);
      // }
      // else if (desc_type == DESC_SHOT_LRF)
      // {
      //   // Convert to xyz
      //   PointCloudXYZ::Ptr cloud_xyz(new PointCloudXYZ);
      //   copyPointCloud(*cloud, *cloud_xyz);
      //   PointCloudXYZ::Ptr keypoints_xyz(new PointCloudXYZ);
      //   copyPointCloud(*keypoints, *keypoints_xyz);

      //   // Estimate features
      //   SHOTLocalReferenceFrameEstimationOMP<PointXYZ, SHOT352> desc;
      //   desc.setSearchSurface(cloud_xyz);
      //   desc.setInputCloud(keypoints_xyz);
      //   search::KdTree<PointXYZ>::Ptr kdtree(new search::KdTree<PointXYZ>);
      //   desc.setSearchMethod(kdtree);
      //   desc.setRadiusSearch(0.08);
      //   desc.compute(*descriptors);
      // }

      ros::WallTime desc_start = ros::WallTime::now();
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

