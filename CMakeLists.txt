cmake_minimum_required(VERSION 2.8.3)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules/")

project(pcl_feature_extraction)
find_package(catkin REQUIRED COMPONENTS
	roscpp
	tf
	pcl_ros
	cmake_modules)

####################################################
# Dependencies:
find_package(PkgConfig)

# Dependencies - Eigen:
find_package(Eigen REQUIRED)

# Dependencies - PCL:
find_package(PCL REQUIRED)

catkin_package()

include_directories(include
                    ${catkin_INCLUDE_DIRS}
										${eigen_INCLUDE_DIRS}
                    ${PCL_INCLUDE_DIRS})

# Localization node
add_executable(pcl_feature_extraction
						   src/evaluation.cpp)
target_link_libraries(pcl_feature_extraction
						${eigen_LIBRARIES}
            ${PCL_LIBRARIES}
						${catkin_LIBRARIES})