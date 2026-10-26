#include <gtest/gtest.h>
#include "plan_env/grid_map.h"

// Exercise the actual cloud raycasting/inflation path without a drone or GPU.
class GridMapMemoryTestPeer
{
public:
  static void pose(GridMap &map, const Eigen::Vector3d &position)
  {
    auto odom = std::make_shared<nav_msgs::msg::Odometry>();
    odom->pose.pose.position.x = position.x();
    odom->pose.pose.position.y = position.y();
    odom->pose.pose.position.z = position.z();
    map.odomCallback(odom);
  }

  static void observe(GridMap &map, const Eigen::Vector3d &camera,
                      const std::vector<Eigen::Vector3d> &points, int stamp)
  {
    pcl::PointCloud<pcl::PointXYZ> cloud;
    for (const auto &point : points)
      cloud.push_back(pcl::PointXYZ(point.x(), point.y(), point.z()));
    auto message = std::make_shared<sensor_msgs::msg::PointCloud2>();
    pcl::toROSMsg(cloud, *message);
    message->header.frame_id = "world";
    message->header.stamp.sec = stamp;
    auto pose = std::make_shared<geometry_msgs::msg::PoseStamped>();
    pose->header = message->header;
    pose->pose.position.x = camera.x();
    pose->pose.position.y = camera.y();
    pose->pose.position.z = camera.z();
    map.cloudPoseCallback(message, pose);
  }

  static size_t size(GridMap &map) { return map.md_.obstacle_memory_.size(); }
  static void age(GridMap &map, double seconds)
  {
    for (auto &entry : map.md_.obstacle_memory_)
      entry.second.last_seen -= seconds;
  }
  static double lastSeen(GridMap &map)
  {
    return map.md_.obstacle_memory_.begin()->second.last_seen;
  }
  static void tick(GridMap &map) { map.updateOccupancyCallback(); }
  static void minHits(GridMap &map, int hits) { map.mp_.min_obstacle_hits_ = hits; }
  static void epoch(GridMap &map, int stamp)
  {
    auto message = std::make_shared<std_msgs::msg::Header>();
    message->frame_id = "orb_epoch_2";
    message->stamp.sec = stamp;
    map.mapEpochCallback(message);
  }
};

class ObstacleMemoryTest : public ::testing::Test
{
protected:
  using Peer = GridMapMemoryTestPeer;
  static void SetUpTestSuite()
  {
    int argc = 0;
    rclcpp::init(argc, nullptr);
  }
  static void TearDownTestSuite() { rclcpp::shutdown(); }

  void SetUp() override
  {
    rclcpp::NodeOptions options;
    options.parameter_overrides({
      rclcpp::Parameter("grid_map/resolution", 0.2),
      // Deliberately not a voxel multiple: initial/recentered lattices must agree.
      rclcpp::Parameter("grid_map/map_size_x", 12.1),
      rclcpp::Parameter("grid_map/map_size_y", 12.1),
      rclcpp::Parameter("grid_map/map_size_z", 8.0),
      rclcpp::Parameter("grid_map/local_update_range_x", 3.0),
      rclcpp::Parameter("grid_map/local_update_range_y", 3.0),
      rclcpp::Parameter("grid_map/local_update_range_z", 3.0),
      rclcpp::Parameter("grid_map/ground_height", -4.0),
      rclcpp::Parameter("grid_map/obstacles_inflation", 0.2),
      rclcpp::Parameter("grid_map/self_clearance", 0.25),
      rclcpp::Parameter("grid_map/skip_pixel", 2),
      rclcpp::Parameter("grid_map/max_ray_length", 5.0),
      rclcpp::Parameter("grid_map/virtual_ceil_height", -1.0),
      rclcpp::Parameter("grid_map/odom_depth_timeout", 10.0),
      rclcpp::Parameter("grid_map/cloud_camera_pose", true),
      rclcpp::Parameter("grid_map/obstacle_memory_sec", 2.0),
      rclcpp::Parameter("grid_map/obstacle_memory_radius", 2.0),
    });
    node = std::make_shared<rclcpp::Node>("grid_memory_test", options);
    map = std::make_unique<GridMap>();
    map->initMap(node);
    Peer::pose(*map, camera);
  }

  void observe(const std::vector<Eigen::Vector3d> &points)
  {
    Peer::observe(*map, camera, points, ++stamp);
  }

  rclcpp::Node::SharedPtr node;
  std::unique_ptr<GridMap> map;
  Eigen::Vector3d camera{0.1, 0.1, 0.1};
  const Eigen::Vector3d obstacle{1.1, 0.1, 0.1};
  int stamp{100};
};

TEST_F(ObstacleMemoryTest, SideObstacleSurvivesOppositeViewAndEmptyFrame)
{
  observe({obstacle});
  observe({Eigen::Vector3d(-1.1, 0.1, 0.1)});
  EXPECT_EQ(map->getOccupancy(obstacle), 1);
  EXPECT_EQ(map->getInflateOccupancy(Eigen::Vector3d(1.3, 0.1, 0.1)), 1);
  observe({});
  EXPECT_EQ(map->getOccupancy(obstacle), 1);
  EXPECT_EQ(map->getInflateOccupancy(obstacle), 1);
}

TEST_F(ObstacleMemoryTest, FreeRayClearsPreviousObstacleImmediately)
{
  observe({obstacle});
  observe({Eigen::Vector3d(2.9, 0.1, 0.1)});
  EXPECT_EQ(map->getOccupancy(obstacle), 0);
  EXPECT_EQ(map->getInflateOccupancy(obstacle), 0);
  EXPECT_EQ(Peer::size(*map), 0u);  // New endpoint is outside memory radius.
  EXPECT_EQ(map->getOccupancy(Eigen::Vector3d(2.9, 0.1, 0.1)), 1);
}

TEST_F(ObstacleMemoryTest, ExpiryClearsRawAndInflatedMapWithoutAnotherFrame)
{
  observe({obstacle});
  Peer::age(*map, 2.1);
  Peer::tick(*map);
  EXPECT_EQ(Peer::size(*map), 0u);
  EXPECT_EQ(map->getOccupancy(obstacle), 0);
  EXPECT_EQ(map->getInflateOccupancy(Eigen::Vector3d(1.3, 0.1, 0.1)), 0);
}

TEST_F(ObstacleMemoryTest, MovingBeyondRadiusDropsNearbyMemory)
{
  observe({obstacle});
  Peer::pose(*map, Eigen::Vector3d(-1.5, 0.1, 0.1));
  Peer::tick(*map);
  EXPECT_EQ(Peer::size(*map), 0u);
  EXPECT_EQ(map->getInflateOccupancy(obstacle), 0);
}

TEST_F(ObstacleMemoryTest, WindowShiftPreservesWorldObstacleAndFreeRayClearsIt)
{
  observe({obstacle});
  const auto old_origin = map->getOrigin();
  camera.x() = 2.3;
  Peer::pose(*map, camera);
  observe({});
  EXPECT_GT((old_origin - map->getOrigin()).norm(), 0.1);
  EXPECT_EQ(map->getOccupancy(obstacle), 1);
  EXPECT_EQ(map->getInflateOccupancy(obstacle), 1);
  observe({Eigen::Vector3d(-0.5, 0.1, 0.1)});
  EXPECT_EQ(map->getOccupancy(obstacle), 0);
  EXPECT_EQ(map->getInflateOccupancy(obstacle), 0);
  EXPECT_EQ(Peer::size(*map), 0u);
}

TEST_F(ObstacleMemoryTest, EpochChangeClearsAndRejectsQueuedOldClouds)
{
  observe({obstacle});
  Peer::epoch(*map, 103);
  EXPECT_EQ(Peer::size(*map), 0u);
  EXPECT_EQ(map->getInflateOccupancy(obstacle), 0);
  observe({obstacle});  // Stamp 102 belongs to the previous epoch.
  EXPECT_EQ(map->getOccupancy(obstacle), 0);
  observe({obstacle});
  EXPECT_EQ(map->getOccupancy(obstacle), 1);
}

TEST_F(ObstacleMemoryTest, DuplicateAndOutOfOrderCloudsDoNotRefreshMemory)
{
  observe({obstacle});
  Peer::age(*map, 1.0);
  const double seen = Peer::lastSeen(*map);
  Peer::observe(*map, camera, {obstacle}, stamp);
  Peer::observe(*map, camera, {obstacle}, stamp - 1);
  EXPECT_EQ(Peer::lastSeen(*map), seen);
  Peer::age(*map, 1.1);
  Peer::tick(*map);
  EXPECT_EQ(map->getInflateOccupancy(obstacle), 0);
}

TEST_F(ObstacleMemoryTest, WeakEndpointDoesNotEraseButFreeRayDoes)
{
  Peer::minHits(*map, 2);
  observe({obstacle, obstacle});
  ASSERT_EQ(map->getOccupancy(obstacle), 1);
  observe({obstacle});
  EXPECT_EQ(map->getOccupancy(obstacle), 1);
  observe({Eigen::Vector3d(2.9, 0.1, 0.1)});
  EXPECT_EQ(map->getInflateOccupancy(obstacle), 0);
  EXPECT_EQ(Peer::size(*map), 0u);
}

TEST_F(ObstacleMemoryTest, VehicleFootprintIsNotRetainedOrInflated)
{
  observe({camera});
  EXPECT_EQ(Peer::size(*map), 0u);
  EXPECT_EQ(map->getInflateOccupancy(camera), 0);
  observe({});
  EXPECT_EQ(map->getOccupancy(camera), 0);
}
