"""
gazebo_model.launch.py (ws_mobile) -- spawn differential_drive_robot in Gazebo Sim.

Launch arguments (defaults = original behaviour):
  world        world SDF in mobile_robot/model/ or an absolute path   default indoor_world.sdf
  imu_rate     IMU output rate (Hz), 50 = original; 1000 = physics step, averaged to 50 Hz offline
  gyro_bias_z  gyro z bias_mean (rad/s), 0.0000075 = original; 0.01 = gyro-bias variant
  odom_rate    odometry publish rate (Hz), default 50
  x, y, yaw    spawn pose (m, m, rad), default 0 0 0
Ported from Paper_ws (paper_robot_sim) on 2026-10-09; gz_args kept "-r -v -v4" (Harmonic).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import xacro

PACKAGE = 'mobile_robot'
ROBOT_NAME = 'differential_drive_robot'   # must match the robot name in robot.xacro


def launch_setup(context, *args, **kwargs):
    share = get_package_share_directory(PACKAGE)
    world = LaunchConfiguration('world').perform(context)
    world_path = world if os.path.isabs(world) else os.path.join(share, 'model', world)
    if not os.path.exists(world_path):
        raise FileNotFoundError(f'World file not found: {world_path}')
    gyro_bias_z = LaunchConfiguration('gyro_bias_z').perform(context)
    float(gyro_bias_z)
    imu_rate = LaunchConfiguration('imu_rate').perform(context)
    if not 0 < float(imu_rate) <= 1000:
        raise ValueError(f'imu_rate must be in (0, 1000] Hz (physics step 1 ms), got {imu_rate}')
    odom_rate = LaunchConfiguration('odom_rate').perform(context)
    if not 0 < float(odom_rate) <= 1000:
        raise ValueError(f'odom_rate must be in (0, 1000] Hz, got {odom_rate}')
    sx, sy, syaw = (float(LaunchConfiguration(k).perform(context)) for k in ('x', 'y', 'yaw'))

    robot_description = xacro.process_file(
        os.path.join(share, 'model', 'robot.xacro'),
        mappings={'gyro_bias_z': gyro_bias_z, 'imu_rate': imu_rate, 'odom_rate': odom_rate},
    ).toxml()

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': [f'-r -v -v4 {world_path}'],
                          'on_exit_shutdown': 'true'}.items(),
    )
    spawn = Node(
        package='ros_gz_sim', executable='create',
        arguments=['-name', ROBOT_NAME, '-topic', 'robot_description',
                   '-x', str(sx), '-y', str(sy), '-z', '0.0', '-Y', str(syaw)],
        output='screen',
    )
    robot_state_publisher = Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description, 'use_sim_time': True}],
    )
    bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge',
        arguments=['--ros-args', '-p',
                   f'config_file:={os.path.join(share, "parameters", "bridge_parameters.yaml")}'],
        output='screen',
    )
    print(f'[mobile_robot] world={world_path}  imu_rate={imu_rate}  odom_rate={odom_rate}  '
          f'gyro_bias_z={gyro_bias_z}  spawn=({sx}, {sy}, yaw {syaw})')
    return [gazebo, spawn, robot_state_publisher, bridge]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='indoor_world.sdf',
                              description='world SDF in mobile_robot/model/ or an absolute path'),
        DeclareLaunchArgument('imu_rate', default_value='50',
                              description='IMU output rate (Hz); 1000 = averaged to 50 Hz offline'),
        DeclareLaunchArgument('gyro_bias_z', default_value='0.0000075',
                              description='gyro z bias_mean (rad/s); 0.01 = gyro-bias variant'),
        DeclareLaunchArgument('odom_rate', default_value='50',
                              description='odometry publish rate (Hz)'),
        DeclareLaunchArgument('x', default_value='0.0', description='spawn x (m)'),
        DeclareLaunchArgument('y', default_value='0.0', description='spawn y (m)'),
        DeclareLaunchArgument('yaw', default_value='0.0', description='spawn yaw (rad)'),
        OpaqueFunction(function=launch_setup),
    ])
