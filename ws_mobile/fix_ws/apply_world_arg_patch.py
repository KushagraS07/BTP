#!/usr/bin/env python3
"""
apply_world_arg_patch.py -- Phase 3B step B1.
Adds a `world` launch argument to ws_mobile's gazebo_model.launch.py so the slip world can be
selected without touching anything else:

    ros2 launch mobile_robot gazebo_model.launch.py                              # unchanged default
    ros2 launch mobile_robot gazebo_model.launch.py world:=indoor_world_slip.sdf # slip world

* default stays 'indoor_world.sdf' -> every existing behaviour is identical
* world name is looked up in <share>/mobile_robot/model/ (an absolute path also works)
* backup: gazebo_model.launch.py.bak_preWorldArg ; refuses to run twice
Run:  python3 apply_world_arg_patch.py [path/to/gazebo_model.launch.py]
"""
import os
import shutil
import sys

DEFAULT = os.path.expanduser('~/ws_mobile/src/mobile_robot/launch/gazebo_model.launch.py')
path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
src = open(path).read()

if "DeclareLaunchArgument('world'" in src:
    sys.exit('Already patched -- nothing changed.')

edits = [
    ("from launch.actions import IncludeLaunchDescription\n",
     "from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument\n"
     "from launch.substitutions import LaunchConfiguration, PathJoinSubstitution\n"),
    ("    worldFileRelativePath = 'model/indoor_world.sdf'\n"
     "    pathWorldFile = os.path.join(get_package_share_directory(namePackage), worldFileRelativePath)\n",
     "    # world:=<file in model/> (default indoor_world.sdf; slip world: indoor_world_slip.sdf)\n"
     "    worldArg = DeclareLaunchArgument('world', default_value='indoor_world.sdf',\n"
     "                                     description='world SDF in mobile_robot/model/ or an absolute path')\n"
     "    pathWorldFile = PathJoinSubstitution([get_package_share_directory(namePackage), 'model',\n"
     "                                          LaunchConfiguration('world')])\n"),
    ("launch_arguments={'gz_args':[f'-r -v -v4 {pathWorldFile}'],",
     "launch_arguments={'gz_args':['-r -v -v4 ', pathWorldFile],"),
    ("    launchDescriptionObject = LaunchDescription()\n",
     "    launchDescriptionObject = LaunchDescription()\n"
     "    launchDescriptionObject.add_action(worldArg)\n"),
]
for old, new in edits:
    n = src.count(old)
    if n != 1:
        sys.exit(f'ABORT: expected exactly 1 match, found {n} for:\n{old}\nNothing changed.')
    src = src.replace(old, new)

compile(src, path, 'exec')  # syntax check before writing
bak = path + '.bak_preWorldArg'
shutil.copy2(path, bak)
open(path, 'w').write(src)
print(f'Patched {path}\nBackup  {bak}')
