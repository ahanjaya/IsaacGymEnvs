# Copyright (c) 2021-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
import torch
from typing import Tuple

from isaacgym import gymtorch
from isaacgym import gymapi, gymutil

from isaacgymenvs.utils.torch_jit_utils import to_torch, tensor_clamp
from isaacgymenvs.tasks.base.vec_task import VecTask


class FrankaReach(VecTask):
    def __init__(
        self,
        cfg,
        rl_device,
        sim_device,
        graphics_device_id,
        headless,
        virtual_screen_capture,
        force_render,
    ):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.franka_dof_noise = self.cfg["env"]["frankaDofNoise"]

        # Controller type
        self.control_type = self.cfg["env"]["controlType"]
        assert self.control_type in {
            "osc",
            "joint_tor",
        }, "Invalid control type specified. Must be one of: {osc, joint_tor}"

        # dimensions
        # obs include: eef_pose (7) + q_gripper (2)
        self.cfg["env"]["numObservations"] = 26
        # actions include: joint torques (7) + bool gripper (1)
        self.cfg["env"]["numActions"] = 8

        # # Values to be filled in at runtime
        self.states = {}

        # Tensor placeholders
        # self._root_state = None  # State of root body (n_envs, 13)
        # self._dof_state = None  # State of all joints (n_envs, n_dof)
        # self._q = None  # Joint positions (n_envs, n_dof)
        # self._qd = None  # Joint velocities (n_envs, n_dof)
        # self._rigid_body_state = (
        #     None # State of all rigid bodies (n_envs, n_bodies, 13)
        # )
        # self._contact_forces = None  # Contact forces in sim
        # self._eef_state = None  # end effector state (at grasping point)
        # self._eef_lf_state = None  # end effector state (at left fingertip)
        # self._eef_rf_state = None  # end effector state (at left fingertip)
        # self._j_eef = None  # Jacobian for end effector
        # self._mm = None  # Mass matrix
        # self._arm_control = None  # Tensor buffer for controlling arm
        # self._gripper_control = None  # Tensor buffer for controlling gripper
        # self._pos_control = None  # Position actions
        # self._effort_control = None  # Torque actions
        # self._franka_effort_limits = None  # Actuator effort limits for franka
        # self._global_indices = (
        #     None  # Unique indices corresponding to all envs in flattened array
        # )

        self.up_axis = "z"
        self.up_axis_idx = 2

        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )

        # Franka defaults
        self.franka_default_dof_pos = to_torch(
            [0, 0.1963, 0, -2.6180, 0, 2.9416, 0.7854, 0.035, 0.035],
            device=self.device,
        )

        # Set control limits
        self.cmd_limit = self._franka_effort_limits[:7].unsqueeze(0)

        # Reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        # Refresh tensors
        self._refresh()

        if not self.headless:
            self._init_debug()

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        self._create_ground_plane()
        self._create_envs(
            self.num_envs,
            self.cfg["env"]["envSpacing"],
            int(np.sqrt(self.num_envs)),
        )

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../../assets"
        )
        franka_asset_file = (
            "urdf/franka_description/robots/franka_panda_gripper.urdf"
        )

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                self.cfg["env"]["asset"].get("assetRoot", asset_root),
            )
            franka_asset_file = self.cfg["env"]["asset"].get(
                "assetFileNameFranka", franka_asset_file
            )

        # load franka asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.use_mesh_materials = True
        franka_asset = self.gym.load_asset(
            self.sim, asset_root, franka_asset_file, asset_options
        )
        franka_dof_stiffness = to_torch(
            [0, 0, 0, 0, 0, 0, 0, 5000.0, 5000.0],
            dtype=torch.float,
            device=self.device,
        )
        franka_dof_damping = to_torch(
            [0, 0, 0, 0, 0, 0, 0, 1.0e2, 1.0e2],
            dtype=torch.float,
            device=self.device,
        )

        # Create table asset
        table_pos = [0.0, 0.0, 1.0]
        table_thickness = 0.05
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        table_asset = self.gym.create_box(
            self.sim, *[1.2, 1.2, table_thickness], table_opts
        )

        # Create table stand asset
        table_stand_height = 0.1
        table_stand_pos = [
            -0.5,
            0.0,
            1.0 + table_thickness / 2 + table_stand_height / 2,
        ]
        table_stand_opts = gymapi.AssetOptions()
        table_stand_opts.fix_base_link = True
        table_stand_asset = self.gym.create_box(
            self.sim, *[0.2, 0.2, table_stand_height], table_opts
        )

        # Create target asset
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.use_mesh_materials = True
        target_radius = 0.025
        target_asset = self.gym.create_sphere(
            self.sim, target_radius, asset_options
        )

        self.num_franka_bodies = self.gym.get_asset_rigid_body_count(
            franka_asset
        )
        self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)
        print("num franka bodies: ", self.num_franka_bodies)
        print("num franka dofs: ", self.num_franka_dofs)

        # set franka dof properties
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        self.franka_dof_lower_limits = []
        self.franka_dof_upper_limits = []
        self._franka_effort_limits = []
        for i in range(self.num_franka_dofs):
            franka_dof_props["driveMode"][i] = (
                gymapi.DOF_MODE_POS if i > 6 else gymapi.DOF_MODE_EFFORT
            )
            if self.physics_engine == gymapi.SIM_PHYSX:
                franka_dof_props["stiffness"][i] = franka_dof_stiffness[i]
                franka_dof_props["damping"][i] = franka_dof_damping[i]
            else:
                franka_dof_props["stiffness"][i] = 7000.0
                franka_dof_props["damping"][i] = 50.0

            self.franka_dof_lower_limits.append(franka_dof_props["lower"][i])
            self.franka_dof_upper_limits.append(franka_dof_props["upper"][i])
            self._franka_effort_limits.append(franka_dof_props["effort"][i])

        self.franka_dof_lower_limits = to_torch(
            self.franka_dof_lower_limits, device=self.device
        )
        self.franka_dof_upper_limits = to_torch(
            self.franka_dof_upper_limits, device=self.device
        )
        self._franka_effort_limits = to_torch(
            self._franka_effort_limits, device=self.device
        )
        self.franka_dof_speed_scales = torch.ones_like(
            self.franka_dof_lower_limits
        )
        self.franka_dof_speed_scales[[7, 8]] = 0.1
        franka_dof_props["effort"][7] = 200
        franka_dof_props["effort"][8] = 200

        # Define start pose for franka
        franka_start_pose = gymapi.Transform()
        franka_start_pose.p = gymapi.Vec3(
            -0.45, 0.0, 1.0 + table_thickness / 2 + table_stand_height
        )
        franka_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for table
        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(*table_pos)
        table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self._table_surface_pos = np.array(table_pos) + np.array(
            [0, 0, table_thickness / 2]
        )

        # Define start pose for table stand
        table_stand_start_pose = gymapi.Transform()
        table_stand_start_pose.p = gymapi.Vec3(*table_stand_pos)
        table_stand_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for target
        target_start_pose = gymapi.Transform()
        self._target_start_pos = self._table_surface_pos + np.array(
            [0.0, 0.0, target_radius]
        )
        target_start_pose.p = gymapi.Vec3(*self._target_start_pos)
        target_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # compute aggregate size
        num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
        max_agg_bodies = (
            num_franka_bodies + 3
        )  # 1 for table, 1 table stand, 1 target
        max_agg_shapes = (
            num_franka_shapes + 3
        )  # 1 for table, 1 table stand, 1 target

        self.frankas = []
        self.envs = []

        # Create environments
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # Create actors and define aggregate group appropriately depending on setting
            # NOTE: franka should ALWAYS be loaded first in sim!
            self.gym.begin_aggregate(
                env_ptr, max_agg_bodies, max_agg_shapes, True
            )

            # Create franka
            franka_actor = self.gym.create_actor(
                env_ptr, franka_asset, franka_start_pose, "franka", i, 0, 0
            )
            self.gym.set_actor_dof_properties(
                env_ptr, franka_actor, franka_dof_props
            )

            # Create table
            self.gym.create_actor(
                env_ptr, table_asset, table_start_pose, "table", i, 1, 0
            )
            self.gym.create_actor(
                env_ptr,
                table_stand_asset,
                table_stand_start_pose,
                "table_stand",
                i,
                1,
                0,
            )

            # Create target
            self._target_id = self.gym.create_actor(
                env_ptr, target_asset, target_start_pose, "target", i, 2, 0
            )
            self.gym.set_rigid_body_color(
                env_ptr,
                self._target_id,
                0,
                gymapi.MESH_VISUAL,
                gymapi.Vec3(1.0, 0.0, 0.0),
            )

            self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)
            self.frankas.append(franka_actor)

        # Setup data
        self.init_data()

    def _init_debug(self):
        # Focus viewer's camera on the first environment.
        self.flag_camera_look_at = self.cfg["render"].get(
            "enableCameraLookAtEnv", False
        )
        self.i_look_at_env = self.cfg["render"].get("cameraLookAtEnvId", 0)
        self.debug_cam_pos = gymapi.Vec3(*self.cfg["render"]["cameraPosition"])
        self.debug_cam_target = gymapi.Vec3(
            *self.cfg["render"]["cameraTarget"]
        )
        self.flag_debug_vis = self.cfg["render"].get("enableDebugVis", False)
        self.gym.viewer_camera_look_at(
            self.viewer,
            self.envs[self.i_look_at_env],
            self.debug_cam_pos,
            self.debug_cam_target,
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_T, "camera_look_at"
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_Y, "env_prev"
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_U, "env_next"
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_R, "reset_envs"
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_P, "debug_vis"
        )

    def keyboard(self, event):
        if event.action == "camera_look_at" and event.value > 0:
            self.flag_camera_look_at = not self.flag_camera_look_at
        elif event.action == "env_prev" and event.value > 0:
            self.i_look_at_env = max(0, self.i_look_at_env - 1)
            self.update_debug_camera()
        elif event.action == "env_next" and event.value > 0:
            self.i_look_at_env = min(self.i_look_at_env + 1, self.num_envs - 1)
            self.update_debug_camera()
        elif event.action == "reset_envs" and event.value > 0:
            self.reset_buf[:] = 1
        elif event.action == "debug_vis" and event.value > 0:
            self.gym.clear_lines(self.viewer)
            self.flag_debug_vis = not self.flag_debug_vis

    def viewer_update(self):
        if self.flag_debug_vis:
            self._draw_debug_vis()

    def update_debug_camera(self):
        if not self.flag_camera_look_at:
            return

        self.gym.viewer_camera_look_at(
            self.viewer,
            self.envs[self.i_look_at_env],
            self.debug_cam_pos,
            self.debug_cam_target,
        )

    def init_data(self):
        # Setup sim handles
        env_ptr = self.envs[0]
        franka_handle = 0

        self.handles = {
            # Franka
            "hand": self.gym.find_actor_rigid_body_handle(
                env_ptr, franka_handle, "panda_hand"
            ),
            "leftfinger_tip": self.gym.find_actor_rigid_body_handle(
                env_ptr, franka_handle, "panda_leftfinger_tip"
            ),
            "rightfinger_tip": self.gym.find_actor_rigid_body_handle(
                env_ptr, franka_handle, "panda_rightfinger_tip"
            ),
            "grip_site": self.gym.find_actor_rigid_body_handle(
                env_ptr, franka_handle, "panda_grip_site"
            ),
            # Target
            "target": self.gym.find_actor_rigid_body_handle(
                env_ptr, self._target_id, "target"
            ),
        }

        # Get total DOFs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(
            self.sim
        )
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(
            self.sim
        )
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(
            self.num_envs, -1, 13
        )
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(
            self.num_envs, -1, 2
        )
        self._rigid_body_state = gymtorch.wrap_tensor(
            _rigid_body_state_tensor
        ).view(self.num_envs, -1, 13)
        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]
        self._eef_state = self._rigid_body_state[
            :, self.handles["grip_site"], :
        ]
        self._eef_lf_state = self._rigid_body_state[
            :, self.handles["leftfinger_tip"], :
        ]
        self._eef_rf_state = self._rigid_body_state[
            :, self.handles["rightfinger_tip"], :
        ]
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "franka")
        jacobian = gymtorch.wrap_tensor(_jacobian)
        hand_joint_index = self.gym.get_actor_joint_dict(
            env_ptr, franka_handle
        )["panda_hand_joint"]
        self._j_eef = jacobian[:, hand_joint_index, :, :7]

        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "franka")
        mm = gymtorch.wrap_tensor(_massmatrix)
        self._mm = mm[:, :7, :7]
        self._target_state = self._root_state[:, self._target_id, :]

        # Initialize actions
        self._pos_control = torch.zeros(
            (self.num_envs, self.num_dofs),
            dtype=torch.float,
            device=self.device,
        )
        self._effort_control = torch.zeros_like(self._pos_control)

        # Initialize control
        self._arm_control = self._effort_control[:, :7]
        self._gripper_control = self._pos_control[:, 7:9]

        # Initialize indices
        self._global_indices = torch.arange(
            self.num_envs * 4, dtype=torch.int32, device=self.device
        ).view(self.num_envs, -1)

    def _update_states(self):
        self.states.update(
            {
                # Franka
                "q": self._q[:, :],
                "q_gripper": self._q[:, -2:],
                "eef_pos": self._eef_state[:, :3],
                "eef_quat": self._eef_state[:, 3:7],
                "eef_vel": self._eef_state[:, 7:],
                "eef_lf_pos": self._eef_lf_state[:, :3],
                "eef_rf_pos": self._eef_rf_state[:, :3],
                # Target
                "target_pos": self._target_state[:, :3],
                "target_quat": self._target_state[:, 3:7],
            }
        )

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        # Refresh states
        self._update_states()

    def _draw_debug_vis(self):
        """Draws visualizations for dubugging (slows down simulation a lot).
        Default behaviour: draws height measurement points
        """
        # draw height lines
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        eef_sphere_geom = gymutil.WireframeSphereGeometry(
            0.02, 8, 8, None, color=(0, 1, 0)
        )
        target_sphere_geom = gymutil.WireframeSphereGeometry(
            0.04, 8, 8, None, color=(0, 0, 1)
        )

        for i in range(self.num_envs):
            # Visualize End Effector
            eef_x, eef_y, eef_z = self.states["eef_pos"][i, :3].cpu().numpy()
            eef_sphere_pose = gymapi.Transform(
                gymapi.Vec3(eef_x, eef_y, eef_z), r=None
            )
            gymutil.draw_lines(
                eef_sphere_geom,
                self.gym,
                self.viewer,
                self.envs[i],
                eef_sphere_pose,
            )

            # Visualize Target Pos
            target_x, target_y, target_z = (
                self.states["target_pos"][i, :3].cpu().numpy()
            )
            target_sphere_pose = gymapi.Transform(
                gymapi.Vec3(target_x, target_y, target_z), r=None
            )
            gymutil.draw_lines(
                target_sphere_geom,
                self.gym,
                self.viewer,
                self.envs[i],
                target_sphere_pose,
            )

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_franka_reward(
            self.reset_buf, self.progress_buf, self.max_episode_length
        )

    def compute_observations(self):
        self._refresh()
        self.obs_buf = torch.zeros_like(self.obs_buf)

        return self.obs_buf

    def reset_idx(self, env_ids):
        # Reset agent
        reset_noise = torch.rand((len(env_ids), 9), device=self.device)
        pos = tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0)
            + self.franka_dof_noise * 2.0 * (reset_noise - 0.5),
            self.franka_dof_lower_limits.unsqueeze(0),
            self.franka_dof_upper_limits,
        )

        # Overwrite gripper init pos (no noise since these are always position controlled)
        pos[:, -2:] = self.franka_default_dof_pos[-2:]

        # Reset the internal obs accordingly
        self._q[env_ids, :] = pos
        self._qd[env_ids, :] = torch.zeros_like(self._qd[env_ids])

        # Set any position control to the current position, and any vel / effort control to be 0
        # NOTE: Task takes care of actually propagating these controls in sim using the SimActions API
        self._pos_control[env_ids, :] = pos
        self._effort_control[env_ids, :] = torch.zeros_like(pos)

        ########################################################################
        # Reset target root states pos
        num_resets = len(env_ids)
        sampled_target_state = torch.zeros(num_resets, 13, device=self.device)

        # Initialize rotation, which is no rotation (quat w = 1)
        sampled_target_state[:, 6] = 1.0

        # Sampling xy is "centered" around middle of table
        centered_target_xy_state = torch.tensor(
            self._table_surface_pos[:2],
            device=self.device,
            dtype=torch.float32,
        )
        sampled_target_state[:, :2] = centered_target_xy_state.unsqueeze(
            0
        ) + 2.0 * self.start_position_noise * (
            torch.rand(num_resets, 2, device=self.device) - 0.5
        )

        # Set z value, minimum is the start_position_noise
        sampled_target_state[:, 2] = self._target_start_pos[
            2
        ] + 2.0 * self.start_position_noise * torch.rand(
            num_resets, device=self.device
        )

        self._target_state[env_ids] = sampled_target_state

        # Deploy updates
        multi_env_ids_int32 = self._global_indices[env_ids, 0].flatten()
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._pos_control),
            gymtorch.unwrap_tensor(multi_env_ids_int32),
            len(multi_env_ids_int32),
        )
        self.gym.set_dof_actuation_force_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._effort_control),
            gymtorch.unwrap_tensor(multi_env_ids_int32),
            len(multi_env_ids_int32),
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(multi_env_ids_int32),
            len(multi_env_ids_int32),
        )

        # Update target states
        multi_env_ids_target_int32 = self._global_indices[
            env_ids, -1
        ].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_target_int32),
            len(multi_env_ids_target_int32),
        )

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)

        # Split arm and gripper command
        u_arm, u_gripper = self.actions[:, :-1], self.actions[:, -1]
        u_arm = torch.zeros_like(u_arm)
        u_gripper = torch.zeros_like(u_gripper)

        # Control arm (scale value first)
        u_arm = u_arm * self.cmd_limit / self.action_scale
        self._arm_control[:, :] = u_arm

        # Control gripper
        u_fingers = torch.zeros_like(self._gripper_control)
        u_fingers[:, 0] = torch.where(
            u_gripper >= 0.0,
            self.franka_dof_upper_limits[-2].item(),
            self.franka_dof_lower_limits[-2].item(),
        )
        u_fingers[:, 1] = torch.where(
            u_gripper >= 0.0,
            self.franka_dof_upper_limits[-1].item(),
            self.franka_dof_lower_limits[-1].item(),
        )
        # Write gripper command to appropriate tensor buffer
        self._gripper_control[:, :] = u_fingers

        # Deploy actions
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self._pos_control)
        )
        self.gym.set_dof_actuation_force_tensor(
            self.sim, gymtorch.unwrap_tensor(self._effort_control)
        )

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_franka_reward(
    reset_buf: torch.Tensor,
    progress_buf: torch.Tensor,
    max_episode_length: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Compose rewards
    rewards = torch.zeros_like(progress_buf)

    # Compute resets
    reset_buf = torch.where(
        progress_buf >= max_episode_length - 1,
        torch.ones_like(reset_buf),
        reset_buf,
    )

    return rewards, reset_buf
