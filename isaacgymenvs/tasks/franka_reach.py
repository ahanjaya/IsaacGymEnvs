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

import os
from typing import Tuple

import numpy as np
import torch
from isaacgym import gymapi, gymtorch, gymutil

from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgymenvs.utils.torch_jit_utils import tensor_clamp, to_torch


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
            "cartesian",
            "joint",
        }, "Invalid control type specified. Must be one of: {cartesian, joint}"

        # observation and action space
        self.cfg["env"]["numObservations"] = 18

        if self.control_type == "joint":
            # actions include: joint (7) + bool gripper (1)
            self.cfg["env"]["numActions"] = 7  # TODO: Add gripper control
        elif self.control_type == "cartesian":
            self.cfg["env"]["numActions"] = 3

        self._action_scale = self.cfg["env"]["actionScale"]
        self._dof_vel_scale = self.cfg["env"]["dofVelocityScale"]

        # Values to be filled in at runtime
        self.states = {}

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
            os.path.dirname(os.path.abspath(__file__)),
            self.cfg["env"]["asset"]["assetRoot"],
        )
        franka_asset_file = self.cfg["env"]["asset"]["assetFileNameFranka"]

        ##########################################################
        # Create all assets
        ##########################################################
        # franka asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        franka_asset = self.gym.load_asset(
            self.sim, asset_root, franka_asset_file, asset_options
        )

        # table asset
        table_pos = [0.0, 0.0, 1.0]
        table_thickness = 0.05
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        table_asset = self.gym.create_box(
            self.sim, *[1.2, 1.2, table_thickness], table_opts
        )

        # table stand asset
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

        # target asset
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

        ##########################################################
        # Set up franka dof properties
        ##########################################################
        self.rigid_body_dict_franka = self.gym.get_asset_rigid_body_dict(
            franka_asset
        )
        self.num_franka_bodies = self.gym.get_asset_rigid_body_count(
            franka_asset
        )
        self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)
        print("num franka bodies: ", self.num_franka_bodies)
        print("num franka dofs: ", self.num_franka_dofs)

        franka_dof_stiffness = torch.tensor(
            [400, 400, 400, 400, 400, 400, 400, 1.0e6, 1.0e6],
            dtype=torch.float32,
            device=self.device,
        )
        franka_dof_damping = torch.tensor(
            [80, 80, 80, 80, 80, 80, 80, 1.0e2, 1.0e2],
            dtype=torch.float,
            device=self.device,
        )
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        self.franka_dof_lower_limits = []
        self.franka_dof_upper_limits = []
        for i in range(self.num_franka_dofs):
            franka_dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            if self.physics_engine == gymapi.SIM_PHYSX:
                franka_dof_props["stiffness"][i] = franka_dof_stiffness[i]
                franka_dof_props["damping"][i] = franka_dof_damping[i]
            else:
                franka_dof_props["stiffness"][i] = 7000.0
                franka_dof_props["damping"][i] = 50.0

            self.franka_dof_lower_limits.append(franka_dof_props["lower"][i])
            self.franka_dof_upper_limits.append(franka_dof_props["upper"][i])

        self.franka_dof_lower_limits = to_torch(
            self.franka_dof_lower_limits, device=self.device
        )
        self.franka_dof_upper_limits = to_torch(
            self.franka_dof_upper_limits, device=self.device
        )
        self.franka_dof_speed_scales = torch.ones_like(
            self.franka_dof_lower_limits
        )
        self.franka_dof_speed_scales[[7, 8]] = 0.1
        franka_dof_props["effort"][7] = 200
        franka_dof_props["effort"][8] = 200

        ##########################################################
        # Define start poses
        ##########################################################
        # start pose for franka
        franka_start_pose = gymapi.Transform()
        franka_start_pose.p = gymapi.Vec3(
            -0.45, 0.0, 1.0 + table_thickness / 2 + table_stand_height
        )
        franka_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # start pose for table
        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(*table_pos)
        table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self._table_surface_pos = np.array(table_pos) + np.array(
            [0, 0, table_thickness / 2]
        )

        # start pose for table stand
        table_stand_start_pose = gymapi.Transform()
        table_stand_start_pose.p = gymapi.Vec3(*table_stand_pos)
        table_stand_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # start pose for target
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

        self.envs = []
        self.frankas = []
        self.targets = []

        indexes_sim_franka = []
        indexes_sim_target = []

        ##########################################################
        # Create environments
        ##########################################################
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
            indexes_sim_franka.append(
                self.gym.get_actor_index(
                    env_ptr, franka_actor, gymapi.DOMAIN_SIM
                )
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
            target_actor = self.gym.create_actor(
                env_ptr, target_asset, target_start_pose, "target", i, 2, 0
            )
            self.gym.set_rigid_body_color(
                env_ptr,
                target_actor,
                0,
                gymapi.MESH_VISUAL,
                gymapi.Vec3(1.0, 0.0, 0.0),
            )
            indexes_sim_target.append(
                self.gym.get_actor_index(
                    env_ptr, target_actor, gymapi.DOMAIN_SIM
                )
            )

            self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)
            self.frankas.append(franka_actor)
            self.targets.append(target_actor)

        self.indexes_sim_franka = torch.tensor(
            indexes_sim_franka, dtype=torch.int32, device=self.device
        )
        self.indexes_sim_target = torch.tensor(
            indexes_sim_target, dtype=torch.int32, device=self.device
        )

        # Setup data
        self.init_data()

    def init_data(self):
        # Setup sim handles
        env_ptr = self.envs[0]
        franka_handle = self.frankas[0]
        target_handle = self.targets[0]

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
                env_ptr, target_handle, "target"
            ),
        }

        # Franka defaults
        self.franka_default_dof_pos = torch.tensor(
            np.radians([0, -45, 0, -135, 0, 90, 45, 2, 2]),
            device=self.device,
            dtype=torch.float32,
        )
        self.franka_dof_targets = torch.zeros(
            (self.num_envs, self.num_franka_dofs),
            dtype=torch.float32,
            device=self.device,
        )
        self.franka_dof_gripper = torch.zeros(
            (self.num_envs, 2), dtype=torch.float32, device=self.device
        )

        # Setup tensor buffers and views: roots, DOFs, rigid bodies.
        root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(
            self.sim
        )

        if self.control_type == "cartesian":
            jacobian_tensor = self.gym.acquire_jacobian_tensor(
                self.sim, "franka"
            )

        # self._refresh()

        self.root_state = gymtorch.wrap_tensor(root_state_tensor)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state_tensor)

        if self.control_type == "cartesian":
            self.jacobian = gymtorch.wrap_tensor(jacobian_tensor)

        # Root states
        self.root_pos = self.root_state[:, :3].view(self.num_envs, -1, 3)
        self.root_rot = self.root_state[:, 3:7].view(self.num_envs, -1, 4)
        self.root_vel_lin = self.root_state[:, 7:10].view(self.num_envs, -1, 3)
        self.root_vel_ang = self.root_state[:, 10:13].view(
            self.num_envs, -1, 3
        )

        # DoF states
        self.dof_pos = self.dof_state.view(self.num_envs, -1, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, -1, 2)[..., 1]

        # Rigid body states
        self.rigid_body_pos = self.rigid_body_state[:, :3].view(
            self.num_envs, -1, 3
        )
        self.rigid_body_rot = self.rigid_body_state[:, 3:7].view(
            self.num_envs, -1, 4
        )
        self.rigid_body_vel_lin = self.rigid_body_state[:, 7:10].view(
            self.num_envs, -1, 3
        )
        self.rigid_body_vel_ang = self.rigid_body_state[:, 10:13].view(
            self.num_envs, -1, 3
        )

        if self.control_type == "cartesian":
            hand_joint_index = self.gym.get_actor_joint_dict(
                env_ptr, franka_handle
            )["panda_hand_joint"]
            self.jacobian_eef = self.jacobian[:, hand_joint_index, :, :7]

        # TODO: Reset P&P cube maybe need this
        self._global_indices = torch.arange(
            self.num_envs * 4, dtype=torch.int32, device=self.device
        ).view(self.num_envs, -1)

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

    def _update_states(self):
        self.states.update(
            {
                # Franka
                "dof_pos": self.dof_pos[:, :],
                "dof_gripper": self.dof_pos[:, -2:],
                # End effector
                "eef_pos": self.rigid_body_pos[:, self.handles["grip_site"]],
                "eef_rot": self.rigid_body_rot[:, self.handles["grip_site"]],
                "eef_vel_lin": self.rigid_body_vel_lin[
                    :, self.handles["grip_site"]
                ],
                "ee_vel_ang": self.rigid_body_vel_ang[
                    :, self.handles["grip_site"]
                ],
                "eef_lf_pos": self.rigid_body_pos[
                    :, self.handles["leftfinger_tip"]
                ],
                "eef_rf_pos": self.rigid_body_pos[
                    :, self.handles["rightfinger_tip"]
                ],
                # Target
                "target_pos": self.root_pos[:, self.handles["target"]],
                "target_rot": self.root_rot[:, self.handles["target"]],
            }
        )

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        if self.control_type == "cartesian":
            self.gym.refresh_jacobian_tensors(self.sim)

        # Refresh states
        self._update_states()

    def _draw_debug_vis(self):
        """Draws visualizations for dubugging (slows down simulation a lot).
        Default behaviour: draws height measurement points
        """
        # draw height lines
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        visualize_sphere = {
            "eef_pos": gymutil.WireframeSphereGeometry(
                0.02,
                8,
                8,
                None,
                color=(0, 1, 0),
            ),
            "target_pos": gymutil.WireframeSphereGeometry(
                0.04,
                8,
                8,
                None,
                color=(0, 0, 1),
            ),
        }

        for i in range(self.num_envs):
            for pose, geom in visualize_sphere.items():
                x, y, z = self.states[pose][i].cpu().numpy()
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(
                    geom, self.gym, self.viewer, self.envs[i], sphere_pose
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
        ##################################################################
        # Reset franka
        ##################################################################
        num_resets = len(env_ids)
        dof_noise = torch.rand(
            (num_resets, self.num_franka_dofs), device=self.device
        )
        pos = self.franka_default_dof_pos.unsqueeze(
            0
        ) + self.franka_dof_noise * 2.0 * (dof_noise - 0.5)
        pos = tensor_clamp(
            pos, self.franka_dof_lower_limits, self.franka_dof_upper_limits
        )

        # Overwrite gripper init pos
        # (no noise since these are always position controlled)
        pos[:, -2:] = self.franka_default_dof_pos[-2:]
        # pos[:, -2:] = 0.0 # close gripper

        self.franka_dof_targets[env_ids, :] = pos[:]
        self.dof_pos[env_ids, :] = pos[:]
        self.dof_vel[env_ids, :] = 0.0

        indexes = self.indexes_sim_franka[env_ids]
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.franka_dof_targets),
            gymtorch.unwrap_tensor(indexes),
            len(env_ids),
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(indexes),
            len(env_ids),
        )

        ##################################################################
        # Reset target
        ##################################################################
        pos = torch.zeros(num_resets, 3, device=self.device)
        pos_noise = torch.rand((num_resets, 3), device=self.device)

        # Sampling xy is "centered" around middle of table
        centered_target_xy_state = torch.tensor(
            self._table_surface_pos[:2],
            device=self.device,
            dtype=torch.float32,
        )
        pos[:, :2] = centered_target_xy_state.unsqueeze(
            0
        ) + self.start_position_noise * 2.0 * (pos_noise[:, :2] - 0.5)

        # Set z value, minimum is the start_position_noise
        pos[:, 2] = (
            self._target_start_pos[2]
            + self.start_position_noise * 2.0 * pos_noise[:, 2]
        )
        self.root_pos[env_ids, self.handles["target"], :] = pos[:]

        indexes = self.indexes_sim_target[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state),
            gymtorch.unwrap_tensor(indexes),
            len(env_ids),
        )

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)

        # TODO: Implement gripper control from actions (currently not used)
        # Split arm and gripper command
        # u_arm, u_gripper = self.actions[:, :-1], self.actions[:, -1]

        u_arm = self.actions
        u_gripper = torch.ones(
            self.num_envs, dtype=torch.float32, device=self.device,
        )

        # Control arm (scale value first)
        if self.control_type == "joint":
            targets_arm = (
                self.franka_dof_targets[:, :7]
                + self.franka_dof_speed_scales[:7]
                * self.dt
                * u_arm
                * self._action_scale
            )

        # Control gripper, last actions from the network
        # is a boolean to open/close gripper
        # + open - close
        u_fingers = torch.zeros(
            (self.num_envs, 2), dtype=torch.float32, device=self.device
        )
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
        targets_gripper = u_fingers[:]

        targets = torch.cat([targets_arm, targets_gripper], dim=-1)
        self.franka_dof_targets = tensor_clamp(
            targets, self.franka_dof_lower_limits, self.franka_dof_upper_limits
        )

        # Deploy actions
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self.franka_dof_targets)
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
