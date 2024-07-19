import os
from typing import Dict

import numpy as np
import torch
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *
from isaacgymenvs.utils.torch_jit_utils import *
from isaacgymenvs.utils.utils import sample_frictions

from utils import motion_imitation_utils as miu

from .base.vec_task import VecTask


class ImitationMotionA1(VecTask):

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

        self.elapsed_time = 0.0
        self._load_cfg_params()

        self.cfg["env"]["numObservations"] = (
            self.p_state_dim + self.p_action_dim + 48 + 8
        )
        self.cfg["env"]["numActions"] = self.p_action_dim
        self.cfg["headless"] = headless

        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )

        if not headless:
            self.init_debug()

        self._setup_utility_tensors()

    def init_debug(self):
        """Initializes miscellaneous debugging properties."""
        # Focus viewer's camera on the first environment.
        self.debug_cam_pos = gymapi.Vec3(0.7, 1.5, 0.7)
        self.debug_cam_target = gymapi.Vec3(0.5, 0.0, 0)
        self.debug_cam_offset = [0.0, -1.0, 0.20]
        self.gym.viewer_camera_look_at(
            self.viewer, None, self.debug_cam_pos, self.debug_cam_target
        )

        self.flag_camera_follow = self.cfg["render"].get("enableCameraFollow", False)
        self.i_follow_env = self.cfg["render"].get("cameraFollowEnvId", 0)
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_T, "camera_follow"
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_Y, "follow_env_prev"
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_U, "follow_env_next"
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_R, "reset_envs"
        )

        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_W, "update_cam_pos_w"
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_S, "update_cam_pos_s"
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_A, "update_cam_pos_a"
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_D, "update_cam_pos_d"
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_Q, "update_cam_pos_q"
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_E, "update_cam_pos_e"
        )

    def print_info(self):
        rigid_body_dict = self.gym.get_actor_rigid_body_dict(
            self.envs[0], self.sim_actors[0]
        )
        print(f"Number of rigid bodies: {len(rigid_body_dict)}.")
        print("#### Rigid Bodies Info ####")
        for k, v in rigid_body_dict.items():
            props = self.gym.get_actor_rigid_body_properties(
                self.envs[0], self.sim_actors[0]
            )[v]
            print(
                f"[{v}]: `{k}`. Mass: {props.mass:.3f} | InvMass: {props.invMass:.3f}"
            )

        dof_props = self.gym.get_actor_dof_properties(self.envs[0], self.sim_actors[0])
        print(dof_props)
        print("#" * 80)
        input("freeze")

    def _load_cfg_params(self):
        self.motion_fn = self.cfg["env"]["motionFn"]
        print(f"Will load motion `{self.motion_fn}`.")
        self.p_max_time = self.cfg["env"]["maxTime"]
        self.p_kp = self.cfg["env"]["kp"]
        self.p_kd = self.cfg["env"]["kd"]
        self.p_root_reset_dist = self.cfg["env"]["rootResetDist"]
        self.p_ref_motion_z_offset = self.cfg["env"]["ref_motion_z_offset"]
        if self.p_root_reset_dist > 3.0:
            print(
                "** WARNING ** `root_reset_dist` seems very large. This is commonly used for visualization."
            )
        self.p_weight_dof_pos = self.cfg["env"]["weight_dof_pos"]
        self.p_weight_dof_vel = self.cfg["env"]["weight_dof_vel"]
        self.p_weight_ef = self.cfg["env"]["weight_ef"]
        self.p_weight_root_pose = self.cfg["env"]["weight_root_pose"]
        self.p_weight_root_vel = self.cfg["env"]["weight_root_vel"]

        self.p_scale_dof_pos = self.cfg["env"]["scale_dof_pos"]
        self.p_scale_dof_vel = self.cfg["env"]["scale_dof_vel"]
        self.p_scale_ef = self.cfg["env"]["scale_ef"]
        self.p_scale_root_pose = self.cfg["env"]["scale_root_pose"]
        self.p_scale_root_vel = self.cfg["env"]["scale_root_vel"]

        self.dt = self.cfg["sim"]["dt"]
        self.sim_rate = 1 / self.dt
        self.p_num_sticky_actions = self.cfg["env"]["controlFrequencyInv"]
        self.p_state_dim = self.cfg["env"]["state_dim"]
        self.p_action_dim = self.cfg["env"]["action_dim"]

        self.p_action_scale = self.cfg["env"]["action_scale"]

        self.p_rand_static_friction = self.cfg["env"]["rand_static_friction"]
        self.p_rand_restitution = self.cfg["env"]["rand_restitution"]
        self.p_feet_contact_threshold = self.cfg["env"]["feet_contact_threshold"]
        self.p_reset_contact_thresh = self.cfg["env"]["reset_contact_thresh"]

        # Hides reference character and disables reset conditions.
        self.p_test_viz_mode = self.cfg["env"]["testVizMode"]
        if self.p_test_viz_mode:
            self.p_root_reset_dist = 1e10
            self.p_reset_contact_thresh = 1e10

        if "logging" in self.cfg:
            self.enable_logging = self.cfg["logging"]["enabled"]
            self.max_logging_episodes = self.cfg["logging"]["num_episodes"]
            self.min_logging_episode_steps = self.cfg["logging"]["min_episode_steps"]
            self.logging_save_period = self.cfg["logging"]["save_period"]
            self.logging_save_path = self.cfg["logging"]["save_path"]
            self.record_obs = self.cfg["logging"]["record_obs"].split(",")
            self.record_ret = self.cfg["logging"]["record_ret"].split(",")
        else:
            self.enable_logging = False
            self.max_logging_episodes = 0
            self.min_logging_episode_steps = 0
            self.logging_save_period = 0
            self.logging_save_path = ""
            self.record_obs = []
            self.record_ret = []

        self.p_obs_scale_dict = {
            "linvel": self.cfg["env"]["obs_scales"]["linvel"],
            "angvel": self.cfg["env"]["obs_scales"]["angvel"],
        }

        # Action logs.
        self.test_mode = self.cfg["env"]["testMode"]
        if not self.test_mode:
            return

        self.checkpoint_path = "/".join(self.cfg["env"]["checkpoint"].split("/")[:-2])
        self.checkpoint_name = self.checkpoint_path.split("/")[-1]

    def _setup_utility_tensors(self):
        """Creates tensors used to read and modify the actors' states."""

        # Robot actions are offset from these default PD targets.
        self.default_joints_pose = to_torch(
            [0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8],
            device=self.device,
            dtype=torch.float32,
        )

        # The current reference motion index, per actor.
        self.ref_motion_index = torch.zeros_like(self.progress_buf)

        # Indexes of kinematic and learner actors
        self.kin_actors_index = torch.arange(
            0, 2 * self.num_envs, 2, dtype=torch.int32
        )  # 0, 2, 4, 6, ...
        self.sim_actors_index = self.kin_actors_index + 1  # 1, 3, 5, 7, ..
        self.kin_actors_index = self.kin_actors_index.to(self.device)
        self.sim_actors_index = self.sim_actors_index.to(self.device)

        # Root state Tensor.
        self._root_tensor = self.gym.acquire_actor_root_state_tensor(
            self.sim
        )  # Num Actors x 13
        self.root_tensor = gymtorch.wrap_tensor(self._root_tensor)
        self.root_tensor_kin = self.root_tensor.view(self.num_envs, 2, 13)[:, 0, :]
        self.root_tensor_sim = self.root_tensor.view(self.num_envs, 2, 13)[:, 1, :]

        # Useful views of the root tensor
        self.root_positions = self.root_tensor[:, 0:3]
        self.root_orientations = self.root_tensor[:, 3:7]
        self.root_linear_vel = self.root_tensor[:, 7:10]
        self.root_angular_vel = self.root_tensor[:, 10:13]

        self.root_sim_positions = self.root_tensor_sim[:, 0:3]
        self.root_sim_orientations = self.root_tensor_sim[:, 3:7]
        self.root_sim_linear_vel = self.root_tensor_sim[:, 7:10]
        self.root_sim_angular_vel = self.root_tensor_sim[:, 10:13]

        self.root_kin_positions = self.root_tensor_kin[:, 0:3]
        self.root_kin_orientations = self.root_tensor_kin[:, 3:7]
        self.root_kin_linear_vel = self.root_tensor_kin[:, 7:10]
        self.root_kin_angular_vel = self.root_tensor_kin[:, 10:13]

        self._rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_tensor = gymtorch.wrap_tensor(self._rigid_body_tensor)
        self.rb_kin_tensor = self.rb_tensor.view(self.num_envs, 2, self.num_rigid_body, 13)[:, 0, :]
        self.rb_sim_tensor = self.rb_tensor.view(self.num_envs, 2, self.num_rigid_body, 13)[:, 1, :]
        self.rb_kin_pos_tensor = self.rb_tensor.view(self.num_envs, 2, self.num_rigid_body, 13)[:, 0, :, :3]
        self.rb_sim_pos_tensor = self.rb_tensor.view(self.num_envs, 2, self.num_rigid_body, 13)[:, 1, :, :3]
        self.rb_sim_quat_tensor = self.rb_tensor.view(self.num_envs, 2, self.num_rigid_body, 13)[:, 1, :, 3:7]

        # Relevant indexes of rigid bodies.
        self.rb_feet_indexes = to_torch(
            [4, 8, 12, 16], dtype=torch.long, device=self.device
        )

        # PD Targets tensor. Modified at every loop to send commands to the
        # actors. First 12 indexes correspond to Kinematic actors. Last 12 indexes
        # correspond to learner actors.
        self.pd_targets_tensor = torch.zeros(self.num_envs, 24, dtype=torch.float32).to(
            self.device
        )

        self._state_dof = self.gym.acquire_dof_state_tensor(self.sim)
        self.state_dof_tensor = gymtorch.wrap_tensor(self._state_dof)
        self.dof_pos_kin_actors = self.state_dof_tensor.view(self.num_envs, 2, self.num_dof, 2)[:, 0, :, 0]
        self.dof_pos_sim_actors = self.state_dof_tensor.view(self.num_envs, 2, self.num_dof, 2)[:, 1, :, 0]
        self.dof_vel_kin_actors = self.state_dof_tensor.view(self.num_envs, 2, self.num_dof, 2)[:, 0, :, 1]
        self.dof_vel_sim_actors = self.state_dof_tensor.view(self.num_envs, 2, self.num_dof, 2)[:, 1, :, 1]

        # Useful unit vectors pointing forward and upwards.
        self.z_basis_vec = to_torch([0, 0, 1], device=self.device).repeat(
            (self.num_envs, 1)
        )

        # Contact force tensors.
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(
            self.num_envs, 2, -1, 3
        )
        self.contact_forces_sim = self.contact_forces[:, 1, :, :]

        self.actions = torch.zeros(
            (self.num_envs, 12), dtype=torch.float32, device=self.device
        )
        self.prev_actions = torch.zeros(
            (self.num_envs, 12), dtype=torch.float32, device=self.device
        )

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # Log episode information.
        torch_zeros = lambda: torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.episode_sums = {
            "rew_total": torch_zeros(),
            "rew_dof_pos": torch_zeros(),
            "rew_dof_vel": torch_zeros(),
            "rew_ef_pose": torch_zeros(),
            "rew_root_pose": torch_zeros(),
            "rew_root_vel": torch_zeros(),
        }

        self.action_list_trajectories = []
        self.dof_list_trajectories = []

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.dt = self.dt
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        env_limits = self._create_envs(
            self.num_envs, self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs))
        )
        self._create_ground_plane(env_limits)

        # Extra reward buffer to check/debug each element of the reward.
        self.rew_debug = torch.zeros(
            (self.num_envs, 5), dtype=torch.float32, device=self.device
        )

    def _load_motion(self, motion_path, max_time):
        """Loads a reference motion from disk. Pre-generates all frames and push
        them to GPU for later use.

        """
        print("Loading motion data...")
        self.motion = miu.MotionData(motion_path)
        print(f"\tFrames: {self.motion.get_num_frames()}")
        print(f"\tFrame duration: {self.motion.get_frame_duration()}")

        step_size = self.dt * self.p_num_sticky_actions
        self.motion_length = self.motion.get_num_frames()

        # Pre-generate all frames for the whole episode + some extra cycles.
        # The extra cycles are needed because the robot is reset to a random
        # reference index between 0 and 2 cycles.
        time_axis = np.arange(
            0, max_time + 5 * step_size * self.motion_length, step_size
        )
        print(f"\tTime_axis: {time_axis.shape}")

        self.np_pose_frames = []
        self.np_vel_frames = []
        for t in time_axis:
            pose = self.motion.calc_frame(t)
            vels = self.motion.calc_frame_vel(t)
            # NOTE: Order of joints in Isaac Gym differs from PyBullet.
            # PyBullet: FR, FL, RR, RL | IsaacGym: FL, FR, RL, RR.
            reordered_pose = np.concatenate(
                [
                    pose[:7],  # XYZ + Quat (No change).
                    pose[10:13],
                    pose[7:10],
                    pose[16:19],
                    pose[13:16],  # Reordered joint pos.
                ]
            )

            reordered_vels = np.concatenate(
                [
                    vels[:6],  # Lin and ang vel (No change).
                    vels[9:12],
                    vels[6:9],
                    vels[15:18],
                    vels[12:15],
                ]
            )

            self.np_pose_frames.append(reordered_pose)
            self.np_vel_frames.append(reordered_vels)

        self.np_pose_frames = np.array(self.np_pose_frames)
        self.np_vel_frames = np.array(self.np_vel_frames)
        print(f"\tPose frames: {self.np_pose_frames.shape}")
        print(f"\tVel frames: {self.np_vel_frames.shape}")

        # Offset reference motion Z axis. Used to unstuck the reference motion
        # from the ground.
        self.np_pose_frames[:, 2] += self.p_ref_motion_z_offset
        assert self.np_pose_frames.shape[0] == self.np_vel_frames.shape[0]

        # Animation length also defines the maximum episode length
        # Makes sure episode finished before we run out of future frames to index
        # in the observations.
        self.max_episode_length = (
            self.np_pose_frames.shape[0] - 4 * self.motion_length - 1
        )
        print(f"Max episode length is {self.max_episode_length}.")
        self.start_logs_time = 1.0
        self.duration_logs_time = 2.0

        # Convert to PyTorch GPU tensors.
        self.tensor_ref_pose = torch.tensor(
            self.np_pose_frames, dtype=torch.float32, device=self.device
        )
        self.tensor_ref_vels = torch.tensor(
            self.np_vel_frames, dtype=torch.float32, device=self.device
        )

        # Create other useful views.
        self.tensor_ref_root_pose = self.tensor_ref_pose[:, :7]  # XYZ + Quat
        self.tensor_ref_pd_targets = self.tensor_ref_pose[:, 7:]  # 12 joints
        self.tensor_ref_root_vels = self.tensor_ref_vels[
            :, :6
        ]  # Linear XYZ + Angular XYZ
        self.tensor_ref_pd_vels = self.tensor_ref_vels[:, 6:]

        # Used to sync the postion of kin character to sim character by offseting
        # its position.
        self.tensor_ref_offset_pos = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float32
        )

        lookahead_secs = [0.0333, 0.0666, 0.3333, 1.0]  # Lookahead time in seconds.
        lookahead_inds = [int(s * (1 / step_size) + 0.5) for s in lookahead_secs]
        # Used to increment from current index to get future target poses from
        # the reference motion.
        self.target_pose_inc_indices = torch.tensor(
            lookahead_inds, dtype=torch.long, device=self.device
        )

    def _create_ground_plane(self, env_limits):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        sf, df, rs = sample_frictions(
            self.p_rand_static_friction, self.p_rand_restitution
        )
        assert df < sf  # Safety check.
        plane_params.static_friction = sf
        plane_params.dynamic_friction = df
        plane_params.restitution = rs
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        asset_root = "../assets"
        motions_root = "motions/"
        self._load_motion(
            os.path.join(motions_root, self.motion_fn), max_time=self.p_max_time
        )

        # NOTE: The model without collision is causing the new IsaacGym version to crash.
        # We should investigate this.
        asset_fn = "urdf/a1_description/urdf/a1_alt_no_collision.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = True
        asset_options.use_mesh_materials = False
        asset_options.replace_cylinder_with_capsule = True
        asset_options.collapse_fixed_joints = True
        asset_options.density = 0.001
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.armature = 0.0
        asset_options.thickness = 0.01
        kin_asset = self.gym.load_asset(self.sim, asset_root, asset_fn, asset_options)

        asset_fn = "urdf/a1_description/urdf/a1_alt.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.flip_visual_attachments = True
        asset_options.use_mesh_materials = True
        asset_options.replace_cylinder_with_capsule = True
        asset_options.collapse_fixed_joints = True
        asset_options.density = 0.001
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.armature = 0.0
        asset_options.thickness = 0.01
        sim_asset = self.gym.load_asset(self.sim, asset_root, asset_fn, asset_options)

        self.num_dof = self.gym.get_asset_dof_count(sim_asset)
        dof_properties = self.gym.get_asset_dof_properties(sim_asset)
        self.num_rigid_body = self.gym.get_asset_rigid_body_count(sim_asset)
        self.dof_lower_limits = []
        self.dof_upper_limits = []
        self.dof_torque_limits = []
        for i in range(self.num_dof):
            self.dof_lower_limits.append(dof_properties["lower"][i])
            self.dof_upper_limits.append(dof_properties["upper"][i])
            self.dof_torque_limits.append(dof_properties["effort"][i])
        self.dof_lower_limits = to_torch(self.dof_lower_limits, device=self.device)
        self.dof_upper_limits = to_torch(self.dof_upper_limits, device=self.device)
        self.dof_torque_limits = to_torch(self.dof_torque_limits, device=self.device)
        self.num_rigid_body = self.gym.get_asset_rigid_body_count(sim_asset)
        print(f"Loaded asset has {self.num_rigid_body} rigid bodies.")

        self.envs = []
        # self.kin_actors = []
        # self.sim_actors = []

        env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(0.0, 0.0, 0.6)
        start_pose.r = gymapi.Quat(0.5, 0.5, 0.5, 0.5)

        # Figure out the delimitations of the envs in global coordinates.
        min_env_x = float("inf")
        max_env_x = -float("inf")
        min_env_y = float("inf")
        max_env_y = -float("inf")
        # Stores the origin of each environment. Used to compute Global position when using Tensor API.
        self.envs_origin_global_pos = torch.zeros(
            (num_envs, 3), device=self.device, dtype=torch.float32, requires_grad=False
        )
        for i in range(num_envs):
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)

            # Create kinematic actor
            kin_actor = self.gym.create_actor(
                env, kin_asset, start_pose, "kinematic_actor", i, 1
            )
            props = self.gym.get_actor_dof_properties(env, kin_actor)
            props["driveMode"][:] = gymapi.DOF_MODE_POS
            props["stiffness"][:] = self.p_kp
            props["damping"][:] = self.p_kd
            self.gym.set_actor_dof_properties(env, kin_actor, props)
            # Paint kinematic actor as yellow.
            color = gymapi.Vec3()
            color.x = 1.0
            color.y = 1.0
            color.z = 0.0
            for i_rb in range(self.num_rigid_body):
                self.gym.set_rigid_body_color(
                    env,
                    kin_actor,
                    i_rb,
                    gymapi.MeshType.MESH_VISUAL_AND_COLLISION,
                    color,
                )

            # Create the learner actor
            sim_actor = self.gym.create_actor(
                env, sim_asset, start_pose, "learner_actor", i, 1
            )
            props = self.gym.get_actor_dof_properties(env, sim_actor)
            props["driveMode"][:] = gymapi.DOF_MODE_POS
            props["stiffness"][:] = self.p_kp
            props["damping"][:] = self.p_kd
            self.gym.set_actor_dof_properties(env, sim_actor, props)

            self.envs.append(env)
            # self.kin_actors.append(kin_actor)
            # self.sim_actors.append(sim_actor)

            orig_pos = self.gym.get_env_origin(env)
            self.envs_origin_global_pos[i, 0] = orig_pos.x
            self.envs_origin_global_pos[i, 1] = orig_pos.y
            self.envs_origin_global_pos[i, 2] = orig_pos.z
            if orig_pos.x > max_env_x:
                max_env_x = orig_pos.x
            if orig_pos.x < min_env_x:
                min_env_x = orig_pos.x
            if orig_pos.y > max_env_y:
                max_env_y = orig_pos.y
            if orig_pos.y < min_env_y:
                min_env_y = orig_pos.y

        return ((min_env_x, max_env_x), (min_env_y, max_env_y))

    def keyboard(self, event):
        if event.action == "camera_follow" and event.value > 0:
            self.flag_camera_follow = not self.flag_camera_follow
        elif event.action == "follow_env_prev" and event.value > 0:
            self.i_follow_env = max(0, self.i_follow_env - 1)
        elif event.action == "follow_env_next" and event.value > 0:
            self.i_follow_env = min(self.i_follow_env + 1, self.num_envs - 1)
        elif event.action == "reset_envs" and event.value > 0:
            self.reset_buf[:] = 1
        elif event.action == "update_cam_pos_w" and event.value > 0:
            self.debug_cam_offset[1] += 0.25
        elif event.action == "update_cam_pos_s" and event.value > 0:
            self.debug_cam_offset[1] -= 0.25
        elif event.action == "update_cam_pos_a" and event.value > 0:
            self.debug_cam_offset[0] -= 0.25
        elif event.action == "update_cam_pos_d" and event.value > 0:
            self.debug_cam_offset[0] += 0.25
        elif event.action == "update_cam_pos_q" and event.value > 0:
            self.debug_cam_offset[2] += 0.25
        elif event.action == "update_cam_pos_e" and event.value > 0:
            self.debug_cam_offset[2] -= 0.25

    def viewer_update(self):
        if self.flag_camera_follow:
            self.update_debug_camera()

    def update_debug_camera(self):
        actor_pos = self.root_kin_positions.cpu().numpy()[self.i_follow_env]

        spacing = self.cfg["env"]["envSpacing"]
        row = int(np.sqrt(self.num_envs))
        if row > 1:
            x = self.i_follow_env % row
            y = (self.i_follow_env - x) / row
        else:
            x = self.i_follow_env % 2
            y = (self.i_follow_env - x) / 2
        env_offset = [x * 2 * spacing, y * spacing, 0.0]

        # Smooth the camera movement with a moving average.
        k_smooth = 0.9
        new_cam_pos = gymapi.Vec3(
            actor_pos[0] + self.debug_cam_offset[0] + env_offset[0],
            actor_pos[1] + self.debug_cam_offset[1] + env_offset[1],
            actor_pos[2] + self.debug_cam_offset[2] + env_offset[2],
        )
        new_cam_target = gymapi.Vec3(
            actor_pos[0] + env_offset[0],
            actor_pos[1] + env_offset[1],
            actor_pos[2] + env_offset[2],
        )

        self.debug_cam_pos.x = (
            k_smooth * self.debug_cam_pos.x + (1 - k_smooth) * new_cam_pos.x
        )
        self.debug_cam_pos.y = (
            k_smooth * self.debug_cam_pos.y + (1 - k_smooth) * new_cam_pos.y
        )
        self.debug_cam_pos.z = (
            k_smooth * self.debug_cam_pos.z + (1 - k_smooth) * new_cam_pos.z
        )

        self.debug_cam_target.x = (
            k_smooth * self.debug_cam_target.x + (1 - k_smooth) * new_cam_target.x
        )
        self.debug_cam_target.y = (
            k_smooth * self.debug_cam_target.y + (1 - k_smooth) * new_cam_target.y
        )
        self.debug_cam_target.z = (
            k_smooth * self.debug_cam_target.z + (1 - k_smooth) * new_cam_target.z
        )

        self.gym.viewer_camera_look_at(
            self.viewer, None, self.debug_cam_pos, self.debug_cam_target
        )

    def _get_feet_contacts(self):
        fl_foot_contact = (
            torch.norm(self.contact_forces_sim[:, self.rb_feet_indexes[0], :], dim=1)
            >= self.p_feet_contact_threshold
        )
        fr_foot_contact = (
            torch.norm(self.contact_forces_sim[:, self.rb_feet_indexes[1], :], dim=1)
            >= self.p_feet_contact_threshold
        )
        rl_foot_contact = (
            torch.norm(self.contact_forces_sim[:, self.rb_feet_indexes[2], :], dim=1)
            >= self.p_feet_contact_threshold
        )
        rr_foot_contact = (
            torch.norm(self.contact_forces_sim[:, self.rb_feet_indexes[3], :], dim=1)
            >= self.p_feet_contact_threshold
        )

        return torch.cat(
            [
                fl_foot_contact.unsqueeze(-1),
                fr_foot_contact.unsqueeze(-1),
                rl_foot_contact.unsqueeze(-1),
                rr_foot_contact.unsqueeze(-1),
            ],
            dim=1,
        )

    def compute_observations(self):
        # Update state of the tensors
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        feet_contacts = self._get_feet_contacts()
        self.obs_buf[:] = compute_observations_mi(
            self.root_tensor_sim,
            self.dof_pos_sim_actors,
            self.actions,
            self.ref_motion_index,
            self.tensor_ref_pose,
            self.target_pose_inc_indices,
            self.z_basis_vec,
            feet_contacts,
            self.p_obs_scale_dict,
        )

    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:], self.rew_debug[:] = compute_reward(
            self.progress_buf,
            self.reset_buf,
            self.max_episode_length,
            self.dof_pos_kin_actors,
            self.dof_pos_sim_actors,
            self.dof_vel_kin_actors,
            self.dof_vel_sim_actors,
            self.rb_kin_pos_tensor,
            self.rb_sim_pos_tensor,
            self.rb_feet_indexes,
            self.root_kin_positions,
            self.root_sim_positions,
            self.root_kin_orientations,
            self.root_sim_orientations,
            self.root_kin_linear_vel,
            self.root_sim_linear_vel,
            self.root_kin_angular_vel,
            self.root_sim_angular_vel,
            self.p_root_reset_dist,
            self.p_weight_dof_pos,
            self.p_scale_dof_pos,
            self.p_weight_dof_vel,
            self.p_scale_dof_vel,
            self.p_weight_ef,
            self.p_scale_ef,
            self.p_weight_root_pose,
            self.p_scale_root_pose,
            self.p_weight_root_vel,
            self.p_scale_root_vel,
            self.z_basis_vec,
            self.contact_forces_sim,
            self.p_reset_contact_thresh,
        )

    def pre_physics_step(self, actions):
        self.prev_actions = self.actions.clone()
        # Store current step actions. used in Motion Imitation style observations.
        self.actions = self.default_joints_pose + (
            actions * self.p_action_scale * 3.1415
        )

        # Clamp values
        self.actions = tensor_clamp(
            self.actions, self.dof_lower_limits, self.dof_upper_limits
        )

        # Set the learner actor actions - PD target control.
        # Num_envs x 24
        self.pd_targets_tensor[:, 12:] = self.actions

        # Set PD targets for ALL actors.
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self.pd_targets_tensor)
        )

    def reset_envs(self):
        # Prepare all index tensors related to resetting environments.
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()  # num_envs x 1
        if len(reset_env_ids) == 0:
            return torch.tensor([], dtype=torch.int32, device=self.device)

        # Log terminating episode data.
        self.extras["episode"] = {}
        self.extras["episode"]["average_length"] = torch.mean(
            self.progress_buf[reset_env_ids].float()
        )
        for key in self.episode_sums.keys():
            self.extras["episode"][key] = torch.mean(
                self.episode_sums[key][reset_env_ids] / self.progress_buf[reset_env_ids]
            )
            self.episode_sums[key][reset_env_ids] = 0.0

        reset_env_ids_int32 = reset_env_ids.to(torch.int32)
        reset_kin_actors_ids = reset_env_ids_int32 * 2
        reset_sim_actors_ids = reset_kin_actors_ids + 1
        reset_actor_indexes = torch.cat([reset_kin_actors_ids, reset_sim_actors_ids])

        #### Reference State Initialization ####
        # Generate a random keyframe (index) of the reference motion between 0
        # and the first two cycles.
        # reset_i = torch.randint_like(reset_env_ids, low=0,
        # high=self.motion_length * 2)

        reset_i = torch.zeros_like(reset_env_ids)
        self.ref_motion_index[reset_env_ids] = reset_i

        self.progress_buf[reset_env_ids] = 0
        self.reset_buf[reset_env_ids] = 0
        # Reset cycle position sync accumulator.
        self.tensor_ref_offset_pos[reset_env_ids, :] = 0.0

        self.root_sim_positions[reset_env_ids] = self.tensor_ref_root_pose[reset_i, :3]
        self.root_sim_orientations[reset_env_ids] = quat_unit(
            self.tensor_ref_root_pose[reset_i, 3:7]
        )
        self.root_sim_linear_vel[reset_env_ids] = self.tensor_ref_root_vels[reset_i, :3]
        self.root_sim_angular_vel[reset_env_ids] = self.tensor_ref_root_vels[
            reset_i, 3:
        ]

        self.root_tensor_kin[reset_env_ids, :7] = self.tensor_ref_root_pose[reset_i]
        self.root_tensor_kin[reset_env_ids, 7:] = self.tensor_ref_root_vels[reset_i]

        self.dof_pos_kin_actors[reset_env_ids] = self.tensor_ref_pd_targets[reset_i]
        self.dof_pos_sim_actors[reset_env_ids] = self.tensor_ref_pd_targets[reset_i]
        self.dof_vel_kin_actors[reset_env_ids] = self.tensor_ref_pd_vels[reset_i]
        self.dof_vel_sim_actors[reset_env_ids] = self.tensor_ref_pd_vels[reset_i]

        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.state_dof_tensor),
            gymtorch.unwrap_tensor(reset_actor_indexes),
            len(reset_actor_indexes),
        )

        # Fill history tensors with the reset frame.
        self.actions[reset_env_ids] = self.tensor_ref_pd_targets[reset_i]

        return reset_sim_actors_ids

    def update_actor_root_states(self, reset_sim_actor_ids):
        self._update_kin_chars_tensors()

        # Set the actor root state for the sim actors that need to reset and for ALL kinematic
        # actors.
        actor_indexes_to_update = torch.cat(
            [reset_sim_actor_ids, self.kin_actors_index]
        )
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_tensor),
            gymtorch.unwrap_tensor(actor_indexes_to_update),
            len(actor_indexes_to_update),
        )

    def post_physics_step(self):
        self.progress_buf += 1
        self.ref_motion_index += 1

        # Resets everything except sim actors root state tensor. We can only afford
        # one gymapi tensor function call, so it is updated together with the kin
        # actors in `update_actor_root_states`.
        reset_sim_actor_ids = self.reset_envs()

        self.update_actor_root_states(reset_sim_actor_ids)
        self.compute_observations()
        self.compute_reward()
        self.log_info()

    def log_info(self):
        self.episode_sums["rew_total"] += self.rew_buf
        self.episode_sums["rew_dof_pos"] += self.rew_debug[:, 0]
        self.episode_sums["rew_dof_vel"] += self.rew_debug[:, 1]
        self.episode_sums["rew_ef_pose"] += self.rew_debug[:, 2]
        self.episode_sums["rew_root_pose"] += self.rew_debug[:, 3]
        self.episode_sums["rew_root_vel"] += self.rew_debug[:, 4]

    def _update_kin_chars_tensors(self):
        curr_phase = self.progress_buf // self.motion_length > 0
        first_cycle_i = self.progress_buf % self.motion_length
        reset_phase = first_cycle_i == 0
        resync_env_ids = (
            curr_phase.logical_and(reset_phase).nonzero(as_tuple=False).flatten()
        )

        # Compute accumulated offset over episode. Only update on cycle ends.
        # Note: Offsets are only computed for x and y. z (height) is ignored.
        self.tensor_ref_offset_pos[resync_env_ids, :2] = (
            self.tensor_ref_root_pose[self.ref_motion_index][resync_env_ids, :2]
            - self.root_sim_positions[resync_env_ids, :2]
        )

        #### Updates kinematic characters ####
        # Position and orientation also comes from the reference motion
        self.root_tensor_kin[:, :3] = (
            self.tensor_ref_root_pose[self.ref_motion_index, :3]
            - self.tensor_ref_offset_pos
        )
        self.root_tensor_kin[:, 3:7] = self.tensor_ref_root_pose[
            self.ref_motion_index, 3:7
        ]

        # Velocity directly comes from the reference motion.
        self.root_tensor_kin[:, 7:] = self.tensor_ref_root_vels[self.ref_motion_index]
        self.pd_targets_tensor[:, :12] = self.tensor_ref_pd_targets[
            self.ref_motion_index
        ]


@torch.jit.script
def compute_reward(
    progress_buf: torch.Tensor,
    reset_buf: torch.Tensor,
    max_episode_length: int,
    dof_pos_kin: torch.Tensor,
    dof_pos_sim: torch.Tensor,
    dof_vel_kin: torch.Tensor,
    dof_vel_sim: torch.Tensor,
    rb_pos_kin: torch.Tensor,
    rb_pos_sim: torch.Tensor,
    rb_feet_indexes: torch.Tensor,
    root_kin_pos: torch.Tensor,
    root_sim_pos: torch.Tensor,
    root_kin_ori: torch.Tensor,
    root_sim_ori: torch.Tensor,
    root_kin_linvel: torch.Tensor,
    root_sim_linvel: torch.Tensor,
    root_kin_angvel: torch.Tensor,
    root_sim_angvel: torch.Tensor,
    root_reset_dist: float,
    weight_dof_pos: float,
    scale_dof_pos: float,
    weight_dof_vel: float,
    scale_dof_vel: float,
    weight_ef: float,
    scale_ef: float,
    weight_root_pose: float,
    scale_root_pose: float,
    weight_root_vel: float,
    scale_root_vel: float,
    z_basis_vec: torch.Tensor,
    contact_forces_sim: torch.Tensor,
    contact_forces_thresh: float,
):
    """Compute the motion imitation reward function."""

    # DoF pos reward.
    dof_pos_diff = torch.square(dof_pos_kin - dof_pos_sim)
    dof_pos_rew = torch.exp(-scale_dof_pos * dof_pos_diff.sum(dim=1))

    # DoF velocity reward.
    dof_vel_diff = torch.square(dof_vel_kin - dof_vel_sim)
    dof_vel_rew = torch.exp(-scale_dof_vel * dof_vel_diff.sum(dim=1))

    # End-effector position reward.
    height_err_scale = 3.0

    ef_pos_kin = rb_pos_kin[:, rb_feet_indexes]
    ef_pos_sim = rb_pos_sim[:, rb_feet_indexes]

    # Normalize end effector positions relative to root body
    ef_pos_kin = ef_pos_kin - root_kin_pos.unsqueeze(1)
    ef_pos_sim = ef_pos_sim - root_sim_pos.unsqueeze(1)

    # Angle around Z axis
    sim_angle = calc_heading(root_sim_ori)
    sim_inv_heading_rot = quat_from_angle_axis(sim_angle, z_basis_vec)
    kin_angle = calc_heading(root_kin_ori)
    kin_inv_heading_rot = quat_from_angle_axis(kin_angle, z_basis_vec)

    for idx in range(4):
        ef_pos_kin[:, idx] = quat_rotate(kin_inv_heading_rot, ef_pos_kin[:, idx])
        ef_pos_sim[:, idx] = quat_rotate(sim_inv_heading_rot, ef_pos_sim[:, idx])

    ef_diff_xy = torch.square(ef_pos_kin[:, :, :2] - ef_pos_sim[:, :, :2])
    ef_diff_xy = ef_diff_xy.sum(dim=2)
    ef_diff_z = height_err_scale * torch.square(
        ef_pos_kin[:, :, 2] - ef_pos_sim[:, :, 2]
    )
    ef_diff = (ef_diff_xy + ef_diff_z).sum(dim=1)
    ef_rew = torch.exp(-scale_ef * ef_diff)

    # Root pose reward. Position + Orientation.
    root_pos_diff = torch.square(root_kin_pos - root_sim_pos)
    root_pos_err = root_pos_diff.sum(dim=1)
    root_rot_diff = quat_mul(root_kin_ori, quat_conjugate(root_sim_ori))
    root_rot_diff = normalize(root_rot_diff)
    # axis-angle representation but we only care about the angle
    root_rot_diff_angle = normalize_angle(2 * torch.acos(root_rot_diff[:, 3]))
    root_rot_err = torch.square(root_rot_diff_angle)

    # Compound position and orientation error for root as in motion_imitation codebase.
    root_pose_err = root_pos_err + 0.5 * root_rot_err
    root_pose_rew = torch.exp(-scale_root_pose * root_pose_err)

    # Root velocity reward.
    root_linvel_diff = torch.square(root_kin_linvel - root_sim_linvel)
    root_linvel_err = root_linvel_diff.sum(dim=1)
    root_angvel_diff = torch.square(root_kin_angvel - root_sim_angvel)
    root_angvel_err = root_angvel_diff.sum(dim=1)
    root_vel_diff = root_linvel_err + 0.1 * root_angvel_err
    root_vel_rew = torch.exp(-scale_root_vel * root_vel_diff)

    # ##### Compute resulting reward. ######
    reward = (
        dof_pos_rew * weight_dof_pos
        + dof_vel_rew * weight_dof_vel
        + ef_rew * weight_ef
        + root_pose_rew * weight_root_pose
        + root_vel_rew * weight_root_vel
    )

    # ##### Reset logic. #####
    reset = torch.where(
        progress_buf >= max_episode_length - 1, 1, reset_buf
    )  # Reset if episode finished, congrats.
    reset = torch.where(
        root_pos_err >= root_reset_dist, 1, reset
    )  # Reset if position drifts too much
    reset = torch.where(
        root_rot_err >= root_reset_dist, 1, reset
    )  # Reset if orientation drifts too much

    # Check if root body is touching ground
    cn_root_body = torch.norm(contact_forces_sim[:, 0, :], dim=1)
    cn_fl_hip = torch.norm(contact_forces_sim[:, 1, :], dim=1)
    cn_fr_hip = torch.norm(contact_forces_sim[:, 5, :], dim=1)
    cn_rl_hip = torch.norm(contact_forces_sim[:, 9, :], dim=1)
    cn_rr_hip = torch.norm(contact_forces_sim[:, 13, :], dim=1)
    cn_fl_thigh = torch.norm(contact_forces_sim[:, 2, :], dim=1)
    cn_fr_thigh = torch.norm(contact_forces_sim[:, 6, :], dim=1)
    cn_rl_thigh = torch.norm(contact_forces_sim[:, 10, :], dim=1)
    cn_rr_thigh = torch.norm(contact_forces_sim[:, 14, :], dim=1)
    cn_fl_calf = torch.norm(contact_forces_sim[:, 3, :], dim=1)
    cn_fr_calf = torch.norm(contact_forces_sim[:, 7, :], dim=1)
    cn_rl_calf = torch.norm(contact_forces_sim[:, 11, :], dim=1)
    cn_rr_calf = torch.norm(contact_forces_sim[:, 15, :], dim=1)
    contact_res_hip = torch.max(
        torch.max(torch.max(cn_fl_hip, cn_fr_hip), cn_rl_hip), cn_rr_hip
    )
    contact_res_thigh = torch.max(
        torch.max(torch.max(cn_fl_thigh, cn_fr_thigh), cn_rl_thigh), cn_rr_thigh
    )
    contact_res_calf = torch.max(
        torch.max(torch.max(cn_fl_calf, cn_fr_calf), cn_rl_calf), cn_rr_calf
    )

    contact_res = torch.max(
        torch.max(contact_res_calf, torch.max(contact_res_hip, contact_res_thigh)),
        cn_root_body,
    )
    contact_res_reset = torch.logical_and(
        contact_res > contact_forces_thresh, progress_buf > 1
    )
    reset = reset | contact_res_reset

    zeros = torch.zeros_like(reward)
    reward = torch.where(root_pos_err >= root_reset_dist, zeros, reward)
    reward = torch.where(root_rot_err >= root_reset_dist, zeros, reward)
    reward = torch.where(contact_res_reset, zeros, reward)

    rew_debug = torch.cat(
        [
            dof_pos_rew.unsqueeze(1),
            dof_vel_rew.unsqueeze(1),
            ef_rew.unsqueeze(1),
            root_pose_rew.unsqueeze(1),
            root_vel_rew.unsqueeze(1),
        ],
        dim=1,
    )

    return reward, reset, rew_debug


@torch.jit.script
def compute_agent_state(
    root_tensor: torch.Tensor,
    dof_state_pos: torch.Tensor,
    feet_contacts: torch.Tensor,
    obs_scales: Dict[str, float],
):
    root_orientation = root_tensor[:, 3:7]
    global_linvel = root_tensor[:, 7:10]
    robot_linvels = quat_rotate_inverse(root_orientation, global_linvel)

    root_rpy = get_euler_xyz(root_orientation)
    # Normalizing angles.
    roll, pitch, yaw = root_rpy

    roll = normalize_angle_tensor(roll)
    pitch = normalize_angle_tensor(pitch)
    yaw = normalize_angle_tensor(yaw)
    robot_euler_xyz = torch.cat(
        [
            roll.unsqueeze(-1),
            pitch.unsqueeze(-1),
            yaw.unsqueeze(-1),
        ],
        dim=1,
    )

    # Normalize angular velocities relative to root orientation.
    root_angvels = root_tensor[:, 10:]
    root_angvels = quat_rotate_inverse(root_orientation, root_angvels)

    # Pose + dof positions.
    # State is Roll, Pitch, deltaRoll, deltaPitch + all joints position + feet contacts.
    curr_state = torch.cat(
        [
            robot_euler_xyz,
            robot_linvels[:, :] * obs_scales["linvel"],
            root_angvels[:, :] * obs_scales["angvel"],
            dof_state_pos,
            feet_contacts,
        ],
        dim=1,
    )

    return curr_state


@torch.jit.script
def compute_future_frames(
    root_tensor: torch.Tensor,
    ref_motion_index: torch.Tensor,
    ref_motion_frames: torch.Tensor,
    target_inc_indexes: torch.Tensor,
    z_basis_vec: torch.Tensor,
):
    # Angle around Z axis
    root_orientation = root_tensor[:, 3:7]
    angle = calc_heading(root_orientation)
    inv_heading_rot = quat_from_angle_axis(angle, z_basis_vec)

    future_inds = ref_motion_index.unsqueeze(1) + target_inc_indexes
    future_target_frames = ref_motion_frames[future_inds, 3:]  # `3:` skips pos info.

    # Normalize orientation.
    future_frames_euler_xy = torch.zeros(
        (root_tensor.shape[0], 4, 2), dtype=torch.float32, device="cuda:0"
    )

    for idx_frame in range(4):
        current_frame_quat = future_target_frames[:, idx_frame, :4]
        current_frame_quat = quat_mul(inv_heading_rot, current_frame_quat)
        current_frame_quat = normalize(current_frame_quat)

        current_frame_euler_xyz = get_euler_xyz(current_frame_quat)
        frame_roll, frame_pitch, _ = current_frame_euler_xyz
        future_frames_euler_xy[:, idx_frame, 0] = normalize_angle_tensor(frame_roll)
        future_frames_euler_xy[:, idx_frame, 1] = normalize_angle_tensor(frame_pitch)

    # Flatten data to insert into observation vector.
    future_target_dofs = future_target_frames[:, :, 4:].reshape(-1, 48)
    future_frames_euler_xy = future_frames_euler_xy.reshape(-1, 2 * 4)

    return torch.cat(
        [
            future_target_dofs,
            future_frames_euler_xy,
        ],
        dim=1,
    )


@torch.jit.script
def compute_observations_mi(
    root_tensor: torch.Tensor,
    dof_state_pos: torch.Tensor,
    actions: torch.Tensor,
    ref_motion_index: torch.Tensor,
    ref_motion_frames: torch.Tensor,
    target_inc_indexes: torch.Tensor,
    z_basis_vec: torch.Tensor,
    feet_contacts: torch.Tensor,
    obs_scales: Dict[str, float],
):
    """Computes the observations tensor, and updates buffers."""

    curr_state = compute_agent_state(
        root_tensor, dof_state_pos, feet_contacts, obs_scales
    )
    future_frames = compute_future_frames(
        root_tensor,
        ref_motion_index,
        ref_motion_frames,
        target_inc_indexes,
        z_basis_vec,
    )

    _obs = torch.cat(
        [
            curr_state,
            actions,
            future_frames,
        ],
        dim=1,
    )

    return _obs
