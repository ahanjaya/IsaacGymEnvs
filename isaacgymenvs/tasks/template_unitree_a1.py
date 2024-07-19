import time

import cv2
import numpy as np
import torch
import torchvision
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *
from isaacgymenvs.utils.torch_jit_utils import *
from isaacgymenvs.utils.utils import sample_frictions
from torchvision.utils import make_grid

from .base.vec_task import VecTask


class TemplateUnitreeA1(VecTask):

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
        self._load_cfg_params()

        self.cfg["env"]["numObservations"] = self.p_observation_dim
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
        self._setup_utility_tensors()

        if not headless:
            self._init_debug()
        if self.capture_depth_cam:
            self._init_depth_cams()


    def _init_debug(self):
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

    def _init_depth_cams(self):
        if self.viewer and self.show_depth_cam:
            cv2.namedWindow('Depth Map', cv2.WINDOW_GUI_NORMAL)

        camera_properties = gymapi.CameraProperties()
        camera_properties.horizontal_fov = self.p_depth_cam_horizontal_fov
        camera_properties.width = self.p_depth_cam_width
        camera_properties.height = self.p_depth_cam_height
        camera_properties.enable_tensors = True

        # create depth cameras attached to the base of robots and store depth map tensors
        self.depth_map_tensors = [None] * self.num_envs
        self.cam_handles = []
        for i in range(self.num_envs):
            depth_cam_handle = self.gym.create_camera_sensor(self.envs[i], camera_properties)
            self.cam_handles.append(depth_cam_handle)
            camera_offset = gymapi.Vec3(self.p_depth_cam_offset[0],
                                        self.p_depth_cam_offset[1],
                                        self.p_depth_cam_offset[2])
            camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.deg2rad(0))
            camera_transform = gymapi.Transform(camera_offset, camera_rotation)
            actor_handle = self.gym.get_actor_handle(self.envs[i], 0)
            body_handle = self.gym.get_actor_rigid_body_handle(self.envs[i], actor_handle, self.p_depth_cam_rigid_body_id)

            self.gym.attach_camera_to_body(depth_cam_handle, self.envs[i], body_handle, camera_transform, gymapi.FOLLOW_TRANSFORM)

            cam_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[i], depth_cam_handle, gymapi.IMAGE_DEPTH)
            torch_cam_tensor = gymtorch.wrap_tensor(cam_tensor)
            self.depth_map_tensors[i] = torch_cam_tensor

        self.depth_resize_transform = torchvision.transforms.Resize(
            (self.p_depth_cam_resize_to[1], self.p_depth_cam_resize_to[0]),
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC
        )

        # store depth maps of all envs
        self.depth_maps = torch.zeros(
            self.num_envs,
            self.p_depth_cam_resize_to[1],  # camera_properties.height,
            self.p_depth_cam_resize_to[0],  # camera_properties.width,
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
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
        self.p_kp = self.cfg["env"]["kp"]
        self.p_kd = self.cfg["env"]["kd"]
        self.dt = self.cfg["sim"]["dt"]
        self.sim_rate = 1 / self.dt
        self.p_num_sticky_actions = self.cfg["env"]["controlFrequencyInv"]
        self.p_observation_dim = self.cfg["env"]["observation_dim"]
        self.p_action_dim = self.cfg["env"]["action_dim"]
        self.p_action_scale = self.cfg["env"]["action_scale"]

        self.p_rand_static_friction = self.cfg["env"]["rand_static_friction"]
        self.p_rand_restitution = self.cfg["env"]["rand_restitution"]

        # depth cam
        depth_cam_cfg = self.cfg["env"]["depth_cam"]
        self.capture_depth_cam = depth_cam_cfg["capture"]
        self.show_depth_cam = depth_cam_cfg["show"]
        self.p_depth_cam_view = depth_cam_cfg["view"]
        self.p_depth_cam_width = depth_cam_cfg["width"]
        self.p_depth_cam_height = depth_cam_cfg["height"]
        self.p_depth_cam_horizontal_fov = depth_cam_cfg["horizontal_fov"]
        self.p_depth_cam_clip_distance = depth_cam_cfg["clip_distance"]
        self.p_depth_cam_rigid_body_id = depth_cam_cfg["rigid_body_id"]
        self.p_depth_cam_resize_to = depth_cam_cfg["resize_to"]
        self.p_depth_cam_offset = depth_cam_cfg["camera_offset"]
        self.p_depth_cam_update_interval = depth_cam_cfg["update_interval"]

    def _setup_utility_tensors(self):
        """Creates tensors used to read and modify the actors' states."""
        self.max_episode_length = 100
        self.common_step_counter = 0

        # Robot actions are offset from these default PD targets.
        self.default_joints_pose = torch.tensor(
            [0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8],
            dtype=torch.float32,
            device=self.device,
        )
        self.default_root_pos = torch.tensor(
            [0.0, 0.0, 0.3005], device=self.device, dtype=torch.float32
        )
        self.default_root_ori = torch.tensor(
            [0.0, 0.0, 0.0, 1.0], device=self.device, dtype=torch.float32
        )
        # tensor([-2.4701e-03,  6.0407e-02,  1.4949e-04,  9.9817e-01], device='cuda:0')

        # Root state Tensor.
        self._root_tensor = self.gym.acquire_actor_root_state_tensor(
            self.sim
        )  # Num Actors x 13
        self.root_tensor = gymtorch.wrap_tensor(self._root_tensor)
        self.root_tensor_sim = self.root_tensor

        # Useful views of the root tensor
        self.root_positions = self.root_tensor[:, 0:3]
        self.root_orientations = self.root_tensor[:, 3:7]
        self.root_linear_vel = self.root_tensor[:, 7:10]
        self.root_angular_vel = self.root_tensor[:, 10:13]

        self.root_sim_positions = self.root_tensor_sim[:, 0:3]
        self.root_sim_orientations = self.root_tensor_sim[:, 3:7]
        self.root_sim_linear_vel = self.root_tensor_sim[:, 7:10]
        self.root_sim_angular_vel = self.root_tensor_sim[:, 10:13]

        self._rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_tensor = gymtorch.wrap_tensor(self._rigid_body_tensor)
        self.rb_sim_tensor = self.rb_tensor.view(self.num_envs, -1, 13)
        self.rb_sim_pos_tensor = self.rb_sim_tensor[:, :, :3]
        self.rb_sim_quat_tensor = self.rb_sim_tensor[:, :, 3:7]

        # Relevant indexes of rigid bodies.
        self.rb_feet_indexes = to_torch(
            [4, 8, 12, 16], dtype=torch.long, device=self.device
        )

        self._state_dof = self.gym.acquire_dof_state_tensor(self.sim)
        self.state_dof_tensor = gymtorch.wrap_tensor(self._state_dof)
        self.dof_pos_sim_actors = self.state_dof_tensor.view(self.num_envs, -1, 2)[:, :, 0]
        self.dof_vel_sim_actors = self.state_dof_tensor.view(self.num_envs, -1, 2)[:, :, 1]

        # Contact force tensors.
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(
            self.num_envs, -1, 3
        )
        self.contact_forces_sim = self.contact_forces[:, :, :]

        self.actions = torch.zeros(
            (self.num_envs, 12), dtype=torch.float32, device=self.device
        )

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.dt = self.dt
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81

        # Note: In headless mode, for camera the graphics device id should be
        # the same as the device id.
        if self.headless and self.capture_depth_cam:
            self.graphics_device_id = self.device_id

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

        # unitree a1 asset
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

        # set unitree a1 dof properties
        self.num_a1_dofs = self.gym.get_asset_dof_count(sim_asset)
        dof_properties = self.gym.get_asset_dof_properties(sim_asset)
        self.num_a1_bodies = self.gym.get_asset_rigid_body_count(sim_asset)
        self.dof_lower_limits = []
        self.dof_upper_limits = []

        for i in range(self.num_a1_dofs):
            self.dof_lower_limits.append(dof_properties["lower"][i])
            self.dof_upper_limits.append(dof_properties["upper"][i])

        self.dof_lower_limits = to_torch(self.dof_lower_limits, device=self.device)
        self.dof_upper_limits = to_torch(self.dof_upper_limits, device=self.device)
        print(f"Num A1 bodies: {self.num_a1_bodies}")
        print(f"Num A1 dofs: {self.num_a1_dofs}")

        ##############################################
        self.envs = []

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

        for i in range(num_envs):
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)

            # Create the learner actor
            sim_actor = self.gym.create_actor(
                env, sim_asset, start_pose, "robot", i, 1
            )
            props = self.gym.get_actor_dof_properties(env, sim_actor)
            props["driveMode"][:] = gymapi.DOF_MODE_POS
            props["stiffness"][:] = self.p_kp
            props["damping"][:] = self.p_kd
            self.gym.set_actor_dof_properties(env, sim_actor, props)

            self.envs.append(env)

            orig_pos = self.gym.get_env_origin(env)
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
        if self.show_depth_cam:
            self._show_depth_maps()

    def update_debug_camera(self):
        actor_pos = self.root_sim_positions.cpu().numpy()[self.i_follow_env]

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

    def _update_depth_maps(self):
        # Hack: work around to capture depth maps in headless mode
        if self.headless:
            self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)

        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        # post-process and normalize depth maps
        for i in range(self.num_envs):
            depth_map = self.depth_map_tensors[i]  # type: ignore
            clip_distance = self.p_depth_cam_clip_distance

            # -inf implies no depth value, set it to zero. output will be black.
            depth_map[depth_map == -torch.inf] = -clip_distance
            depth_map[depth_map < -clip_distance] = -clip_distance

            # flip the direction so near-objects are light and far objects are dark
            # normalize from [clip, 0] to [-255, 0]
            depth_map = -255.0 * (depth_map / torch.min(depth_map + 1e-4))
            self.depth_maps[i, :, :] = self.depth_resize_transform(depth_map.unsqueeze(0))

        self.gym.end_access_image_tensors(self.sim)

    def _show_depth_maps(self):
        if self.p_depth_cam_view == 'single':
            depth_map_npy = self.depth_maps[self.i_follow_env, :, :].detach().cpu().numpy().astype(np.uint8)
        elif self.p_depth_cam_view == 'all':
            grid_depth_img = make_grid(self.depth_maps.unsqueeze(1), nrow=round(self.depth_maps.shape[0] ** 0.5))
            depth_map_npy = grid_depth_img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

        cv2.imshow("Depth Map", depth_map_npy)
        # cv2.imshow("Depth Map", cv2.resize(
        #     depth_map_npy, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
        # )
        cv2.waitKey(1)

    def compute_observations(self):
        # Update state of the tensors
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.obs_buf[:] = torch.zeros_like(self.obs_buf)

    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:] = compute_reward(
            self.progress_buf,
            self.reset_buf,
            self.max_episode_length,
        )

    def pre_physics_step(self, actions):
        # Store current step actions. used in Motion Imitation style observations.
        actions = torch.zeros_like(actions)
        self.actions = self.default_joints_pose + (
            actions * self.p_action_scale
        )

        # Clamp values
        self.actions = tensor_clamp(
            self.actions, self.dof_lower_limits, self.dof_upper_limits
        )

        # Set PD targets for ALL actors.
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self.actions)
        )

    def reset_idx(self, env_ids):
        #### Reference State Initialization ####
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.actions[env_ids] = 0.0

        env_ids_int32 = env_ids.to(torch.int32)
        # DoF state initialization.
        self.dof_pos_sim_actors[env_ids_int32] = self.default_joints_pose
        self.dof_vel_sim_actors[env_ids_int32] = 0.0
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.state_dof_tensor),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

        # Root state initialization.
        self.root_sim_positions[env_ids] = self.default_root_pos
        self.root_sim_orientations[env_ids] = self.default_root_ori
        self.root_sim_linear_vel[env_ids] = 0.0
        self.root_sim_angular_vel[env_ids] = 0.0
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_tensor),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def post_physics_step(self):
        self.progress_buf += 1
        self.common_step_counter += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        if self.capture_depth_cam and (
             (self.common_step_counter % self.p_depth_cam_update_interval) == 0
        ):
            self._update_depth_maps()

        self.compute_observations()
        self.compute_reward()

@torch.jit.script
def compute_reward(
    progress_buf: torch.Tensor,
    reset_buf: torch.Tensor,
    max_episode_length: int,
):
    ##### Compute resulting reward. ######
    reward = torch.zeros_like(progress_buf)

    # ##### Reset logic. #####
    reset = torch.where(
        progress_buf >= max_episode_length - 1, 1, reset_buf
    )

    return reward, reset
