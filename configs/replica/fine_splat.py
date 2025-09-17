import os
from os.path import join as p_join

scenes = ["room0", "room1", "room2",
          "office0", "office1", "office2",
          "office3", "office4"]

primary_device = "cuda:0"
scene_name = scenes[0]
seed = 5  # 实验序号

map_every = 1
keyframe_every = 5
mapping_window_size = 24
tracking_iters = 40
mapping_iters = 60

group_name = "Replica_Fine_3DGS_5k_a2i"
run_name = f"{scene_name}_{seed}"

config = dict(
    workdir=f"./experiments/{group_name}",
    run_name=run_name,
    seed=seed,
    primary_device=primary_device,
    map_every=map_every, # Mapping every nth frame
    keyframe_every=keyframe_every, # Keyframe every nth frame
    mapping_window_size=mapping_window_size, # Mapping window size
    report_global_progress_every=5, # Report Global Progress every nth frame
    eval_every=1, # Evaluate every nth frame (at end of SLAM)
    scene_radius_depth_ratio=3, # Max First Frame Depth to Scene Radius Ratio (For Pruning/Densification)
    mean_sq_dist_method="projective", # ["projective", "knn"] (Type of Mean Squared Distance Calculation for Scale of Gaussians)
    gaussian_distribution="isotropic", # ["isotropic", "anisotropic"] (Isotropic -> Spherical Covariance, Anisotropic -> Ellipsoidal Covariance)
    report_iter_progress=False,
    load_checkpoint=True,  # 正常gaussian_splatting中根本没用到这个参数，所以这里改成True(原本为False)
    checkpoint_time_idx=0,
    save_checkpoints=False, # Save Checkpoints
    checkpoint_interval=5, # Checkpoint Interval
    use_wandb=True,

    wandb=dict(
        entity="theairlab",
        project="SplaTAM",
        group=group_name,
        name=run_name,
        save_qual=False,
        eval_save_qual=True,
    ),
    data=dict(
        basedir="./data/Replica",
        gradslam_data_cfg="./configs/data/replica.yaml",
        sequence=scene_name,
        desired_image_height_init=170,
        desired_image_width_init=300,
        desired_image_height=340,
        desired_image_width=600,
        start=0,
        end=-1,
        stride=1,
        num_frames=-1,
        eval_stride=1,
        eval_num_frames=2000,
    ),
    train=dict(
        num_iters_mapping=5000,  # 训练迭代次数
        sil_thres=0.5, # For Addition of new Gaussians & Visualization
        use_sil_for_loss=True, # Use Silhouette for Loss during Tracking
        loss_weights=dict(
            im=0.5,
            depth=1.0,
        ),
        lrs_mapping=dict(
            means3D=0.00032,
            rgb_colors=0.0025,
            unnorm_rotations=0.001,
            logit_opacities=0.05,
            log_scales=0.005,
            cam_unnorm_rots=0.0000,
            cam_trans=0.0000,
        ),
        lrs_mapping_means3D_final=0.0000032,
        lr_delay_mult=0.01,
        # added by wincent
        prune_gaussians=True, # Prune Gaussians during Mapping
        # pruning_dict=dict( # Needs to be updated based on the number of mapping iterations
        #     start_after=0,
        #     remove_big_after=0,
        #     stop_after=20,
        #     prune_every=20,
        #     removal_opacity_threshold=0.005,
        #     final_removal_opacity_threshold=0.005,
        #     reset_opacities=False,
        #     reset_opacities_every=500, # Doesn't consider iter 0
        # ),
        pruning_dict=dict(
            start_after=0,
            remove_big_after=0,
            stop_after=1500,  # ✅ 让 prune 贯穿全程
            prune_every=100,  # ✅ 每 100 轮清理一次
            removal_opacity_threshold=0.1,  # ✅ 保守但仍有效
            final_removal_opacity_threshold=0.05,
            reset_opacities=True,  # ✅ 重置低质量点
            reset_opacities_every=500,
        ),

        use_gaussian_splatting_densification=True, # Use Gaussian Splatting-based Densification during Mapping
        # densify_dict=dict( # Needs to be updated based on the number of mapping iterations
        #     start_after=500,
        #     remove_big_after=3000,
        #     stop_after=15000,
        #     densify_every=100,
        #     grad_thresh=0.0002,
        #     num_to_split_into=2,
        #     removal_opacity_threshold=0.005,
        #     final_removal_opacity_threshold=0.005,
        #     reset_opacities=True,
        #     reset_opacities_every=3000, # Doesn't consider iter 0
        # ),
        densify_dict=dict(
            start_after=1500,
            remove_big_after=500,
            stop_after=3000,  # 这个参数及时关闭能够阻止OOM
            densify_every=200,                     # ✅ 降低频率
            grad_thresh=0.0005,                    # ✅ 提高门槛
            num_to_split_into=2,
            removal_opacity_threshold=0.005,
            final_removal_opacity_threshold=0.005,
            reset_opacities=False,
            reset_opacities_every=3000,  # 主要是这个reset_opacities_every会非常影响显存
            max_gaussians=2_000_000,               # ✅ 显存控制 这里暂时没用到
        ),
        # densify_dict=dict(
        #     start_after=0,
        #     remove_big_after=500,
        #     stop_after=1500,  # 这个参数及时关闭能够阻止OOM
        #     densify_every=200,  # ✅ 降低频率
        #     grad_thresh=0.0005,  # ✅ 提高门槛
        #     num_to_split_into=2,
        #     removal_opacity_threshold=0.005,
        #     final_removal_opacity_threshold=0.005,
        #     reset_opacities=False,
        #     reset_opacities_every=3000,  # 主要是这个reset_opacities_every会非常影响显存
        #     max_gaussians=2_000_000,  # ✅ 显存控制 这里暂时没用到
        # ),
    ),
    viz=dict(
        render_mode='color', # ['color', 'depth' or 'centers']
        offset_first_viz_cam=True, # Offsets the view camera back by 0.5 units along the view direction (For Final Recon Viz)
        show_sil=False, # Show Silhouette instead of RGB
        visualize_cams=True, # Visualize Camera Frustums and Trajectory
        viz_w=600, viz_h=340,
        viz_near=0.01, viz_far=100.0,
        view_scale=2,
        viz_fps=5, # FPS for Online Recon Viz
        enter_interactive_post_online=True, # Enter Interactive Mode after Online Recon Viz
    ),
)