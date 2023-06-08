import torch
from kbnet.kbnet_model import KBNetModel


class KBnetPredictor:
    def __init__(self, args=None, depth_model=None, pose_model=None):
        if depth_model is None and args is not None:
            depth_model = KBNetModel(
                input_channels_image=args.input_channels_image,
                input_channels_depth=args.input_channels_depth,
                min_pool_sizes_sparse_to_dense_pool=args.min_pool_sizes_sparse_to_dense_pool,
                max_pool_sizes_sparse_to_dense_pool=args.max_pool_sizes_sparse_to_dense_pool,
                n_convolution_sparse_to_dense_pool=args.n_convolution_sparse_to_dense_pool,
                n_filter_sparse_to_dense_pool=args.n_filter_sparse_to_dense_pool,
                n_filters_encoder_image=args.n_filters_encoder_image,
                n_filters_encoder_depth=args.n_filters_encoder_depth,
                resolutions_backprojection=args.resolutions_backprojection,
                n_filters_decoder=args.n_filters_decoder,
                deconv_type=args.deconv_type,
                weight_initializer=args.weight_initializer,
                activation_func=args.activation_func,
                min_predict_depth=args.min_predict_depth,
                max_predict_depth=args.max_predict_depth,
                device=args.device,
            )
            depth_model.restore_model(args.depth_model_restore_path)
            depth_model.eval()
        if pose_model is None and args is not None:
            pose_model = get_pose_model(args.device)
        self.depth_model = depth_model
        self.pose_model = pose_model

    def predict(self, image, sparse_depth, intrinsics):
        validity_map_depth = torch.where(
            sparse_depth > 0, torch.ones_like(sparse_depth), sparse_depth
        )
        filtered_validity_map_depth0 = validity_map_depth
        output_depth = self.depth_model.forward(
            image=image,
            sparse_depth=sparse_depth,
            validity_map_depth=filtered_validity_map_depth0,
            intrinsics=intrinsics,
        )
        return output_depth


def get_pose_model(device, encoder_type="resnet18"):
    from kbnet.posenet_model import PoseNetModel

    pose_model = PoseNetModel(
        encoder_type=encoder_type,
        rotation_parameterization="axis",
        weight_initializer="xavier_normal",
        activation_func="relu",
        device=device,
    )

    pose_model.train()
    pose_model_restore_path = "/media/master/wext/msc_studies/second_semester/research_project/related_work/calibrated-backprojection-network/pretrained_models/kitti/posenet-kitti.pth"
    pose_model.restore_model(pose_model_restore_path)
    return pose_model
