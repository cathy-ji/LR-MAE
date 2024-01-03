# Copyright (c) Facebook, Inc. and its affiliates.
import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn

from models.MAE import Point_AE, Group, Point_AE_LEM
from third_party.pointnet2.pointnet2_modules import PointnetSAModuleVotes
from third_party.pointnet2.pointnet2_utils import furthest_point_sample
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.pc_util import scale_points, shift_scale_points

from models.helpers import GenericMLP
from models.position_embedding import PositionEmbeddingCoordsSine
from models.transformer import (MaskedTransformerEncoder, TransformerDecoder,
                                TransformerDecoderLayer, TransformerEncoder,
                                TransformerEncoderLayer)


class BoxProcessor(object):
    """
    Class to convert 3DETR MLP head outputs into bounding boxes
    """

    def __init__(self, dataset_config):
        self.dataset_config = dataset_config

    def compute_predicted_center(self, center_offset, query_xyz, point_cloud_dims):
        center_unnormalized = query_xyz + center_offset
        center_normalized = shift_scale_points(
            center_unnormalized, src_range=point_cloud_dims
        )
        return center_normalized, center_unnormalized

    def compute_predicted_size(self, size_normalized, point_cloud_dims):
        scene_scale = point_cloud_dims[1] - point_cloud_dims[0]
        scene_scale = torch.clamp(scene_scale, min=1e-1)
        size_unnormalized = scale_points(size_normalized, mult_factor=scene_scale)
        return size_unnormalized

    def compute_predicted_angle(self, angle_logits, angle_residual):
        if angle_logits.shape[-1] == 1:
            # special case for datasets with no rotation angle
            # we still use the predictions so that model outputs are used
            # in the backwards pass (DDP may complain otherwise)
            angle = angle_logits * 0 + angle_residual * 0
            angle = angle.squeeze(-1).clamp(min=0)
        else:
            angle_per_cls = 2 * np.pi / self.dataset_config.num_angle_bin
            pred_angle_class = angle_logits.argmax(dim=-1).detach()
            angle_center = angle_per_cls * pred_angle_class
            angle = angle_center + angle_residual.gather(
                2, pred_angle_class.unsqueeze(-1)
            ).squeeze(-1)
            mask = angle > np.pi
            angle[mask] = angle[mask] - 2 * np.pi
        return angle

    def compute_objectness_and_cls_prob(self, cls_logits):
        assert cls_logits.shape[-1] == self.dataset_config.num_semcls + 1
        cls_prob = torch.nn.functional.softmax(cls_logits, dim=-1)
        objectness_prob = 1 - cls_prob[..., -1]
        return cls_prob[..., :-1], objectness_prob

    def box_parametrization_to_corners(
        self, box_center_unnorm, box_size_unnorm, box_angle
    ):
        return self.dataset_config.box_parametrization_to_corners(
            box_center_unnorm, box_size_unnorm, box_angle
        )


class Model3DETR(nn.Module):
    """
    Main 3DETR model. Consists of the following learnable sub-models
    - pre_encoder: takes raw point cloud, subsamples it and projects into "D" dimensions
                Input is a Nx3 matrix of N point coordinates
                Output is a N'xD matrix of N' point features
    - encoder: series of self-attention blocks to extract point features
                Input is a N'xD matrix of N' point features
                Output is a N''xD matrix of N'' point features.
                N'' = N' for regular encoder; N'' = N'//2 for masked encoder
    - query computation: samples a set of B coordinates from the N'' points
                and outputs a BxD matrix of query features.
    - decoder: series of self-attention and cross-attention blocks to produce BxD box features
                Takes N''xD features from the encoder and BxD query features.
    - mlp_heads: Predicts bounding box parameters and classes from the BxD box features
    """

    def __init__(
        self,
        pre_encoder,
        encoder,
        decoder,
        mae,
        dataset_config,
        encoder_dim=256,
        decoder_dim=256,
        position_embedding="fourier",
        mlp_dropout=0.3,
        num_queries=256,
    ):
        super().__init__()
        self.pre_encoder = pre_encoder
        self.encoder = encoder
        if hasattr(self.encoder, "masking_radius"):
            hidden_dims = [encoder_dim]
        else:
            hidden_dims = [encoder_dim, encoder_dim]
        self.encoder_to_decoder_projection = GenericMLP(
            input_dim=encoder_dim,
            hidden_dims=hidden_dims,
            output_dim=decoder_dim,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            output_use_activation=True,
            output_use_norm=True,
            output_use_bias=False,
        )
        self.pos_embedding = PositionEmbeddingCoordsSine(
            d_pos=decoder_dim, pos_type=position_embedding, normalize=True
        )
        self.query_projection = GenericMLP(
            input_dim=decoder_dim,
            hidden_dims=[decoder_dim],
            output_dim=decoder_dim,
            use_conv=True,
            output_use_activation=True,
            hidden_use_bias=True,
        )
        self.num_blocks = 16
        self.block_size = 2048
        self.mae = mae
        self.block2query = nn.Conv1d(384, 256, 1)
        self.group_divider = Group(num_group=self.num_blocks, group_size=self.block_size)
        self.decoder = decoder
        self.build_mlp_heads(dataset_config, decoder_dim, mlp_dropout)

        self.num_queries = num_queries
        self.box_processor = BoxProcessor(dataset_config)

    def load_model_from_ckpt(self, ckpt_path):
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path)
            base_ckpt = ckpt['model']

            incompatible = self.load_state_dict(base_ckpt, strict=False)
            if incompatible.missing_keys:
                print('missing_keys')
                print(
                    get_missing_parameters_message(incompatible.missing_keys)
                )
            if incompatible.unexpected_keys:
                print('unexpected_keys')
                print(
                    get_unexpected_parameters_message(incompatible.unexpected_keys)
                )

            print(f'[Transformer] Successful Loading the ckpt from {ckpt_path}')

    def build_mlp_heads(self, dataset_config, decoder_dim, mlp_dropout):
        mlp_func = partial(
            GenericMLP,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            hidden_dims=[decoder_dim, decoder_dim],
            dropout=mlp_dropout,
            input_dim=decoder_dim,
        )

        # Semantic class of the box
        # add 1 for background/not-an-object class
        semcls_head = mlp_func(output_dim=dataset_config.num_semcls + 1)

        # geometry of the box
        center_head = mlp_func(output_dim=3)
        size_head = mlp_func(output_dim=3)
        angle_cls_head = mlp_func(output_dim=dataset_config.num_angle_bin)
        angle_reg_head = mlp_func(output_dim=dataset_config.num_angle_bin)

        mlp_heads = [
            ("sem_cls_head", semcls_head),
            ("center_head", center_head),
            ("size_head", size_head),
            ("angle_cls_head", angle_cls_head),
            ("angle_residual_head", angle_reg_head),
        ]
        self.mlp_heads = nn.ModuleDict(mlp_heads)

    def get_query_embeddings(self, encoder_xyz, point_cloud_dims):
        query_inds = furthest_point_sample(encoder_xyz, self.num_queries)       
        query_inds = query_inds.long()
        query_xyz = [torch.gather(encoder_xyz[..., x], 1, query_inds) for x in range(3)]
        query_xyz = torch.stack(query_xyz)  
        query_xyz = query_xyz.permute(1, 2, 0)      

        # Gater op above can be replaced by the three lines below from the pointnet2 codebase
        # xyz_flipped = encoder_xyz.transpose(1, 2).contiguous()
        # query_xyz = gather_operation(xyz_flipped, query_inds.int())
        # query_xyz = query_xyz.transpose(1, 2)
        pos_embed = self.pos_embedding(query_xyz, input_range=point_cloud_dims)    
        query_embed = self.query_projection(pos_embed)                            
        return query_xyz, query_embed

    def _break_up_pc(self, pc):
        # pc may contain color/normals.

        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def run_encoder(self, point_clouds):
        xyz, features = self._break_up_pc(point_clouds)         
        pre_enc_xyz, pre_enc_features, pre_enc_inds = self.pre_encoder(xyz, features)       
        # xyz: batch x npoints x 3
        # features: batch x channel x npoints
        # inds: batch x npoints

        # nn.MultiHeadAttention in encoder expects npoints x batch x channel features
        pre_enc_features = pre_enc_features.permute(2, 0, 1)

        # xyz points are in batch x npointx channel order
        enc_xyz, enc_features, enc_inds = self.encoder(         
            pre_enc_features, xyz=pre_enc_xyz
        )       
        if enc_inds is None:
            # encoder does not perform any downsampling
            enc_inds = pre_enc_inds        
        else:
            # use gather here to ensure that it works for both FPS and random sampling
            enc_inds = torch.gather(pre_enc_inds, 1, enc_inds.type(torch.int64))
        return enc_xyz, enc_features, enc_inds     

    def get_box_predictions(self, query_xyz, point_cloud_dims, box_features):
        """
        Parameters:
            query_xyz: batch x nqueries x 3 tensor of query XYZ coords
            point_cloud_dims: List of [min, max] dims of point cloud
                              min: batch x 3 tensor of min XYZ coords
                              max: batch x 3 tensor of max XYZ coords
            box_features: num_layers x num_queries x batch x channel
        """
        # box_features change to (num_layers x batch) x channel x num_queries
        box_features = box_features.permute(0, 2, 3, 1)         # torch.Size([8, 8, 256, 256])      num_layers, batch, channel, num_queries
        num_layers, batch, channel, num_queries = (
            box_features.shape[0],
            box_features.shape[1],
            box_features.shape[2],
            box_features.shape[3],
        )
        box_features = box_features.reshape(num_layers * batch, channel, num_queries)

        # mlp head outputs are (num_layers x batch) x noutput x nqueries, so transpose last two dims
        cls_logits = self.mlp_heads["sem_cls_head"](box_features).transpose(1, 2)                       # torch.Size([64, 256, 19])
        center_offset = (
            self.mlp_heads["center_head"](box_features).sigmoid().transpose(1, 2) - 0.5
        )
        size_normalized = (                                                                             # torch.Size([64, 256, 3]) 介于[-0.5~0.5]之间
            self.mlp_heads["size_head"](box_features).sigmoid().transpose(1, 2)
        )
        angle_logits = self.mlp_heads["angle_cls_head"](box_features).transpose(1, 2)                   # torch.Size([64, 256, 1])
        angle_residual_normalized = self.mlp_heads["angle_residual_head"](
            box_features
        ).transpose(1, 2)                                                                               # torch.Size([64, 256, 1])

        # reshape outputs to num_layers x batch x nqueries x noutput
        cls_logits = cls_logits.reshape(num_layers, batch, num_queries, -1)                             # torch.Size([8, 8, 256, 19])
        center_offset = center_offset.reshape(num_layers, batch, num_queries, -1)                       # torch.Size([8, 8, 256, 3])
        size_normalized = size_normalized.reshape(num_layers, batch, num_queries, -1)                   # torch.Size([8, 8, 256, 3])
        angle_logits = angle_logits.reshape(num_layers, batch, num_queries, -1)                         # torch.Size([8, 8, 256, 1])
        angle_residual_normalized = angle_residual_normalized.reshape(                                  # torch.Size([8, 8, 256, 1])
            num_layers, batch, num_queries, -1
        )
        angle_residual = angle_residual_normalized * (
            np.pi / angle_residual_normalized.shape[-1]
        )

        outputs = []
        for l in range(num_layers):
            # box processor converts outputs so we can get a 3D bounding box
            (
                center_normalized,
                center_unnormalized,
            ) = self.box_processor.compute_predicted_center(
                center_offset[l], query_xyz, point_cloud_dims       #
            )
            angle_continuous = self.box_processor.compute_predicted_angle(
                angle_logits[l], angle_residual[l]                  # 
            )
            size_unnormalized = self.box_processor.compute_predicted_size(
                size_normalized[l], point_cloud_dims                # torch.Size([8, 256, 3])
            )
            box_corners = self.box_processor.box_parametrization_to_corners(
                center_unnormalized, size_unnormalized, angle_continuous       
            )

            # below are not used in computing loss (only for matching/mAP eval)
            # we compute them with no_grad() so that distributed training does not complain about unused variables
            with torch.no_grad():
                (
                    semcls_prob,
                    objectness_prob,
                ) = self.box_processor.compute_objectness_and_cls_prob(cls_logits[l])

            box_prediction = {
                "sem_cls_logits": cls_logits[l],
                "center_normalized": center_normalized.contiguous(),
                "center_unnormalized": center_unnormalized,
                "size_normalized": size_normalized[l],
                "size_unnormalized": size_unnormalized,
                "angle_logits": angle_logits[l],
                "angle_residual": angle_residual[l],
                "angle_residual_normalized": angle_residual_normalized[l],
                "angle_continuous": angle_continuous,
                "objectness_prob": objectness_prob,
                "sem_cls_prob": semcls_prob,
                "box_corners": box_corners,
            }
            outputs.append(box_prediction)

        # intermediate decoder layer outputs are only used during training
        aux_outputs = outputs[:-1]
        outputs = outputs[-1]

        return {
            "outputs": outputs,  # output from last layer of decoder
            "aux_outputs": aux_outputs,  # output from intermediate layers of decoder
        }

    def forward(self, inputs, encoder_only=False):
        point_clouds = inputs["point_clouds"]                           # torch.Size([8, 40000, 3])

        neighborhood, center = self.group_divider(point_clouds)
        pts = neighborhood
        bs, num_block, n, _ = pts.shape
        pts = pts.reshape(bs * num_block, n, _)
        _, blocks_feat = self.mae(pts)
        blocks_feat = blocks_feat.reshape(bs, num_block, -1, blocks_feat.size(-1)).mean(2)
        blocks_feat = self.block2query(blocks_feat.permute(0, 2, 1)).permute(0, 2, 1).repeat_interleave(self.num_queries // num_block, dim=1).permute(1, 0, 2).contiguous()

        enc_xyz, enc_features, enc_inds = self.run_encoder(point_clouds)        # torch.Size([4, 2048, 3])      torch.Size([2048, 4, 256])
        enc_features = self.encoder_to_decoder_projection(
            enc_features.permute(1, 2, 0)
        ).permute(2, 0, 1)      
        # encoder features: npoints x batch x channel
        # encoder xyz: npoints x batch x 3

        if encoder_only:
            # return: batch x npoints x channels
            return enc_xyz, enc_features.transpose(0, 1)

        point_cloud_dims = [
            inputs["point_cloud_dims_min"],
            inputs["point_cloud_dims_max"],
        ]       # torch.Size([8, 3]), torch.Size([8, 3])
        query_xyz, query_embed = self.get_query_embeddings(enc_xyz, point_cloud_dims)     
        # query_embed: batch x channel x npoint
        enc_pos = self.pos_embedding(enc_xyz, input_range=point_cloud_dims)               

        # decoder expects: npoints x batch x channel
        enc_pos = enc_pos.permute(2, 0, 1)                  # torch.Size([2048, 8, 256])    
        query_embed = query_embed.permute(2, 0, 1)          # torch.Size([256, 8, 256])    
        tgt = torch.zeros_like(query_embed) + blocks_feat               # torch.Size([256, 8, 256])     
        box_features = self.decoder(
            tgt, enc_features, query_pos=query_embed, pos=enc_pos
        )[0]        # box_features  torch.Size([8, 256, 8, 256]) 

        box_predictions = self.get_box_predictions(
            query_xyz, point_cloud_dims, box_features
        )
        return box_predictions


def build_preencoder(args):
    mlp_dims = [3 * int(args.use_color), 64, 128, args.enc_dim]
    preencoder = PointnetSAModuleVotes(
        radius=0.2,
        nsample=64,
        npoint=args.preenc_npoints,
        mlp=mlp_dims,
        normalize_xyz=True,
    )
    return preencoder


def build_encoder(args):
    if args.enc_type == "vanilla":
        encoder_layer = TransformerEncoderLayer(
            d_model=args.enc_dim,
            nhead=args.enc_nhead,
            dim_feedforward=args.enc_ffn_dim,
            dropout=args.enc_dropout,
            activation=args.enc_activation,
        )
        encoder = TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=args.enc_nlayers
        )
    elif args.enc_type in ["masked"]:
        encoder_layer = TransformerEncoderLayer(
            d_model=args.enc_dim,
            nhead=args.enc_nhead,
            dim_feedforward=args.enc_ffn_dim,
            dropout=args.enc_dropout,
            activation=args.enc_activation,
        )
        interim_downsampling = PointnetSAModuleVotes(
            radius=0.4,
            nsample=32,
            npoint=args.preenc_npoints // 2,
            mlp=[args.enc_dim, 256, 256, args.enc_dim],
            normalize_xyz=True,
        )
        
        masking_radius = [math.pow(x, 2) for x in [0.4, 0.8, 1.2]]
        encoder = MaskedTransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=3,
            interim_downsampling=interim_downsampling,
            masking_radius=masking_radius,
        )
    else:
        raise ValueError(f"Unknown encoder type {args.enc_type}")
    return encoder


def build_decoder(args):
    decoder_layer = TransformerDecoderLayer(
        d_model=args.dec_dim,
        nhead=args.dec_nhead,
        dim_feedforward=args.dec_ffn_dim,
        dropout=args.dec_dropout,
    )
    decoder = TransformerDecoder(
        decoder_layer, num_layers=args.dec_nlayers, return_intermediate=True
    )
    return decoder


def build_umae_3detr(args, dataset_config):
    pre_encoder = build_preencoder(args)
    encoder = build_encoder(args)
    decoder = build_decoder(args)
    mae = Point_AE_LEM()
    model = Model3DETR(
        pre_encoder,
        encoder,
        decoder,
        mae,
        dataset_config,
        encoder_dim=args.enc_dim,
        decoder_dim=args.dec_dim,
        mlp_dropout=args.mlp_dropout,
        num_queries=args.nqueries,
    )
    output_processor = BoxProcessor(dataset_config)
    return model, output_processor
