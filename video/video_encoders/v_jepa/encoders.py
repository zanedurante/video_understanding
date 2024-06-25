import pathlib
if __name__ == '__main__':
    import sys
    sys.path.append(str(pathlib.Path(__file__).resolve().parent / 'jepa'))

import torch

from src.models.vision_transformer import vit_base,vit_large


class VJEPABase(torch.nn.Module):
    # V-JEPA base model was pretrained separately. The base model uses 
    # the params hardcoded here.
    def __init__(self,
                 pretrained_path:pathlib.Path=None,
                 frozen:bool=False,
                 lora:bool=False,
                 device:str='cuda',
                 num_frames:int=16) -> None:
        super().__init__()
        # super().__init__(pretrained_path,
        #                  frozen,
        #                  lora,
        #                  device,
        #                  num_frames)
        self.ckpt_f = pretrained_path
        # height,weight,channels,frames
        self.H,self.W,self.C,self.T = 224,224,3,16
        # patch-(height,weight,frames)
        self.PH,self.PW,self.PT = 16,16,2
        # num-patches-(height,weight,frames)
        self.NH,self.NW,self.NT = (int(self.H/self.PH),
                                   int(self.W/self.PW),
                                   int(self.T/self.PT)) 
        self.N = self.NH*self.NW*self.NT # num-patches-total
        self.E = 768
        msg = f'VJEPABase was pretrained with num_frames={self.T}'
        assert num_frames == self.T,msg
        # Whether to use height/width/depth uniformly when creating positional
        # embedding.
        uniform_power = True
        self.encoder = vit_base(img_size=self.H,
                                patch_size=self.PH,
                                num_frames=self.T,
                                tubelet_size=self.PT,
                                uniform_power=uniform_power)
        
    def _load_pretrained(self):
        if self.ckpt_f is None:
            return
        print(f'loading pretrained weights from {self.ckpt_f}')
        ckpt = torch.load(self.ckpt_f, map_location='cpu')
        ckpt = ckpt['target_encoder']
        ckpt = {k.replace('module.', ''):v 
                for k, v in ckpt.items()}
        ckpt = {k.replace('backbone.', ''):v 
                for k, v in ckpt.items()}
        self.encoder.load_state_dict(ckpt,strict=True)
        
    def forward(self, video_batch:torch.Tensor):
        C,T,H,W = self.C,self.T,self.H,self.W
        msg = (f'video_batch must be of shape [B,{T},{C},{H},{W}] or '
               f'[B,{C},{T},{H},{W}]')
        assert (video_batch.ndim == 5 and 
                (video_batch.shape[1:] == (T,C,H,W) or
                 video_batch.shape[1:] == (C,T,H,W))),msg
        # V-JEPA expects [B,C,T,H,W]. Transpose here if needed.
        if video_batch.shape[1:] == (T,C,H,W):
            video_batch = video_batch.transpose(1,2) # [B,C,T,H,W]
        B = video_batch.shape[0]
        NH,NW,NT,N,E = self.NH,self.NW,self.NT,self.N,self.E
        emb = self.encoder(video_batch) # [B,N,E]
        emb = emb.reshape([B,NT,NH*NW,E]) # [B,NT,NH*NW,E]
        return emb
    
    def get_spatio_temporal_embeds(self, video_batch:torch.Tensor, **kwargs):
        return self.forward(video_batch)
    
    def get_spatio_temporal_embed_dims(self, **kwargs):
        NH,NW,NT,E = self.NH,self.NW,self.NT,self.E
        return (NT,NH*NW,E)

    def convert_spatio_temporal_embeds_to_video(self, spatio_temporal_embeds):
        # Just returning the input. This should return a tensor of shape [B,D]
        # where D is the dim of the video level embeds.
        # TODO: Investigate a better way to do this

        # average across temporal dim (currently h, w, t, e)
        spatio_temporal_embeds = spatio_temporal_embeds.mean(dim=1)
        # flatten 
        spatio_temporal_embeds = spatio_temporal_embeds.flatten(start_dim=1)
        return spatio_temporal_embeds

    def get_video_level_embeds(self, video_batch):
        st_embeds = self.get_spatio_temporal_embeds(video_batch)
        video_embeds = self.convert_spatio_temporal_embeds_to_video(st_embeds)
        return video_embeds
    
    def get_video_level_embed_dim(self):
        return self.NH * self.NW * self.E


class VJEPALarge(torch.nn.Module):
    # V-JEPA base model was pretrained separately. The base model uses 
    # the params hardcoded here.
    def __init__(self,
                 pretrained_path:pathlib.Path=None,
                 frozen:bool=False,
                 lora:bool=False,
                 device:str='cuda',
                 num_frames:int=16) -> None:
        super().__init__()
        # super().__init__(pretrained_path,
        #                  frozen,
        #                  lora,
        #                  device,
        #                  num_frames)
        self.ckpt_f = pretrained_path
        # height,weight,channels,frames
        self.H,self.W,self.C,self.T = 224,224,3,16
        # patch-(height,weight,frames)
        self.PH,self.PW,self.PT = 16,16,2
        # num-patches-(height,weight,frames)
        self.NH,self.NW,self.NT = (int(self.H/self.PH),
                                   int(self.W/self.PW),
                                   int(self.T/self.PT)) 
        self.N = self.NH*self.NW*self.NT # num-patches-total
        self.E = 1024
        msg = f'VJEPALarge was pretrained with num_frames={self.T}'
        assert num_frames == self.T,msg
        # Whether to use height/width/depth uniformly when creating positional
        # embedding.
        uniform_power = True
        self.encoder = vit_large(img_size=self.H,
                                 patch_size=self.PH,
                                 num_frames=self.T,
                                 tubelet_size=self.PT,
                                 uniform_power=uniform_power)
        
    def _load_pretrained(self):
        if self.ckpt_f is None:
            return
        print(f'loading pretrained weights from {self.ckpt_f}')
        ckpt = torch.load(self.ckpt_f, map_location='cpu')
        ckpt = ckpt['target_encoder']
        ckpt = {k.replace('module.', ''):v 
                for k, v in ckpt.items()}
        ckpt = {k.replace('backbone.', ''):v 
                for k, v in ckpt.items()}
        self.encoder.load_state_dict(ckpt,strict=True)
        
    def forward(self, video_batch:torch.Tensor):
        C,T,H,W = self.C,self.T,self.H,self.W
        msg = (f'video_batch must be of shape [B,{T},{C},{H},{W}] or '
               f'[B,{C},{T},{H},{W}]')
        assert (video_batch.ndim == 5 and 
                (video_batch.shape[1:] == (T,C,H,W) or
                 video_batch.shape[1:] == (C,T,H,W))),msg
        # V-JEPA expects [B,C,T,H,W]. Transpose here if needed.
        if video_batch.shape[1:] == (T,C,H,W):
            video_batch = video_batch.transpose(1,2) # [B,C,T,H,W]
        B = video_batch.shape[0]
        NH,NW,NT,N,E = self.NH,self.NW,self.NT,self.N,self.E
        emb = self.encoder(video_batch) # [B,N,E]
        emb = emb.reshape([B,NT,NH*NW,E]) # [B,NT,NH*NW,E]
        return emb
    
    def get_spatio_temporal_embeds(self, video_batch:torch.Tensor, **kwargs):
        return self.forward(video_batch)
    
    def get_spatio_temporal_embed_dims(self, **kwargs):
        NH,NW,NT,E = self.NH,self.NW,self.NT,self.E
        return (NT,NH*NW,E)

    def convert_spatio_temporal_embeds_to_video(self, spatio_temporal_embeds):
        # Just returning the input. This should return a tensor of shape [B,D]
        # where D is the dim of the video level embeds.
        # TODO: Investigate a better way to do this

        # average across temporal dim (currently h, w, t, e)
        spatio_temporal_embeds = spatio_temporal_embeds.mean(dim=1)
        # flatten 
        spatio_temporal_embeds = spatio_temporal_embeds.flatten(start_dim=1)
        return spatio_temporal_embeds

    def get_video_level_embeds(self, video_batch):
        st_embeds = self.get_spatio_temporal_embeds(video_batch)
        video_embeds = self.convert_spatio_temporal_embeds_to_video(st_embeds)
        return video_embeds
    
    def get_video_level_embed_dim(self):
        return self.NH * self.NW * self.E


if __name__ == '__main__':
    import argparse
    from functools import partial
    from tqdm import tqdm
    from src.datasets.video_dataset import VideoDataset
    from app.vjepa.transforms import make_transforms

    test_model = 'base'
    if test_model == 'base':
        ckpt_f = pathlib.Path('/data/output/jepa/jepa_b16/jepa-latest.pth.tar')
    else:
        ckpt_f = pathlib.Path('/data/output/jepa/jepa_l16/jepa-latest.pth.tar')
    transform = make_transforms(random_horizontal_flip=False,
                                random_resize_aspect_ratio=(0.75,1.35),
                                random_resize_scale=(0.3,1.0),
                                reprob=0.0,
                                auto_augment=False,
                                motion_shift=False,
                                crop_size=224)
    dataset = VideoDataset(data_paths=['/data/jepa_data.csv'],
                           frames_per_clip=16,
                           frame_step=2,
                           num_clips=1,
                           transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=2,
                                             num_workers=0)
    device = torch.device('cuda')
    if test_model == 'base':
        encoder = VJEPABase(pretrained_path=ckpt_f)
    else:
        encoder = VJEPALarge(pretrained_path=ckpt_f)
    encoder._load_pretrained()
    encoder.to(device)
    for i,batch in enumerate(tqdm(dataloader,'batch')):
        batch = batch[0][0].to(device)
        embs = encoder(batch)
        if i >= 100:
            break
