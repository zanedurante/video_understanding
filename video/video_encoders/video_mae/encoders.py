import pathlib

import torch

from models.modeling_pretrain import PretrainVisionTransformerEncoder

class VideoMAEv2Base(torch.nn.Module):
    # VideoMAEv2 base model was pretrained separately. The base model uses 
    # the params hardcoded here.
    def __init__(self,
                 pretrained_path:pathlib.Path=None,
                 frozen:bool=False,
                 lora:bool=False,
                 device:str='cude',
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
        msg = f'VideoMAEv2Base was pretrained with num_frames={self.T}'
        assert num_frames == self.T,msg
        self.encoder = PretrainVisionTransformerEncoder(
            img_size=self.H, 
            patch_size=self.PH,
            in_chans=self.C,
            num_classes=0,
            embed_dim=self.E,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            norm_layer=torch.nn.LayerNorm,
            init_values=0.0,
            tubelet_size=self.PT,
            use_learnable_pos_emb=False,
            with_cp=True,
            all_frames=self.T,
            cos_attn=False)
        
    def _load_pretrained(self):
        if self.ckpt_f is None:
            return
        print(f'loading pretrained weights from {self.ckpt_f}')
        ckpt = torch.load(self.ckpt_f, map_location='cpu')
        errs = self.load_state_dict(ckpt['model'],strict=False)
        # There will be more keys in the checkpoint file than needed. Make sure
        # there are no missing keys. Having unexpected keys is okay.
        assert not errs.missing_keys
        
    def forward(self, video_batch):
        C,T,H,W = self.C,self.T,self.H,self.W
        msg = f'video_batch must be of shape [B,{C},{T},{H},{W}]'
        assert (video_batch.ndim == 5 and 
                video_batch.shape[1:] == (C,T,H,W)),msg
        B = video_batch.shape[0]
        NH,NW,NT,N,E = self.NH,self.NW,self.NT,self.N,self.E
        # Create an empty mask, i.e. do not mask anything.
        # This is to use VideoMAEv2 implementation which requires `mask`.
        mask = torch.full([B,N], False,
                          dtype=torch.bool, device=video_batch.device)
        emb = self.encoder(video_batch,mask) # [B,N,E]
        emb = emb.reshape([B,NT,NH,NW,E])
        return emb
    
    def get_spatio_temporal_embeds(self, video_batch):
        return self.forward(video_batch)
    
    def get_spatio_temporal_embed_dims(self):
        NH,NW,NT,N,E = self.NH,self.NW,self.NT,self.N,self.E
        return (NT,NH,NW,E)

    def convert_spatio_temporal_embeds_to_video(self, spatio_temporal_embeds):
        # Just returning the input. This should return a tensor of shape [B,D]
        # where D is the dim of the video level embeds.
        return spatio_temporal_embeds
    
    def get_video_level_embed_dim(self):
        return self.E
                    

if __name__ == '__main__':
    import argparse
    from functools import partial
    from tqdm import tqdm
    import utils
    import dataset

    ckpt_f = pathlib.Path('/data/output/vit_b_hybrid_pt_800e/checkpoint-51.pth')
    args = argparse.Namespace(input_size=224,
                              mask_type='tube',
                              window_size=(8,14,14),
                              mask_ratio=0.0,
                              decoder_mask_ratio=0.5,
                              decoder_mask_type='run_cell',
                              data_root='/data/icu',
                              data_path='/data/clips.txt',
                              fname_tmpl='img_{:05}.jpg',
                              num_frames=16,
                              sampling_rate=2,
                              num_sample=2,
                              batch_size=2,
                              num_workers=0,
                              ckpt_f=ckpt_f)
    dataset = dataset.build_pretraining_dataset(args)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=partial(utils.multiple_pretrain_samples_collate,fold=False))
    device = torch.device('cuda')
    encoder = VideoMAEv2Base(pretrained_path=args.ckpt_f)
    encoder._load_pretrained()
    encoder.to(device)
    for batch in tqdm(dataloader,'batch'):
        images,_,_ = batch
        images = images.to(device)
        embs = encoder(images)
