import yaml

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file) 
    return config
unet_config = load_config("/home/frank/JIAYUANLI/OVRS/OVRS/cat_seg/modeling/transformer/mamba_decoder/config/train_config.yaml")
f_maps_dim = unet_config['model']['f_maps']

in_channels = 1
input_res = [1,128,128,128]

is_do_ds = False
#-------------------------------- for full hyma---------------------------------------------------
dim_after_stem = unet_config['model']['in_channels']

#start
start_res = [32, 32, 32] 
reduce_rate_before_hymavi_start = 4
v1_dim_start = start_res[1]*start_res[2]*f_maps_dim[0]//reduce_rate_before_hymavi_start 
v2_dim_start = start_res[1]*start_res[0]*f_maps_dim[0]//reduce_rate_before_hymavi_start
v3_dim_start = start_res[0]*start_res[2]*f_maps_dim[0]//reduce_rate_before_hymavi_start
embed_dim_att_start = 96 #qk vectors (q=k=96)
value_dim_att_start = 96 #v vector
num_attention_heads_start = 3
mamba_states_start = 16

#end
end_res = [32, 32, 32] 
reduce_rate_before_hymavi_end = 4
v1_dim_end = end_res[1]*end_res[2]*f_maps_dim[0]//reduce_rate_before_hymavi_end
v2_dim_end = end_res[1]*end_res[0]*f_maps_dim[0]//reduce_rate_before_hymavi_end
v3_dim_end = end_res[0]*end_res[2]*f_maps_dim[0]//reduce_rate_before_hymavi_end
embed_dim_att_end = 96
value_dim_att_end = 96 #v vector
num_attention_heads_end = 3
mamba_states_end = 16

#---------------------------- for Hymavi---------------------------------------
intermediate_mamba_size_expand_rate = 1 #to ensure the homogeneous size after transformation (and can be resized to the ogirinal size), this should be set by 1
num_attention_heads = 24
mamba_proj_bias=False

# After the encoder, the feature map has shape (B, C, D, H, W). 
# Note that view 1 is axial, view 2 is sagittal, and view 3 is coronal. 
# Therefore, v1 now is reshape to (B, C* H * W, D) <=> (Batch, Dim, Sequence_len)
H = 4
W = 4
D = 4
C = f_maps_dim[-1]
reduce_rate_before_hymavi = 2
v1_dim = H*W*C//reduce_rate_before_hymavi 
v2_dim = H*D*C//reduce_rate_before_hymavi
v3_dim = W*D*C//reduce_rate_before_hymavi

embed_dim_att = 768 #dim of q, k, v
value_dim_att = 768 #v vector

mamba_states = 48 

num_classes = 16

#-------------------------------for decoder---------------------------------------------------
swin_input_dec_resolutions = [[16,16,16], [8,8,8]] #this should be in the opposite order since the layer index of the decoder is from End layer


#---------------------params for swin-----------------------------------
reduce_rate_before_hymaviswin = [4, 8] 
window_sizes = [[3,5,5], [3,7,7]] 
swin_input_resolutions = [[16,16,16], [8,8,8]]

hidden_sz_swin = []
swin_f_map_dims = []
seq_len_swin = [] #for calculate attention in swin, here, the input shape for window attention is (B, L, D) -> L is seq_len (L is different for each view and is calculated after applying window partition)
for i, wds in enumerate(window_sizes):
    D_, H_, W_ = wds[0], wds[1], wds[2]
    cur_fmap_dim = f_maps_dim[(len(f_maps_dim)-1)-len(swin_input_resolutions) + i]
    v1 = H_*W_* cur_fmap_dim//reduce_rate_before_hymaviswin[i]
    v2 = H_*D_* cur_fmap_dim//reduce_rate_before_hymaviswin[i]
    v3 = W_*D_* cur_fmap_dim//reduce_rate_before_hymaviswin[i]
    hidden_sz_swin.append([v1,v2,v3])
    swin_f_map_dims.append(cur_fmap_dim)
    
    seq_len_v1 = D_
    seq_len_v2 = W_
    seq_len_v3 = H_

    seq_len_swin.append([seq_len_v1, seq_len_v2, seq_len_v3])
print(">>>>>>>>>>>>>>>>>> HIDDEN SIZES OF SWIN: ", hidden_sz_swin, " <<<<<<<<<<<<<<<<<<")

swin_shift_sz = [[1,2,2], [1,3,3]]
embed_dim_att_swin = [192, 384] 
value_dim_att_swin = [192, 384] 
num_attention_heads_swin = [6, 12] 
mamba_states_swin = [16, 24]
reduce_rate_in_swin = 8