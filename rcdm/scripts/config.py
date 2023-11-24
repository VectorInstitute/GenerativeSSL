import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    config.image_size = 128
    config.class_cond = False
    config.type_model = "simclr"
    config.use_head = False
    config.model_path = ""  # Replace with the path to your model if you have one
    config.num_images = 2
    config.use_ddim = False
    config.no_shared = True
    config.clip_denoised = True
    config.attention_resolutions = "32,16,8"
    config.diffusion_steps = 1000
    config.learn_sigma = True
    config.noise_schedule = "linear"
    config.num_channels = 256
    config.num_heads = 4
    config.num_res_blocks = 2
    config.resblock_updown = True
    config.use_fp16 = False
    config.use_scale_shift_norm = True
    
    return config