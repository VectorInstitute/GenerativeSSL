import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.image_size = 128  # The size of the images to generate.
    config.class_cond = False  # If true, use class conditional generation.
    config.type_model = "simclr"  # Type of model to use (e.g., simclr, dino).
    config.use_head = False  # If true, use the projector/head for SSL representation.
    config.pre_trained_model_dir = "/ssd003/projects/aieng/genssl" # Path to the directory containing the model.
    config.model_path = ""  # Replace with the path to your model if you have one.
    config.use_ddim = False  # If true, use DDIM sampler.
    config.no_shared = True  # If false, enables squeeze and excitation.
    config.clip_denoised = True  # If true, clip denoised images.
    config.attention_resolutions = "32,16,8"  # Resolutions to use for attention layers.
    config.diffusion_steps = 100  # Number of diffusion steps.
    config.learn_sigma = True  # If true, learn the noise level.
    config.noise_schedule = "linear"  # Type of noise schedule (e.g., linear).
    config.num_channels = 256  # Number of channels in the model.
    config.num_heads = 4  # Number of attention heads.
    config.num_res_blocks = 2  # Number of residual blocks.
    config.resblock_updown = True  # If true, use up/down sampling in resblocks.
    config.use_fp16 = False  # If true, use 16-bit floating point precision.
    config.use_scale_shift_norm = True  # If true, use scale-shift normalization.
    config.ssl_image_size = 224  # Size of the input images for the SSL model.
    config.ssl_image_channels = (
        3  # Number of channels of the input images for the SSL model.
    )
    config.timestep_respacing = "ddim2" # Type of timestep respacing (e.g., ddim25).

    return config
