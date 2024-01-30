import torch
from rcdm.guided_diffusion_rcdm.get_ssl_models import get_model
from rcdm.guided_diffusion_rcdm.get_rcdm_models import get_dict_rcdm_model
from rcdm.guided_diffusion_rcdm.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion
)


class RCDMInference(object):
    def __init__(self, config, device_id):
        """
        Initialize the RCDMInference class with necessary parameters and load models.
        """
        self.config = config
        self.device_id = device_id

        # Load SSL model
        self.ssl_model = get_model(self.config.type_model, self.config.use_head, self.config.model_dir).cuda(self.device_id).eval()
        for p in self.ssl_model.parameters():
            p.requires_grad = False

        # Load RCDM model
        model_defaults = model_and_diffusion_defaults()
        model_args = {k: getattr(self.config, k, model_defaults[k]) for k in model_defaults}
        
        self.model, self.diffusion = create_model_and_diffusion(
            **model_args, G_shared=self.config.no_shared, feat_cond=True, ssl_dim=self.ssl_model(torch.zeros(1, config.ssl_image_channels, config.ssl_image_size, config.ssl_image_size).cuda()).size(1)
        )

        if self.config.model_path == "":
            trained_model = get_dict_rcdm_model(self.config.type_model, self.config.use_head, self.config.model_dir)
        else:
            trained_model = torch.load(self.config.model_path, map_location="cpu")
        self.model.load_state_dict(trained_model, strict=True)
        self.model.cuda(self.device_id)

    def __call__(self, img):
        """
        Run the RCDM model inference on an images.

        Args:
            img (torch.Tensor): An image to apply RCDM to.

        Returns:
            List[torch.Tensor]: List of generated image tensors.
        """

        sample_fn = self.diffusion.p_sample_loop if not self.config.use_ddim else self.diffusion.ddim_sample_loop

        img = img.unsqueeze(0).repeat(1, 1, 1, 1).cuda(self.device_id)
        model_kwargs = {}

        with torch.no_grad():
            feat = self.ssl_model(img).detach()
            model_kwargs["feat"] = feat
        sample = sample_fn(
            self.model,
            (1, 3, self.config.image_size, self.config.image_size),
            clip_denoised=self.config.clip_denoised,
            model_kwargs=model_kwargs,
        )

        return sample.detach().squeeze(0)