import torch
from .guided_diffusion_rcdm import dist_util
from .guided_diffusion_rcdm.get_ssl_models import get_model
from .guided_diffusion_rcdm.get_rcdm_models import get_dict_rcdm_model
from .guided_diffusion_rcdm.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion
)

class RCDMInference(object):
    def __init__(self, config):
        """
        Initialize the RCDMInference class with necessary parameters and load models.
        """
        self.config = config

        # Load SSL model
        self.ssl_model = get_model(self.config.type_model, self.config.use_head).cuda().eval()
        for p in self.ssl_model.parameters():
            p.requires_grad = False

        # Load RCDM model
        model_defaults = model_and_diffusion_defaults()
        model_args = {k: getattr(self.config, k, model_defaults[k]) for k in model_defaults}
        self.model, self.diffusion = create_model_and_diffusion(
            **model_args, G_shared=self.config.no_shared, feat_cond=True, ssl_dim=self.ssl_model(torch.zeros(1, 3, 224, 224).cuda()).size(1)
        )

        if self.config.model_path == "":
            trained_model = get_dict_rcdm_model(self.config.type_model, self.config.use_head)
        else:
            trained_model = torch.load(self.config.model_path, map_location="cpu")
        self.model.load_state_dict(trained_model, strict=True)
        self.model.to(dist_util.dev())

    def __call__(self, img):
        """
        Run the RCDM model inference on an images.

        Args:
            img (torch.Tensor): An image to apply RCDM to.

        Returns:
            List[torch.Tensor]: List of generated image tensors.
        """
        print("Starting RCDM model inference...")

        sample_fn = self.diffusion.p_sample_loop if not self.config.use_ddim else self.diffusion.ddim_sample_loop

        img = img.unsqueeze(0).repeat(self.config.num_images, 1, 1, 1).cuda()
        model_kwargs = {}

        with torch.no_grad():
            feat = self.ssl_model(img).detach()
            model_kwargs["feat"] = feat
        sample = sample_fn(
            self.model,
            (self.config.num_images, 3, self.config.image_size, self.config.image_size),
            clip_denoised=self.config.clip_denoised,
            model_kwargs=model_kwargs,
        )

        print("Sampling completed!")
        return sample.squeeze(0)