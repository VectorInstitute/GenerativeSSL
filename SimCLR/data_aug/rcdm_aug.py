import torch
from torchvision import transforms

from icgan.data_utils import utils as data_utils
from rcdm.guided_diffusion_rcdm import dist_util
from rcdm.guided_diffusion_rcdm.get_rcdm_models import get_dict_rcdm_model
from rcdm.guided_diffusion_rcdm.get_ssl_models import get_model
from rcdm.guided_diffusion_rcdm.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)


class RCDMInference(object):
    def __init__(self, config, device_id):
        """
        Initialize the RCDMInference class with necessary parameters and load models.
        """
        self.config = config
        self.device_id = device_id

        # Load SSL model
        self.ssl_model = (
            get_model(self.config.type_model, self.config.use_head, self.config.pretrained_models_dir)
            .cuda(self.device_id)
            .eval()
        )
        for p in self.ssl_model.parameters():
            p.requires_grad = False

        # Load RCDM model
        model_defaults = model_and_diffusion_defaults()
        model_args = {
            k: getattr(self.config, k, model_defaults[k]) for k in model_defaults
        }

        self.model, self.diffusion = create_model_and_diffusion(
            **model_args,
            G_shared=self.config.no_shared,
            feat_cond=True,
            ssl_dim=self.ssl_model(
                torch.zeros(
                    1,
                    config.ssl_image_channels,
                    config.ssl_image_size,
                    config.ssl_image_size,
                ).cuda(self.device_id)
            ).size(1),
        )

        if self.config.model_path == "":
            trained_model = get_dict_rcdm_model(
                self.config.type_model, self.config.use_head, self.config.pretrained_models_dir
            )
        else:
            trained_model = torch.load(self.config.model_path, map_location="cpu")
        self.model.load_state_dict(trained_model, strict=True)
        self.model.to(dist_util.dev())

    def preprocess_input_image(self, input_image, size=224):
        transform_list = transforms.Compose(
            [
                data_utils.CenterCropLongEdge(),
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(self.config.norm_mean, self.config.norm_std),
            ]
        )
        tensor_image = transform_list(input_image)
        tensor_image = torch.nn.functional.interpolate(
            tensor_image.unsqueeze(0), 224, mode="bicubic", align_corners=True
        )
        return tensor_image

    def __call__(self, img):
        """
        Run the RCDM model inference on an images.

        Args:
            img (torch.Tensor): An image to apply RCDM to.

        Returns:
            List[torch.Tensor]: List of generated image tensors.
        """
        print("Starting RCDM model inference...")

        sample_fn = (
            self.diffusion.p_sample_loop
            if not self.config.use_ddim
            else self.diffusion.ddim_sample_loop
        )
        print("1",img.shape)
        img = img.unsqueeze(0).repeat(1, 1, 1, 1)
        print("2",img.shape)
        img = self.preprocess_input_image(img).cuda(self.device_id)
        print("3",img.shape)
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

        print("Sampling completed!")
        return sample.squeeze(0)
