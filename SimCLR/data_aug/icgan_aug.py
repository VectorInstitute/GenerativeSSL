import os
import torch 
import numpy as np
from PIL import Image as Image_PIL
from scipy.stats import truncnorm
from torch import nn
import torchvision.transforms as transforms
from icgan.inference import utils as inference_utils
from icgan.data_utils import utils as data_utils

class ICGANInference:
    def __init__(self, config, device_id):

        self.config = config
        np.random.seed(self.config.seed)
        torch.manual_seed(np.random.randint(self.config.seed))
        self.state = np.random.RandomState(self.config.seed)
        self.device_id = device_id

        self.feature_extractor = self.load_feature_extractor(self.config.feat_ext_path)
        self.model = self.load_generative_model()
        self.replace_to_inplace_relu(self.model)

    def replace_to_inplace_relu(self, model):
        for child_name, child in model.named_children():
            if isinstance(child, nn.ReLU):
                setattr(model, child_name, nn.ReLU(inplace=False))
            else:
                self.replace_to_inplace_relu(child)

    def load_icgan(self, root_=''):
        root = os.path.join(root_, self.config.experiment_name)
        config = torch.load("%s/%s.pth" % (root, "state_dict_best0"))['config']
        config["weights_root"] = root_
        config["model_backbone"] = 'biggan'
        config["experiment_name"] = self.config.experiment_name
        G, config = inference_utils.load_model_inference(config)
        G = G.cuda(self.device_id)
        G.eval()
        return G

    def get_output(self, noise_vector, input_label, input_features):
        if self.config.stochastic_truncation:
            with torch.no_grad():
                trunc_indices = noise_vector.abs() > 2*self.config.truncation
                size = torch.count_nonzero(trunc_indices).cpu().numpy()
                trunc = truncnorm.rvs(-2*self.config.truncation, 2*self.config.truncation, size=(1,size)).astype(np.float32)
                noise_vector.data[trunc_indices] = torch.tensor(trunc, requires_grad=True, device='cuda')
        else:
            noise_vector = noise_vector.clamp(-2*self.config.truncation, 2*self.config.truncation)
  
        out = self.model(noise_vector, input_label, input_features.cuda())
        return out

    def normality_loss(self, vec):
        mu2 = vec.mean().square()
        sigma2 = vec.var()
        return mu2+sigma2-torch.log(sigma2)-1

    def load_generative_model(self):
        model = self.load_icgan(root_='')
        return model

    def load_feature_extractor(self, feat_ext_path):
        feature_extractor = data_utils.load_pretrained_feature_extractor(feat_ext_path, feature_extractor='selfsupervised')
        feature_extractor = feature_extractor.cuda(self.device_id)
        feature_extractor.eval()
        return feature_extractor

    def preprocess_input_image(self, input_image_path, size): 
        pil_image = Image_PIL.open(input_image_path).convert('RGB')
        transform_list =  transforms.Compose([data_utils.CenterCropLongEdge(), transforms.Resize((size,size)), transforms.ToTensor(), transforms.Normalize(self.config.norm_mean, self.config.norm_std)])
        tensor_image = transform_list(pil_image)
        tensor_image = torch.nn.functional.interpolate(tensor_image.unsqueeze(0), 224, mode="bicubic", align_corners=True)
        return tensor_image

    def preprocess_generated_image(self, image):
        transform_list =  transforms.Normalize(self.config.norm_mean, self.config.norm_std)
        image = transform_list(image*0.5 + 0.5)
        image = torch.nn.functional.interpolate(image, 224, mode="bicubic", align_corners=True)
        return image

    def __call__(self, input_image_tensor):
        with torch.no_grad():
            input_features, _ = self.feature_extractor(input_image_tensor.unsqueeze(0).cuda(self.device_id))
        input_features/=torch.linalg.norm(input_features, dim=-1, keepdims=True)

        # Create noise, instance and class vector
        noise_vector = truncnorm.rvs(-2*self.config.truncation, 2*self.config.truncation, 
                                     size=(self.config.num_samples, self.config.noise_size), 
                                     random_state=self.state).astype(np.float32)
        noise_vector = torch.tensor(noise_vector, requires_grad=False).cuda(self.device_id)
        instance_vector = torch.tensor(input_features, requires_grad=False).repeat(self.config.num_samples, 1).cuda(self.device_id)
        input_label = None

        sample = self.get_output(noise_vector, input_label 
                                if input_label is not None else None, instance_vector
                                if instance_vector is not None else None)

        sample = sample.detach().squeeze(0)

        return sample