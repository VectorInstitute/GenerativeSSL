import torch
from torch import nn
from torchvision import models

from ..exceptions.exceptions import InvalidBackboneError


class PretrainedResNet(nn.Module):
    def __init__(self, base_model, pretrained_model_file, linear_eval=True, num_classes=10):
        super(PretrainedResNet, self).__init__()

        self.pretrained_model_file = pretrained_model_file

        self.resnet_dict = {
            "resnet18": models.resnet18(pretrained=False, num_classes=num_classes),
            "resnet50": models.resnet50(pretrained=False, num_classes=num_classes),
        }

        self.backbone = self._get_basemodel(base_model)

        # load pretrained weights
        log = self._load_pretrained()

        assert log.missing_keys == ["fc.weight", "fc.bias"]

        if linear_eval:
            # freeze all layers but the last fc
            self._freeze_backbone()
            parameters = list(filter(lambda p: p.requires_grad, self.backbone.parameters()))
            assert len(parameters) == 2  # fc.weight, fc.bias

    def _load_pretrained(self):
        checkpoint = torch.load(self.pretrained_model_file, map_location='cpu')
        state_dict = checkpoint["state_dict"]
        for k in list(state_dict.keys()):
            if k.startswith("module.backbone."):
                if not k.startswith("module.backbone.fc"):
                    # remove prefix
                    state_dict[k[len("module.backbone.") :]] = state_dict[k]
            del state_dict[k]
        log = self.backbone.load_state_dict(state_dict, strict=False)
        return log
        

    def _freeze_backbone(self):
        # freeze all layers but the last fc
        for name, param in self.backbone.named_parameters():
            if name not in ["fc.weight", "fc.bias"]:
                param.requires_grad = False
        return


    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50",
            )
        else:
            return model

    def forward(self, x):
        return self.backbone(x)
