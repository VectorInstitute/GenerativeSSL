# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import torch
from solo.methods import WMSE

from .utils import gen_base_cfg, gen_batch, gen_trainer, prepare_dummy_dataloaders


def test_wmse():
    cfg = gen_base_cfg("wmse", batch_size=8, num_classes=100, momentum=True)
    method_kwargs = {
        "proj_hidden_dim": 1024,
        "proj_output_dim": cfg.optimizer.batch_size // 4,
        "whitening_size": cfg.optimizer.batch_size // 2,
        "whitening_iters": 1,
        "whitening_eps": 1e-2,
    }

    cfg.method_kwargs = method_kwargs
    model = WMSE(cfg)

    # test arguments
    model.add_and_assert_specific_cfg(cfg)

    # test parameters
    assert model.learnable_params is not None

    # test forward
    batch, _ = gen_batch(cfg.optimizer.batch_size, cfg.data.num_classes, "imagenet100")
    out = model(batch[1][0])
    assert (
        "logits" in out
        and isinstance(out["logits"], torch.Tensor)
        and out["logits"].size() == (cfg.optimizer.batch_size, cfg.data.num_classes)
    )
    assert (
        "feats" in out
        and isinstance(out["feats"], torch.Tensor)
        and out["feats"].size() == (cfg.optimizer.batch_size, model.features_dim)
    )
    assert (
        "z" in out
        and isinstance(out["z"], torch.Tensor)
        and out["z"].size()
        == (cfg.optimizer.batch_size, method_kwargs["proj_output_dim"])
    )

    for num_large_crops in [2, 4]:
        # imagenet
        cfg.data.num_large_crops = num_large_crops
        cfg.method_kwargs["output_dim"] = cfg.optimizer.batch_size // 4
        cfg.method_kwargs["whitening_size"] = cfg.optimizer.batch_size // 2
        model = WMSE(cfg)

        trainer = gen_trainer(cfg)
        train_dl, val_dl = prepare_dummy_dataloaders(
            "imagenet100",
            num_large_crops=cfg.data.num_large_crops,
            num_small_crops=0,
            num_classes=cfg.data.num_classes,
            batch_size=cfg.optimizer.batch_size,
        )
        trainer.fit(model, train_dl, val_dl)

        # cifar
        cfg.method_kwargs["output_dim"] = cfg.optimizer.batch_size // 4
        cfg.method_kwargs["whitening_size"] = cfg.optimizer.batch_size // 2
        model = WMSE(cfg)

        trainer = gen_trainer(cfg)
        train_dl, val_dl = prepare_dummy_dataloaders(
            "cifar10",
            num_large_crops=cfg.data.num_large_crops,
            num_small_crops=0,
            num_classes=cfg.data.num_classes,
            batch_size=cfg.optimizer.batch_size,
        )
        trainer.fit(model, train_dl, val_dl)
