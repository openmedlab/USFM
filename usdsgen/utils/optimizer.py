from functools import partial

from torch import optim as optim


def build_optimizer(config, model, logger):
    model_cfg = config.model.model_cfg
    logger.info(f">>>>>>>>>> Build Optimizer for model {model_cfg.type} <<<<<<<<<<")

    if model_cfg.type in ["swin", "vit"]:
        if model_cfg.type == "swin":
            num_layers = sum(model_cfg.depths)
            get_layer_func = partial(
                get_swin_layer, num_layers=num_layers + 2, depths=model_cfg.depths
            )
        else:
            num_layers = model_cfg.depth
            get_layer_func = partial(get_vit_layer, num_layers=num_layers + 2)
        scales = list(
            config.train.layer_decay**i for i in reversed(range(num_layers + 2))
        )

        skip = {}
        skip_keywords = {}
        if hasattr(model, "no_weight_decay"):
            skip = model.no_weight_decay()
            logger.info(f"No weight decay: {skip}")
        if hasattr(model, "no_weight_decay_keywords"):
            skip_keywords = model.no_weight_decay_keywords()
            logger.info(f"No weight decay keywords: {skip_keywords}")

        parameters = get_finetune_param_groups(
            model,
            logger,
            config.train.base_lr,
            config.train.weight_decay,
            get_layer_func,
            scales,
            skip,
            skip_keywords,
        )
    else:
        parameters = model.parameters()

    opt_lower = config.train.optimizer.name.lower()
    optimizer = None
    if opt_lower == "sgd":
        optimizer = optim.SGD(
            parameters,
            momentum=config.train.optimizer.momentum,
            nesterov=True,
            lr=config.train.base_lr,
            weight_decay=config.train.weight_decay,
        )
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(
            parameters,
            eps=config.train.optimizer.eps,
            betas=config.train.optimizer.betas,
            lr=config.train.base_lr,
            weight_decay=config.train.weight_decay,
        )

    # logger.info(optimizer)
    return optimizer


def get_pretrain_param_groups(model, logger, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []
    has_decay_name = []
    no_decay_name = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            len(param.shape) == 1
            or name.endswith(".bias")
            or (name in skip_list)
            or check_keywords_in_name(name, skip_keywords)
        ):
            no_decay.append(param)
            no_decay_name.append(name)
        else:
            has_decay.append(param)
            has_decay_name.append(name)
    logger.info(f"No decay params: {no_decay_name}")
    logger.info(f"Has decay params: {has_decay_name}")
    return [{"params": has_decay}, {"params": no_decay, "weight_decay": 0.0}]


def get_vit_layer(name, num_layers):
    if name in ("cls_token", "mask_token", "pos_embed"):
        return 0
    elif name.startswith("patch_embed"):
        return 0
    elif name.startswith("rel_pos_bias"):
        return num_layers - 1
    elif name.startswith("blocks"):
        layer_id = int(name.split(".")[1])
        return layer_id + 1
    else:
        return num_layers - 1


def get_swin_layer(name, num_layers, depths):
    if name in ("mask_token"):
        return 0
    elif name.startswith("patch_embed"):
        return 0
    elif name.startswith("layers"):
        layer_id = int(name.split(".")[1])
        block_id = name.split(".")[3]
        if block_id == "reduction" or block_id == "norm":
            return sum(depths[: layer_id + 1])
        layer_id = sum(depths[:layer_id]) + int(block_id)
        return layer_id + 1
    else:
        return num_layers - 1


def get_finetune_param_groups(
    model,
    logger,
    lr,
    weight_decay,
    get_layer_func,
    scales,
    skip_list=(),
    skip_keywords=(),
):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            len(param.shape) == 1
            or name.endswith(".bias")
            or (name in skip_list)
            or check_keywords_in_name(name, skip_keywords)
        ):
            group_name = "no_decay"
            this_weight_decay = 0.0
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_layer_func is not None:
            layer_id = get_layer_func(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if scales is not None:
                scale = scales[layer_id]
            else:
                scale = 1.0

            parameter_group_names[group_name] = {
                "group_name": group_name,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": lr * scale,
                "lr_scale": scale,
            }
            parameter_group_vars[group_name] = {
                "group_name": group_name,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": lr * scale,
                "lr_scale": scale,
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    # logger.info("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin
