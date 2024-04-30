import torch.nn.parallel
import torchvision.models as models


def _load_torchvision_model(model, pretrained=True):
    model = getattr(models, model)(pretrained=pretrained)
    return model


def _init_data_parallel(model, device):
    if device == 'gpu':
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        model = torch.nn.DataParallel(model).cuda()
        return model


def get_model(args):

    model, start_epoch, optimizer = None, None, None
    # load model:
    print('| loading model {} ...'.format(args.model))

    model = _load_torchvision_model(args.model,
                                    pretrained=args.pretrained)

    return model, start_epoch, optimizer
