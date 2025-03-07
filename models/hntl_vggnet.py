import torch.utils.model_zoo as model_zoo
import torch
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True, img_size=32):
        super(VGG, self).__init__()
        self.features = features
        #torch.nn.ModuleDict(features)
        self.layer_n = len(features)
        if img_size == 64:
            feature_dim = 512*2*2 
        elif img_size == 32:
            feature_dim = 512 #*7*7
        elif img_size == 112:
            feature_dim = 512*3*3 #*7*7
        self.feature_dim = feature_dim
        self.classifier1 = nn.Sequential(
            #self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, num_classes),
        )
        if init_weights:
            self._initialize_weights()
        
        #self.c1 = nn.Conv2d(64, 64, kernel_size=1, padding=1, groups=64)

    def forward(self, x, y=None):
        if y == None:
            x = self.features[:10](x)
            x = self.features[10:](x)
            #for i in range(self.layer_n):
                #if i == 0:
                #    x = self.features[str(i)](x)
                #    self.c1.weight = torch.nn.Parameter(nn.functional.sigmoid(self.c1.weight)).to('cuda')
                #    x = self.c1(x)
                #else:
                    #x = self.features[str(i)](x)
            x = x.view(x.size(0), -1)
            x = self.classifier1(x)
            return x
        else:
            x0 = self.features[:10](x)
            x = self.features[10:](x0)
            x = x.view(x.size(0), -1)
            x = self.classifier1(x)

            y0 = self.features[:10](y)
            y = self.features[10:](y0)
            y = y.view(y.size(0), -1)
            y = self.classifier1(y)
            return x, y, x0, y0
        
    def forward_f(self, x):
        x = self.features[:10](x)
        x = self.features[10:](x)
        x = x.view(x.size(0), -1)
        f = x
        x = self.classifier1(x)
        return x, f

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward_layer_f(self, x):
        layer_f = []
        for layer in self.features:
            x = layer(x)
            layer_f.append(x)
        x = x.view(x.size(0), -1)
        # f = x
        x = self.classifier1(x)
        return x, layer_f


def creat_disentangle_model(config, only_student=False, num_styles=2, teacher_network=None):
    if teacher_network is None:
        teacher_network = config.teacher_network

    # load VGG feature w/ or w/o pretrain
    if teacher_network == 'vgg11':
        feature = VGG_feature(make_layers(cfg['A'], batch_norm=False), init_weights=True)
    elif teacher_network == 'vgg11bn':
        feature = VGG_feature(make_layers(cfg['A'], batch_norm=True), init_weights=True)
    elif teacher_network == 'vgg13':
        feature = VGG_feature(make_layers(cfg['B'], batch_norm=False), init_weights=True)
    elif teacher_network == 'vgg13bn':
        feature = VGG_feature(make_layers(cfg['B'], batch_norm=True), init_weights=True)
    elif teacher_network == 'vgg19':
        feature = VGG_feature(make_layers(cfg['E'], batch_norm=False), init_weights=True)
    elif teacher_network == 'vgg19bn':
        feature = VGG_feature(make_layers(cfg['E'], batch_norm=True), init_weights=True)

    if config.image_size == 32:
        f_dim = 512
    elif config.image_size == 64:
        f_dim = 512 if config.HNTL_adapt_pooling else 512*2*2
    elif config.image_size == 128:
        # f_dim = 512 if config.HNTL_adapt_pooling else 0
        if config.adapt_pooling:
            raise Exception

    model_c = VGG_feature_VAE_enc(copy.deepcopy(feature),
                                  img_size=config.image_size,
                                  f_dim=f_dim,
                                  num_class=config.num_classes,
                                  pooling=config.HNTL_adapt_pooling)
    model_s = VGG_feature_VAE_enc(copy.deepcopy(feature),
                                  img_size=config.image_size,
                                  f_dim=f_dim,
                                  num_class=num_styles,
                                  pooling=config.HNTL_adapt_pooling)  # 2 domains for DA
    model_dec = VGG_feature_VAE_dec(img_size=config.image_size,
                                    f_dim=f_dim,
                                    pooling=config.HNTL_adapt_pooling)

    if config.teacher_pretrain:
        if config.teacher_network == 'vgg11':
            model_c.feature.load_state_dict(model_zoo.load_url(model_urls['vgg11']), strict=False)
            model_s.feature.load_state_dict(model_zoo.load_url(model_urls['vgg11']), strict=False)
        elif config.teacher_network == 'vgg11bn':
            model_c.feature.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']), strict=False)
            model_s.feature.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']), strict=False)
        if config.teacher_network == 'vgg13':
            model_c.feature.load_state_dict(model_zoo.load_url(model_urls['vgg13']), strict=False)
            model_s.feature.load_state_dict(model_zoo.load_url(model_urls['vgg13']), strict=False)
        elif config.teacher_network == 'vgg13bn':
            model_c.feature.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']), strict=False)
            model_s.feature.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']), strict=False)
        if config.teacher_network == 'vgg19':
            model_c.feature.load_state_dict(model_zoo.load_url(model_urls['vgg19']), strict=False)
            model_s.feature.load_state_dict(model_zoo.load_url(model_urls['vgg19']), strict=False)
        elif config.teacher_network == 'vgg19bn':
            model_c.feature.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']), strict=False)
            model_s.feature.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']), strict=False)

    # if only_student:
    #     return model_c.feature

    return model_c, model_s, model_dec


class VGG_feature(nn.Module):
    def __init__(self, features, init_weights=True):
        super(VGG_feature, self).__init__()
        self.features = features
        self.layer_n = len(features)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        f = self.features(x)
        return f

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class VGG_feature_VAE_dec(nn.Module):
    def __init__(self, img_size=32, f_dim=512, init_weights=True, pooling=True):
        super(VGG_feature_VAE_dec, self).__init__()
        # feature extractor
        self.img_size = img_size
        self.pooling = pooling
        self.f_dim = f_dim
        # reconstruction
        if img_size == 32: 
            hidden_dims = [256, 128, 64, 32]
            assert f_dim == 512
        elif img_size == 64:
            # raise Exception
            if self.pooling:
                hidden_dims = [256, 128, 64, 32, 32]
                assert f_dim == 512
            else: 
                hidden_dims = [512, 256, 128, 64, 32]
                assert f_dim == 512*2*2
        elif img_size == 128:
            if self.pooling:
                hidden_dims = [256, 128, 64, 32, 32, 32]
                assert f_dim == 512
            else: 
                raise Exception
        self.decoder = Data_Decoder_CIFAR(hidden_dims=hidden_dims, z_dim=f_dim)
        # self.decoder = Data_Decoder_MNIST(z_dim=f_dim) #bug
        if init_weights:
            self._initialize_weights()

    def forward(self, f_rp_c, f_rp_s):
        f_rp = torch.cat([f_rp_c, f_rp_s], dim=1)
        x_rec = self.decoder(f_rp)
        return x_rec

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    

class VGG_feature_VAE_enc(nn.Module):
    def __init__(self, feature, img_size=32, f_dim=512, num_class=10, init_weights=True, pooling=True):
        super(VGG_feature_VAE_enc, self).__init__()
        # feature extractor
        self.feature = feature
        self.pooling = pooling
        self.img_size = img_size
        self.f_dim = f_dim
        latent_dim =  int(f_dim/2)
        # vae
        self.fc_mu = nn.Linear(f_dim, latent_dim)
        self.fc_logvar = nn.Linear(f_dim, latent_dim)
        # classifier for content or style
        self.classifier = nn.Linear(latent_dim, num_class)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        f = self.feature(x)
        # f = f.squeeze() # batch * f_dim
        if self.pooling:
            f = F.adaptive_avg_pool2d(f, 1)
        f = f.view(f.shape[0], -1)
        f_mu, f_logvar = self.fc_mu(f), self.fc_logvar(f)
        f_rp = self.reparameterize(f_mu, f_logvar)
        pred = self.classifier(f_rp)
        # x_rec = self.decoder(f_rp)
        if self.training:
            outputs = {
                'f_mu': f_mu,
                'f_logvar': f_logvar,
                'f_rp': f_rp,
                'pred': pred,
                # 'x_rec': x_rec,
            }
        else: 
            outputs = pred
        return outputs
    
    def forward_full(self, x):
        f = self.feature(x)
        # f = f.squeeze() # batch * f_dim
        if self.pooling:
            f = F.adaptive_avg_pool2d(f, 1)
        f = f.view(f.shape[0], -1)
        f_mu, f_logvar = self.fc_mu(f), self.fc_logvar(f)
        f_rp = self.reparameterize(f_mu, f_logvar)
        pred = self.classifier(f_rp)
        # x_rec = self.decoder(f_rp)
        outputs = {
            'f_mu': f_mu,
            'f_logvar': f_logvar,
            'f_rp': f_rp,
            'pred': pred,
            # 'x_rec': x_rec,
        }
        return outputs

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def disentangle(self, f_rp, reverse=False):
        if reverse:
            raise Exception
            # using gradiant reveral layer: use entropy minimization loss
            # else, use negative entropy
            x = self.grl(x)
        with torch.no_grad():
            pred = self.classifier(f_rp)
        return pred
    
    def pred(self, f_rp):
        with torch.no_grad():
            pred = self.classifier(f_rp)
        return pred


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for i in range(len(cfg)):
        v = cfg[i]
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    
    #tmp_list = [(str(i), layers[i]) for i in range(len(layers))]
    #tmp_ret = nn.Sequential(
    #    collections.OrderedDict(
    #        tmp_list
    #    )
    #)
    #return tmp_ret
    return nn.Sequential(*layers)
    #return tmp_list


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A']), **kwargs)

    model.load_state_dict(model_zoo.load_url(model_urls['vgg11']), strict = False)
    return model


def vgg11_bn(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)

    model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']), strict = False)
    return model


def vgg13(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13']), strict = False)
    return model


def vgg13_bn(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']), strict = False)
    return model


def vgg19(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19']), strict = False)
    return model


def vgg19_bn(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']), strict = False)
    return model


class Data_Decoder_MNIST(nn.Module):

    def __init__(self, num_classes=2, hidden_dims=[256, 128, 64, 32], z_dim=1):
        super().__init__()
        self.decoder_input = nn.Linear(z_dim, hidden_dims[0] * 4)
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               ),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=1,
                      kernel_size=4),
            nn.Sigmoid())

    def forward(self, z_1, z_2):
        out = torch.cat((z_1, z_2), dim=1)
        out = self.decoder_input(out)
        out = out.view(-1, 256, 2, 2)
        out = self.decoder(out)
        out = self.final_layer(out)
        return out


class Data_Decoder_CIFAR(nn.Module):
    def __init__(self, hidden_dims=[256, 128, 64, 32], z_dim=1):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.decoder_input = nn.Linear(z_dim, hidden_dims[0] * 4)
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3,
                      kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
            )

    def forward(self, z):
        out = self.decoder_input(z)
        out = out.view(-1, self.hidden_dims[0], 2, 2)
        out = self.decoder(out)
        out = self.final_layer(out)
        return out


if __name__ == '__main__':
    f_dim = 2048
    # hidden_dims = [256, 128, 64, 32]
    hidden_dims = [256, 128, 64, 32, 32, 32]
    decoder = Data_Decoder_CIFAR(hidden_dims=hidden_dims, z_dim=f_dim)
    # x = torch.zeros(1, 3, 32, 32)
    f = torch.zeros(1, f_dim)
    out = decoder(f)
    exit
