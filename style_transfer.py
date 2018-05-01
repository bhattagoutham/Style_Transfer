import numpy as np
import cv2
import argparse
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, UpSampling2D, Activation, Lambda, MaxPooling2D
import torchfile

parser = argparse.ArgumentParser()
parser.add_argument('--content-path', type=str, dest='content_path', help='Content image or folder of images')
parser.add_argument('--style-path', type=str, dest='style_path', help='Style image or folder of images')
parser.add_argument('--alpha', type=float, help="Alpha blend value", default=0.9)
parser.add_argument('--out-path', type=str, dest='out_path', help='Output folder path')
parser.add_argument('--live-path', type=str, dest='live_path', help='live folder path')
parser.add_argument('--weight-path', type=str, dest='weight_path', help='weights path')
args = parser.parse_args()

def disp_img(img, name):
        cv2.imshow(name,img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def pad_reflect(x, padding=1):
    return tf.pad(
      x, [[0, 0], [padding, padding], [padding, padding], [0, 0]],
      mode='REFLECT')

def wct_np(content, style, alpha=0.6, eps=1e-5):
    '''Perform Whiten-Color Transform on feature maps using numpy
       See p.4 of the Universal Style Transfer paper for equations:
       https://arxiv.org/pdf/1705.08086.pdf
    '''    
    # 1xHxWxC -> CxHxW
    content_t = np.transpose(np.squeeze(content), (2, 0, 1))
    style_t = np.transpose(np.squeeze(style), (2, 0, 1))

    # CxHxW -> CxH*W
    content_flat = content_t.reshape(-1, content_t.shape[1]*content_t.shape[2])
    style_flat = style_t.reshape(-1, style_t.shape[1]*style_t.shape[2])

    mc = content_flat.mean(axis=1, keepdims=True)
    fc = content_flat - mc

    fcfc = np.dot(fc, fc.T) / (content_t.shape[1]*content_t.shape[2] - 1)
    
    Ec, wc, _ = np.linalg.svd(fcfc)

    k_c = (wc > 1e-5).sum()

    Dc = np.diag((wc[:k_c]+eps)**-0.5)

    # fc_hat = Dc.dot(Ec[:,:k_c]).dot(fc)

    fc_hat = Ec[:,:k_c].dot(Dc).dot(Ec[:,:k_c].T).dot(fc)

    ms = style_flat.mean(axis=1, keepdims=True)
    fs = style_flat - ms

    fsfs = np.dot(fs, fs.T) / (style_t.shape[1]*style_t.shape[2] - 1)

    Es, ws, _ = np.linalg.svd(fsfs)

    k_s = (ws > 1e-5).sum()
    
    Ds = np.sqrt(np.diag(ws[:k_s]+eps))

    # fcs_hat = Ds.dot(Es[:,:k_s]).dot(fc_hat)

    fcs_hat = Es[:,:k_s].dot(Ds).dot(Es[:,:k_s].T).dot(fc_hat)

    fcs_hat = fcs_hat + ms

    blended = alpha*fcs_hat + (1 - alpha)*(fc)

    # CxH*W -> CxHxW
    blended = blended.reshape(content_t.shape)
    # CxHxW -> 1xHxWxC
    blended = np.expand_dims(np.transpose(blended, (1,2,0)), 0)
    
    return np.float32(blended)



def vgg_from_t7(t7_file, net_type, layer, target_layer=None):

    '''Extract VGG layers from a Torch .t7 model into a Keras model
       Adapted from https://github.com/jonrei/tf-AdaIN/blob/master/AdaIN.py
       Converted caffe->t7 from https://github.com/xunhuang1995/AdaIN-style
    '''

    print("Creating ",net_type+"_"+str(layer))

    t7 = torchfile.load(t7_file, force_8bytes_long=True)
    
    if net_type == "encoder":
        inp = Input(shape=(None, None, 3), name='vgg_input')
    elif net_type == "decoder":
        if layer == 5:
            chanel = pow(2,5+layer-1)    
        else:
            chanel = pow(2,5+layer)
        inp = Input(shape=(None, None, chanel), name='vgg_input')

    x = inp
    # print(x.shape)
    
    for idx,module in enumerate(t7.modules):
        name = module.name.decode() if module.name is not None else None

        # print(idx," : " ,name," : ", module._typename, ", no_of filters: ", module.nOutputPlane, ", kernel_size: ", module.kH)
        
        if idx == 0:
            name = 'preprocess'  # VGG 1st layer preprocesses with a 1x1 conv to multiply by 255 and subtract BGR mean as bias

        if module._typename == b'nn.SpatialConvolution':
            filters = module.nOutputPlane
            kernel_size = module.kH
            weight = module.weight.transpose([2,3,1,0])
            bias = module.bias

            # print("model weights shape ", module.weight.shape)
            # print("transposed model weights shape ", weight.shape)
            # print("model bias shape ",  bias.shape)
            
            x = Conv2D(filters, kernel_size, padding='valid', activation=None, name=name,
                        kernel_initializer=lambda shape: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape: K.constant(bias, shape=shape),
                        trainable=False)(x)

        elif module._typename == b'nn.ReLU':
            x = Activation('relu', name=name)(x)
        
        elif module._typename == b'nn.SpatialMaxPooling':
            x = MaxPooling2D(padding='same', name=name)(x)
        
        elif module._typename == b'nn.SpatialUpSamplingNearest': # Not needed for VGG
            x = UpSampling2D(name=name)(x)
        
        elif module._typename == b'nn.SpatialReflectionPadding':
            x = Lambda(pad_reflect)(x)
        else:
            raise NotImplementedError(module._typename)

        # if name == target_layer:
            # print("Reached target layer", target_layer)
            # break

        # print()

    
    model = Model(inputs=inp, outputs=x)
    # print("\n-------------------------------------------------------------------------\n")
    return model


def stylize(content_path, style_path, out_path, alpha, live_path, weight_path):

    # path to vgg_normalised_conv#_#.t7 and  feature_invertor_conv#_#.t7 files
    encoder_path = weight_path
    decoder_path = weight_path
    encoder = []
    decoder = []

    for i in range(5):
        l = str(i+1)
        encoder.append(vgg_from_t7(encoder_path+'/vgg_normalised_conv'+l+'_1'+'.t7', "encoder", i+1))
        decoder.append(vgg_from_t7(decoder_path+'/feature_invertor_conv'+l+'_1'+'.t7', "decoder", i+1))
        print()


    # style_path = '2.jpg'
    # content_path = '21.jpg'

    style = cv2.imread(style_path)
    content = cv2.imread(content_path)

    # disp_img(content, "content")
    # disp_img(style, "style")

    # tried this, but no big difference
    # style = cv2.cvtColor(style, cv2.COLOR_BGR2HSV)
    # content = cv2.cvtColor(content, cv2.COLOR_BGR2HSV)
    
    print("content size: ", content.shape)
    print("style size: ", style.shape)

    r,c,p = style.shape
    style = style.reshape(1,r,c,p)

    print("\n*******Begin Universal Style Transfer*******\n")

    # alpha = 0.9
    iters = 1
    relu = 5

    for i in range(iters):

        i = relu-i-1
        print("\nLevel ",i+1," : \n-----------")
        
        r,c,p = content.shape
        content = content.reshape(1,r,c,p)

        content_features = encoder[i].predict(content)
        style_features = encoder[i].predict(style)

        print("content_features: ", content_features.shape)
        print("style_features: ", style_features.shape)

        s,r,c,p = content_features.shape
        disp_cont_feat = content_features.reshape(r*c, p)

        s,r,c,p = style_features.shape
        disp_style_feat = style_features.reshape(r*c, p)

        # disp_img(disp_cont_feat, "cont_feat")
        # disp_img(disp_style_feat, "style_feat")

        blended = wct_np(content_features, style_features, alpha, eps=0)
        print("blended_features: ", blended.shape)


        out = decoder[i].predict(blended)
        s,r,c,p = out.shape
        content = out.reshape(r,c,p)

        if live_path == '1':
            disp_img(np.uint8(content), "frame")
        else:    
            disp_img(np.uint8(content), "Level_"+str(i))
        
        print("reconstructed_content_shape: ", content.shape)


    if live_path == '0':
        out_file = content_path[content_path.rfind('/'):content_path.find('.')]+"_"+style_path[style_path.rfind('/')+1:style_path.find('.')]+".png"
        # disp_img(np.uint8(content), "result")
        # out_file = content_path[:content_path.find('.')]+"_"+style_path[:style_path.find('.')]+"_"+"relu_"+str(relu)+"_alpha_"+str(alpha)+"_iters_"+str(iters)+".png"
        
        cv2.imwrite(out_path+"/"+out_file, np.uint8(content))



stylize(args.content_path, args.style_path, args.out_path, args.alpha, args.live_path, args.weight_path)
