
import collections
import torchvision
import torch
import torchvision.transforms.functional as F
import random 
import numbers
import numpy as np
from PIL import Image
import cv2
import os

################### functions for blurring
def blurwithkernel(s_img, k_img):
    """Inputs are two rgb PIL images - and the output should also be PIL image of same size """
    s_img = np.array(s_img)/255
    k_img = np.array(k_img)
    filtered = cv2.filter2D(src=s_img, kernel=k_img/np.sum(np.sum(k_img[:,:,0])), ddepth=-1)
    imout = Image.fromarray((filtered*255).astype(np.uint8))
#    imout.save('blurred_val.jpg')
    return imout

def objectselectivemotionblur(sharp,objectsegmap,kernel):
    '''Inputs 3 PIL images and outputs a single PIL image of same size'''
    blurred_fgmap = blurwithkernel(objectsegmap,kernel)
    inverse_blurred_fgmap = 1 - np.array(blurred_fgmap)/255
    inverse_blurred_fgmap=inverse_blurred_fgmap.reshape(inverse_blurred_fgmap.shape[0],inverse_blurred_fgmap.shape[1],1)
    background_img = inverse_blurred_fgmap*np.array(sharp)
    
    objectsegmap = np.array(objectsegmap)
    objectsegmap = objectsegmap.reshape(objectsegmap.shape[0],objectsegmap.shape[1],1)
    foreground_img = Image.fromarray((objectsegmap/255*np.array(sharp)).astype(np.uint8))
    blurred_foreground_img = blurwithkernel(foreground_img,kernel)

    selectiveblurred_img = Image.fromarray(((np.array(blurred_foreground_img)+np.array(background_img))).astype(np.uint8))
    return selectiveblurred_img





# Transform for class centric motion blur ###################################################
#
class ExtCCMBA(object):
    """Blur the image using a randomly sampled kernel from total 3level kernels.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    def __init__(self, p=0.5, kerneldirectory='/data1/user_data/aakanksha/ZZZ_datasets/blurRelated/VOCdevkit/VOC2012/blur_kernels_levelwise/'):
        self.kerneldirectory = kerneldirectory
        blurlevels = [1,2,3]
        self.kernelpathslist = []
        subdirlist = os.listdir(self.kerneldirectory)
        for subdir in subdirlist:
            if int(subdir.split('_')[-1]) in blurlevels:
                path_directory = os.path.join(self.kerneldirectory,subdir)
                pathfilenames = os.listdir(path_directory)
                for pathfilename in pathfilenames:
                    filepath = os.path.join(path_directory,pathfilename)
                    self.kernelpathslist.append(filepath)
        random.shuffle(self.kernelpathslist)
        self.p = p
    def __call__(self, img, lbl):
        """
        Args:
            img (PIL Image): Sharp Image to be Selectively Blurred.

        Returns:
            PIL Image: Selectively Blur one class.
        """
        outlist = []
        if random.random() < self.p:
            #img = imglist[0]
            #select a random blur kernel and read it
            kernelid = random.randint(0, len(self.kernelpathslist)-1)
            kernel = Image.open(self.kernelpathslist[kernelid])

            #segmap foreground filtering - select one of the n classes or the background - total n+1 classes
            segmap_foreground = np.array(lbl)+1
            ClassesInImage = list(np.unique(lbl))
            #print(ClassesInImage)
            boundary = 255
            if boundary in ClassesInImage: ClassesInImage.remove(boundary)
            random.shuffle(ClassesInImage)
            if len(ClassesInImage)==0:
                return img,lbl

            nclass2consider = random.randint(1,len(ClassesInImage)) #atleast 1 class blurred, upto all classes blurred
            # print('number of classes being considered for blurring',nclass2consider)
            if len(ClassesInImage)==0:
                return img,lbl
            classesconsidered = ClassesInImage[0:nclass2consider]

            #getting the foreground mask as PIL Image
            segmap_out = np.zeros((segmap_foreground.shape[0],segmap_foreground.shape[1], nclass2consider))
            for i,classconsidered in enumerate(classesconsidered):
                segmap_out_slice = np.array(lbl)+1
                segmap_out_slice[segmap_foreground!=(classconsidered+1)]=0
                segmap_out_slice[segmap_foreground==(classconsidered+1)]=255
                segmap_out[:,:,i] = segmap_out_slice
            segmap_outfinal = np.amax(segmap_out, axis=2)
            objectsegmap = Image.fromarray(segmap_outfinal.astype(np.uint8)) 

            #objectselective motion blurring
            out = objectselectivemotionblur(img,objectsegmap,kernel)
            outlist.append(out)
            return outlist[0],lbl
        else:
            return img,lbl

#
#  Extended Transforms for Semantic Segmentation
#
class ExtRandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imglist, lbl):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        imglist2 = []
        if random.random() < self.p:
            for img in imglist:
                imglist2.append(F.hflip(img))
            return  imglist2, F.hflip(lbl)
        return imglist, lbl

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)



class ExtCompose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, imglist, lbl):
        for t in self.transforms:
            imglist, lbl = t(imglist, lbl)
        return imglist, lbl

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ExtCenterCrop(object):
    """Crops the given PIL Image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, imglist, lbl):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        imglist2 =[]
        for img in imglist:
            imglist2.append(F.center_crop(img, self.size))
        return imglist2, F.center_crop(lbl, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class ExtRandomScale(object):
    def __init__(self, scale_range, interpolation=Image.BILINEAR):
        self.scale_range = scale_range
        self.interpolation = interpolation

    def __call__(self,  img,  lbl):
        """
        Args:
            img (PIL Image): Image to be scaled.
            lbl (PIL Image): Label to be scaled.
        Returns:
            PIL Image: Rescaled image.
            PIL Image: Rescaled label.
        """
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        assert img.size == lbl.size
        target_size = ( int(img.size[1]*scale), int(img.size[0]*scale) )
        img2 = F.resize(img, target_size, self.interpolation)
        return img2, F.resize(lbl, target_size, Image.NEAREST)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)

class ExtScale(object):
    """Resize the input PIL Image to the given scale.
    Args:
        Scale (sequence or int): scale factors
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, scale, interpolation=Image.BILINEAR):
        self.scale = scale
        self.interpolation = interpolation

    def __call__(self,  imglist,  lbl):
        """
        Args:
            img (PIL Image): Image to be scaled.
            lbl (PIL Image): Label to be scaled.
        Returns:
            PIL Image: Rescaled image.
            PIL Image: Rescaled label.
        """
        assert img.size == lbl.size
        imglist2 =[]
        for img in imglist:
            target_size = ( int(img.size[1]*self.scale), int(img.size[0]*self.scale) ) # (H, W)
            imglist2.append(F.resize(img, target_size, self.interpolation))

        return imglist2, F.resize(lbl, target_size, Image.NEAREST)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


class ExtRandomRotation(object):
    """Rotate the image by angle.
    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter.
            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    """

    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self,  imglist,  lbl):
        """
            img (PIL Image): Image to be rotated.
            lbl (PIL Image): Label to be rotated.
        Returns:
            PIL Image: Rotated image.
            PIL Image: Rotated label.
        """

        angle = self.get_params(self.degrees)
        imglist2 = []
        for img in imglist:
            imglist2.append(F.rotate(img, angle, self.resample, self.expand, self.center))
        return imglist2, F.rotate(lbl, angle, self.resample, self.expand, self.center)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string

class ExtRandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self,  img,  lbl):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """

        if random.random() < self.p:
            img2=F.hflip(img)
            return img2, F.hflip(lbl)
        return img, lbl

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class ExtRandomVerticalFlip(object):
    """Vertically flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self,  imglist,  lbl):
        """
        Args:
            img (PIL Image): Image to be flipped.
            lbl (PIL Image): Label to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
            PIL Image: Randomly flipped label.
        """
        if random.random() < self.p:
            imglist2 = []
            for img in imglist:
                imglist2.append(F.vflip(img))
            return imglist2, F.vflip(lbl)
            # return F.vflip(img), F.vflip(lbl)
        return imglist, lbl

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)



class ExtPad(object):
    def __init__(self, diviser=32):
        self.diviser = diviser
    
    def __call__(self,  imglist,  lbl):
        h, w = imglist[0].size
        ph = (h//32+1)*32 - h if h%32!=0 else 0
        pw = (w//32+1)*32 - w if w%32!=0 else 0

        imglist2 = []
        for img in imglist:
            imglist2.append(F.pad(img, ( pw//2, pw-pw//2, ph//2, ph-ph//2) ))
        lbl = F.pad(lbl, ( pw//2, pw-pw//2, ph//2, ph-ph//2))
        return imlist2, lbl

class ExtToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """
    def __init__(self, normalize=True, target_type='uint8'):
        self.normalize = normalize
        self.target_type = target_type
    def __call__(self,  pic,  lbl):
        """
        Note that labels will not be normalized to [0, 1].
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
            lbl (PIL Image or numpy.ndarray): Label to be converted to tensor. 
        Returns:
            Tensor: Converted image and label
        """
        if self.normalize:
            pic2 = F.to_tensor(pic)
            return pic2, torch.from_numpy( np.array( lbl, dtype=self.target_type) )
        else:
            pic2 = torch.from_numpy( np.array( pic, dtype=np.float32).transpose(2, 0, 1) )
            return pic2, torch.from_numpy( np.array( lbl, dtype=self.target_type) )

    def __repr__(self):
        return self.__class__.__name__ + '()'

class ExtNormalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor, lbl):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
            tensor (Tensor): Tensor of label. A dummy input for ExtCompose
        Returns:
            Tensor: Normalized Tensor image.
            Tensor: Unchanged Tensor label
        """
        tensor2 = F.normalize(tensor, self.mean, self.std)
        return tensor2, lbl

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class ExtRandomCrop(object):
    """Crop the given PIL Image at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception.
    """

    def __init__(self, size, padding=0, pad_if_needed=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img, lbl):
        """
        Args:
            img (PIL Image): Image to be cropped.
            lbl (PIL Image): Label to be cropped.
        Returns:
            PIL Image: Cropped image.
            PIL Image: Cropped label.
        """
        #print(imglist.size, lbl.size)
        assert img.size == lbl.size, 'size of img and lbl should be the same. %s, %s'%(img.size, lbl.size)
        if self.padding > 0:
            img = F.pad(img, self.padding)
            lbl = F.pad(lbl, self.padding)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, padding=int((1 + self.size[1] - img.size[0]) / 2))
            lbl = F.pad(lbl, padding=int((1 + self.size[1] - lbl.size[0]) / 2))

        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, padding=int((1 + self.size[0] - img.size[1]) / 2))
            lbl = F.pad(lbl, padding=int((1 + self.size[0] - lbl.size[1]) / 2))

        i, j, h, w = self.get_params(img, self.size)

        img2 = F.crop(img, i,j, h, w)

        return img2, F.crop(lbl, i, j, h, w)



class ExtResize(object):
    """Resize the input PIL Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, imglist, lbl):
        """
        Args:
            img (PIL Image): Image to be scaled.
        Returns:
            PIL Image: Rescaled image.
        """
        imglist2 =[]
        for img in imglist:
            imglist2.append(F.resize(img, self.size, self.interpolation))
        return imglist2, F.resize(lbl, self.size, Image.NEAREST)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str) 
    
class ExtColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, imglist, lbl):
        """
        Args:
            img (PIL Image): Input image.

        Returns:
            PIL Image: Color jittered image.
        """
        imglist2 = []
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        for im in imglist:
            imglist2.append(transform(im))

        return imglist2, lbl

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string

class Lambda(object):
    """Apply a user-defined lambda as a transform.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string




#### F****D UPP!!!! ####### 
# class ExtRandomCrop(object):
#     """Crop the given PIL Image at a random location.
#     Args:
#         size (sequence or int): Desired output size of the crop. If size is an
#             int instead of sequence like (h, w), a square crop (size, size) is
#             made.
#         padding (int or sequence, optional): Optional padding on each border
#             of the image. Default is 0, i.e no padding. If a sequence of length
#             4 is provided, it is used to pad left, top, right, bottom borders
#             respectively.
#         pad_if_needed (boolean): It will pad the image if smaller than the
#             desired size to avoid raising an exception.
#     """

#     def __init__(self, size, padding=0, pad_if_needed=True):
#         if isinstance(size, numbers.Number):
#             self.size = (int(size), int(size))
#         else:
#             self.size = size
#         self.padding = padding
#         self.pad_if_needed = pad_if_needed

#     @staticmethod
#     def get_params(img, output_size):
#         """Get parameters for ``crop`` for a random crop.
#         Args:
#             img (PIL Image): Image to be cropped.
#             output_size (tuple): Expected output size of the crop.
#         Returns:
#             tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
#         """
#         w, h = img.size
#         print('imgsize', w,h)
#         th, tw = output_size
#         print('patchsize', w,h)

#         if w == tw and h == th:
#             return 0, 0, h, w

#         i = random.randint(0, h - th)
#         j = random.randint(0, w - tw)
#         return i, j, th, tw

#     def __call__(self, imglist, lbl):
#         """
#         Args:
#             img (PIL Image): Image to be cropped.
#             lbl (PIL Image): Label to be cropped.
#         Returns:
#             PIL Image: Cropped image.
#             PIL Image: Cropped label.
#         """
#         for img in imglist:
#             # img= imglist[0]
#             assert img.size == lbl.size, 'size of img and lbl should be the same. %s, %s'%(img.size, lbl.size)
#         print('1')
#         print(imglist[0].size, self.size)
#         print(imglist[1].size, self.size)

#         # imglist2 = []
#         if self.padding > 0:
#             # imglist2=[]
#             for z,img in enumerate(imglist):
#                 imglist[z]=F.pad(img, self.padding)
#             # imglist = imglist2
#             lbl = F.pad(lbl, self.padding)
#         print('2')
#         print(imglist[0].size, self.size)
#         print(imglist[1].size, self.size)
#         # pad the width if needed
#         if self.pad_if_needed and img.size[0] < self.size[1]:
#             for z,img in enumerate(imglist):
#                 print('inside 1')
#                 # assert img.size == lbl.size, 'size of img and lbl should be the same. %s, %s'%(img.size, lbl.size)
#                 imglist[z]=F.pad(img, padding=int((1 + self.size[1] - img.size[0]) / 2))
#             # img = F.pad(img, padding=int((1 + self.size[1] - img.size[0]) / 2))
#             # imglist = imglist2
#             lbl = F.pad(lbl, padding=int((1 + self.size[1] - lbl.size[0]) / 2))
#         print('3')
#         print(imglist[0].size, self.size)
#         print(imglist[1].size, self.size)
#         # pad the height if needed
#         if self.pad_if_needed and img.size[1] < self.size[0]:
#             # imglist2=[]
#             print('inside 2')
#             for z,img in enumerate(imglist):
#                 # assert img.size == lbl.size, 'size of img and lbl should be the same. %s, %s'%(img.size, lbl.size)
#                 imglist[z] = F.pad(img, padding=int((1 + self.size[0] - img.size[1]) / 2))
#             # img = F.pad(img, padding=int((1 + self.size[0] - img.size[1]) / 2))
#             # imglist = imglist2
#             lbl = F.pad(lbl, padding=int((1 + self.size[0] - lbl.size[1]) / 2))

#         imglist3=[]
#         # if len(imglist2) == 0:
#             # imglist2 = imglist 
#         i, j, h, w = self.get_params(imglist[0], self.size)
#         for img in imglist:
#             imglist3.append(F.crop(img, i, j, h, w))

#         return imglist3, F.crop(lbl, i, j, h, w)

#     def __repr__(self):
#         return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)
