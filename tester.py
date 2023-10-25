import ext_transforms as et
from torch.utils import data
from dataset_voc import VOCSegmentation


train_transform = et.ExtCompose([
                et.ExtRandomScale((0.5, 2.0)),
                et.ExtRandomCrop(size=(256,256), pad_if_needed=True),
                et.ExtCCMBA(kerneldirectory='/data1/user_data/aakanksha/ZZZ_datasets/blurRelated/VOCdevkit/VOC2012/blur_kernels_levelwise/'),
                et.ExtRandomHorizontalFlip(),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])

train_dst = VOCSegmentation(root='./voc/', year='2012' ,image_set='train',download=True, transform=train_transform)
train_loader = data.DataLoader(train_dst, batch_size=4, shuffle=True, num_workers=2,drop_last=True) 

for image, label in train_loader:
    print(image.shape,label.shape)
