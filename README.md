## Improving Robustness of Semantic Segmentation to Motion-Blur using Class-Centric Augmentation (CVPR 2023)
[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Aakanksha_Improving_Robustness_of_Semantic_Segmentation_to_Motion-Blur_Using_Class-Centric_Augmentation_CVPR_2023_paper.pdf) |  [Supp-PDF](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Aakanksha_Improving_Robustness_of_CVPR_2023_supplemental.pdf) | [Bibtex](https://github.com/aka-discover/CCMBA_CVPR23/#citation)


## Code
The code for CCMBA has now been released!
The Pytorch code for the custom transform can be found in ext_transforms.py and sample code for use has been provided in tester.py for the VOC dataset. 
Note that the motion blur kernels need to be generated and stored in the following directory structure for use. 

![directoryStruct](https://github.com/aka-discover/CCMBA_CVPR23/assets/96165929/7d47fd87-a009-4ba2-872d-7835774cde2a)


## Citation

If you find this work useful, please consider citing it.

<pre><code>@InProceedings{Aakanksha_2023_CVPR,
    author    = {Aakanksha and Rajagopalan, A. N.},
    title     = {Improving Robustness of Semantic Segmentation to Motion-Blur Using Class-Centric Augmentation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {10470-10479}
}
</code></pre>

### License
This project is released under the [GNU General Public License v3.0](https://github.com/Mukosame/Zooming-Slow-Mo-CVPR-2020/blob/master/LICENSE).
