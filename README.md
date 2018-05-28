# Joint Retinal Optical Disc and Fovea Detection

Here you can find the implementation of a new strategy for the task of simultaneously locating the optic disc and the fovea in eye fundus images. 

In contrast with previous techniques, the proposed method does not attempt to directly detect only OD and fovea centers. Instead, the distance to both locations is regressed for 
every pixel in a retinal image. This regression problem can be solved by means of a Fully-Convolutional Neural Network. This strategy poses a multi-task-like problem, on which information
of every pixel contributes to generate a globally consistent prediction map where likelihood of OD and fovea locations are maximized.

In particular, we make use of the a U-net architecture, while using a loss function suitable to perform distance regression (L2 Loss).

![](images/unet_fod.png)

Training 
--------

The method was trained and validated on the Messidor dataset. If you wish to replicate, the first step is to exclude the Messidor images that do not contain OD and Fovea 
location information , as provided by [1].
\\
Split the remaining 1136 images in two (*half 1* and *half 2*)  and train your model in two different splits:
* **Split 1** will use *half 1*  for training and *half 2* for testing;
* **Split 2** will use *half 2* for training and *half 1* for testing.

Run the *script-train-messidor.py* file, specifying which split is being used and the directory where to save the model.

Evaluating
----------

To get the resulting predictions on Messidor, use the *predict_od_fov.py* file, specifying which split is being used and model directory.

Running for single image
------------------------

You can run the *demo.py* script, which takes a retinal image and returns the location of the OD and Fovea. When possible, the images should be cropped around the FOV, 
or a mask of the FOV provided. (This script saves an image to a results/ folder, ensure there is one in your directory).

    python demo.py --img_dir images/messidor_test.tif --mask_dir images/messidor_test_mask.tif

![](images/messidor_test_prediction.png)

---------------------------------- 

If you wish to use or reference our work, please cite us using:

    @article{meyer2018odfovea,
    title={A Pixel-wise Distance Regression Approach for Joint Retinal Optical Disc and Fovea Detection},
    author={Meyer, Maria Ines and Galdran, Adrian and Mendonça, Ana Maria and Campilho, Aurélio},
    journal= {},
    year={2018}
    }
    

--------
    
### References
1.  Gegundez-Arias, M.E., Marin, D., Bravo, J.M., Suero, A.: Locating the fovea
center position in digital fundus images using thresholding and feature extraction 
techniques. Computerized Medical Imaging and Graphics 37, 386–393 (2013)

    
