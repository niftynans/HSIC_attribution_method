# -*- coding: utf-8 -*-

import scipy
import os
import numpy as np
import cv2
import tensorflow as tf
import subprocess
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
plt.style.use('seaborn')
import tensorflow_probability as tfp

import xplique
from xplique.attributions import *
from xplique.metrics import *
from xplique_addons import *
from utils import *
red_tr = get_alpha_cmap('Reds')

from xplique.attributions import grad_cam
from xplique.attributions import grad_cam_pp



batch_size = 256
images_classes = [
                  ('fox.png', 278),
                  ('leopard.png', 288),
                  ('polar_bear.png', 296),
                  ('snow_fox.png', 279),
]

X_raw = np.array([load_image(p) for p, y in images_classes])
Y_true = np.array([y for p, y in images_classes])

for m in range(4):
    if m == 0:
        model = tf.keras.applications.ResNet50V2()
        model.layers[-1].activation = tf.keras.activations.linear
        inputs =  tf.keras.applications.resnet_v2.preprocess_input(np.array([x.copy() for x in X_raw], copy=True))
        print(" Model Name: Resnet")

    elif m == 1:
        model = tf.keras.applications.efficientnet.EfficientNetB0()
        model.layers[-1].activation = tf.keras.activations.linear
        inputs =  tf.keras.applications.efficientnet.preprocess_input(np.array([x.copy() for x in X_raw], copy=True))
        print(" Model Name: Efficient-Net")

    elif m == 2:
        model = tf.keras.applications.mobilenet.MobileNet()
        model.layers[-1].activation = tf.keras.activations.linear
        inputs =  tf.keras.applications.mobilenet.preprocess_input(np.array([x.copy() for x in X_raw], copy=True))
        print(" Model Name: Mobile-Net")

    else:
        model = tf.keras.applications.efficientnet.EfficientNetB0()
        model.layers[-1].activation = tf.keras.activations.linear
        inputs =  tf.keras.applications.efficientnet.preprocess_input(np.array([x.copy() for x in X_raw], copy=True))
        print(" Model Name: VGG-16")

    labels = np.argmax(model.predict(inputs, batch_size=batch_size), axis=-1)
    labels_ohe = tf.one_hot(labels, 1000)

    grid_size = 7
    nb_forward = 1536

    for h in range(3):
        if h == 0:
            hsic_explainer = HsicAttributionMethod(model, 
                                                grid_size = grid_size, 
                                                nb_design = nb_forward , 
                                                sampler = HsicLHSSampler(binary=True), 
                                                estimator = HsicEstimator(kernel_type="binary"),
                                                perturbation_function = 'inpainting',
                                                batch_size = 256)

        elif h == 1:
            hsic_explainer = HsicAttributionMethod(model, 
                                                grid_size = grid_size, 
                                                nb_design = nb_forward , 
                                                sampler = HsicSobolSampler(binary=True), 
                                                estimator = HsicEstimator(kernel_type="binary"),
                                                perturbation_function = 'inpainting',
                                                batch_size = 256)
        else:
            hsic_explainer = HsicAttributionMethod(model, 
                                                grid_size = grid_size, 
                                                nb_design = nb_forward , 
                                                sampler = HsicHaltonSampler(binary=True), 
                                                estimator = HsicEstimator(kernel_type="binary"),
                                                perturbation_function = 'inpainting',
                                                batch_size = 256)


        explanations = hsic_explainer(inputs, labels_ohe)
        explanations = np.array(explanations)

        set_size(22, 12)
        # for i in range(4):
        #     plt.subplot(1, 4, i+1)
        #     show(inputs[i])
        #     show(explanations[i], cmap="jet", alpha=0.4)
        # plt.show()

        img_paths = [
            "ILSVRC2012_val_00000001.JPEG",
            "ILSVRC2012_val_00000002.JPEG",
            "ILSVRC2012_val_00000012.JPEG",
            "ILSVRC2012_val_00000030.JPEG"
        ]


        X_raw = np.array([load_image(p) for p in img_paths])

        if m == 0:
            inputs =  tf.keras.applications.resnet_v2.preprocess_input(np.array([x.copy() for x in X_raw], copy=True))
            labels = np.argmax(model.predict(inputs, batch_size=batch_size), axis=-1)
            labels_ohe = tf.one_hot(labels, 1000)

        elif m == 1:
            inputs =  tf.keras.applications.efficientnet.preprocess_input(np.array([x.copy() for x in X_raw], copy=True))
            labels = np.argmax(model.predict(inputs, batch_size=batch_size), axis=-1)
            labels_ohe = tf.one_hot(labels, 1000)

        elif m == 2:
            inputs =  tf.keras.applications.mobilenet.preprocess_input(np.array([x.copy() for x in X_raw], copy=True))
            labels = np.argmax(model.predict(inputs, batch_size=batch_size), axis=-1)
            labels_ohe = tf.one_hot(labels, 1000)

        else:
            model = tf.keras.applications.efficientnet.EfficientNetB0()
            model.layers[-1].activation = tf.keras.activations.linear
            inputs =  tf.keras.applications.efficientnet.preprocess_input(np.array([x.copy() for x in X_raw], copy=True))

        cherrypick = {
                0: 13, #Actual ImegeNet index: 1
                1: 6, #Actual ImegeNet index: 2
            2:24, #Actual ImegeNet index: 12
            3: 13 #Actual ImegeNet index: 30
        }

        grid_size = 6
        nb_forward  = 1024

        set_size(22, 12)
        perc = [20, 40, 40, 40]
        for k, im in enumerate(cherrypick.keys()):
            
            hsic_explainer = HsicAttributionMethod(model, 
                                                grid_size = grid_size, 
                                                nb_design = nb_forward, 
                                                sampler = HsicSampler(binary=True), 
                                                estimator = HsicEstimator(kernel_type="inter", base_inter=cherrypick[im]),
                                                perturbation_function = 'inpainting',
                                                batch_size = 256)

            explanations_inter =  hsic_explainer(inputs[im:im+1], labels_ohe[im:im+1])
            
            hsic_explainer = HsicAttributionMethod(model, 
                                            grid_size = grid_size, 
                                            nb_design = nb_forward , 
                                            sampler = HsicSampler(binary=True), 
                                            estimator = HsicEstimator(kernel_type="binary"),
                                            perturbation_function = 'inpainting',
                                            batch_size = 256)

            explanations_uni = hsic_explainer(inputs[im:im+1], labels_ohe[im:im+1])

            #plt.subplot(2, 4, k +1)
            #show(inputs[im])
            
            percentile = perc[k]
            heatmap = np.array(explanations_uni)
            #show(heatmap, cmap="jet", alpha=0.4)
            #if k == 0:
            #    plt.ylabel(r"\mathcal{S}_i")
            #plt.subplot(2, 4, k + 5)
            
            #show(inputs[im])
            heatmap = np.array(explanations_inter)
            t = np.percentile(heatmap.flatten(), 100 - percentile)
            heatmap = 0.0 + (heatmap > t) * heatmap
            #show(heatmap, cmap=red_tr, alpha=0.7)
            #if k == 0:
            #    plt.ylabel(r"\mathcal{S}_{i \times j")
                
        #plt.tight_layout()
        #plt.show()

        from xplique.metrics.stability import AverageStability
        avg_stab = AverageStability(model, inputs, labels_ohe, 64)
        print("### Average Stability ###")
        print(avg_stab.evaluate(hsic_explainer))

        if m == 0 and h == 0:
            np.save(os.path.join('files','explanations_resnet_LHS.npy'), explanations)
            np.save(os.path.join('files','inputs_resnet_LHS.npy'), inputs)
            np.save(os.path.join('files','labels_ohe_resnet_LHS.npy'), labels_ohe)
        
        elif m == 0 and h == 1:
            np.save(os.path.join('files','explanations_resnet_Sobol.npy'), explanations)
            np.save(os.path.join('files','inputs_resnet_Sobol.npy'), inputs)
            np.save(os.path.join('files','labels_ohe_resnet_Sobol.npy'), labels_ohe)
        
        elif m == 0 and h == 2:
            np.save(os.path.join('files','explanations_resnet_Halton.npy'), explanations)
            np.save(os.path.join('files','inputs_resnet_Halton.npy'), inputs)
            np.save(os.path.join('files','labels_ohe_resnet_Halton.npy'), labels_ohe)
        
        elif m == 1 and h == 0:
            np.save(os.path.join('files','explanations_efficientnet_LHS.npy'), explanations)
            np.save(os.path.join('files','inputs_efficientnet_LHS.npy'), inputs)
            np.save(os.path.join('files','labels_ohe_efficientnet_LHS.npy'), labels_ohe)
        
        elif m == 1 and h == 1:
            np.save(os.path.join('files','explanations_efficientnet_Sobol.npy'), explanations)
            np.save(os.path.join('files','inputs_efficientnet_Sobol.npy'), inputs)
            np.save(os.path.join('files','labels_ohe_efficientnet_Sobol.npy'), labels_ohe)

        elif m == 1 and h == 2:
            np.save(os.path.join('files','explanations_efficientnet_Halton.npy'), explanations)
            np.save(os.path.join('files','inputs_efficientnet_Halton.npy'), inputs)
            np.save(os.path.join('files','labels_ohe_efficientnet_Halton.npy'), labels_ohe)

        elif m == 2 and h == 0:
            np.save(os.path.join('files','explanations_mobilenet_LHS.npy'), explanations)
            np.save(os.path.join('files','inputs_mobilenet_LHS.npy'), inputs)
            np.save(os.path.join('files','labels_ohe_mobilenet_LHS.npy'), labels_ohe)
        
        elif m == 2 and h == 1:
            np.save(os.path.join('files','explanations_mobilenet_Sobol.npy'), explanations)
            np.save(os.path.join('files','inputs_mobilenet_Sobol.npy'), inputs)
            np.save(os.path.join('files','labels_ohe_mobilenet_Sobol.npy'), labels_ohe)
        
        elif m == 2 and h == 2:
            np.save(os.path.join('files','explanations_mobilenet_Halton.npy'), explanations)
            np.save(os.path.join('files','inputs_mobilenet_Halton.npy'), inputs)
            np.save(os.path.join('files','labels_ohe_mobilenet_Halton.npy'), labels_ohe)
        
        elif m == 3 and h == 0:
            np.save(os.path.join('files','explanations_vgg-16_LHS.npy'), explanations)
            np.save(os.path.join('files','inputs_vgg-16_LHS.npy'), inputs)
            np.save(os.path.join('files','labels_ohe_vgg-16_LHS.npy'), labels_ohe)
        
        elif m == 3 and h == 1:
            np.save(os.path.join('files','explanations_vgg-16_Sobol.npy'), explanations)
            np.save(os.path.join('files','inputs_vgg-16_Sobol.npy'), inputs)
            np.save(os.path.join('files','labels_ohe_vgg-16_Sobol.npy'), labels_ohe)
        
        else:
            np.save(os.path.join('files','explanations_vgg-16_Halton.npy'), explanations)
            np.save(os.path.join('files','inputs_vgg-16_Halton.npy'), inputs)
            np.save(os.path.join('files','labels_ohe_vgg-16_Halton.npy'), labels_ohe)

    if m == 0:
        print(" ### RESNET BASE RESULTS ###")

        print("GradCAM model")
        gradcam_explainer = GradCAM(model = model)
        explanations = gradcam_explainer(inputs, labels_ohe)
        explanations = np.array(explanations)

        for k, im in enumerate(cherrypick.keys()):

            gradcam_explainer = GradCAM(model = model)
            explanations_inter =  gradcam_explainer(inputs[im:im+1], labels_ohe[im:im+1])
            
            gradcam_explainer = GradCAM(model = model)
            explanations_uni = gradcam_explainer(inputs[im:im+1], labels_ohe[im:im+1])

            #plt.subplot(2, 4, k +1)
            #show(inputs[im])
            
            percentile = perc[k]
            heatmap = np.array(explanations_uni)
            #show(heatmap, cmap="jet", alpha=0.4)
            #if k == 0:
            #    plt.ylabel(r"\mathcal{S}_i")
            #plt.subplot(2, 4, k + 5)
            
            #show(inputs[im])
            heatmap = np.array(explanations_inter)
            t = np.percentile(heatmap.flatten(), 100 - percentile)
            heatmap = 0.0 + (heatmap > t) * heatmap
            #show(heatmap, cmap=red_tr, alpha=0.7)
            #if k == 0:
            #    plt.ylabel(r"\mathcal{S}_{i \times j")
                
        #plt.tight_layout()
        #plt.show()

        avg_stab = AverageStability(model, inputs, labels_ohe, 64)
        print("### Average Stability ###")
        print(avg_stab.evaluate(gradcam_explainer))

        np.save(os.path.join('files','explanations_Resnet_Gradcam.npy'), explanations)
        np.save(os.path.join('files','inputs_Resnet_Gradcam.npy'), inputs)
        np.save(os.path.join('files','labels_ohe_Resnet_Gradcam.npy'), labels_ohe)


        print("GradCAM++ Model ")
        gradcampp_explainer = GradCAMPP(model = model)
        explanations = gradcampp_explainer(inputs, labels_ohe)
        explanations = np.array(explanations)

        for k, im in enumerate(cherrypick.keys()):
            
            gradcampp_explainer = GradCAMPP(model = model)
            explanations_inter =  gradcam_explainer(inputs[im:im+1], labels_ohe[im:im+1])
            
            gradcampp_explainer = GradCAMPP(model = model)
            explanations_uni = gradcampp_explainer(inputs[im:im+1], labels_ohe[im:im+1])

            #plt.subplot(2, 4, k +1)
            #show(inputs[im])
            
            percentile = perc[k]
            heatmap = np.array(explanations_uni)
            #show(heatmap, cmap="jet", alpha=0.4)
            #if k == 0:
            #    plt.ylabel(r"\mathcal{S}_i")
            #plt.subplot(2, 4, k + 5)
            
            #show(inputs[im])
            heatmap = np.array(explanations_inter)
            t = np.percentile(heatmap.flatten(), 100 - percentile)
            heatmap = 0.0 + (heatmap > t) * heatmap
            #show(heatmap, cmap=red_tr, alpha=0.7)
            #if k == 0:
            #    plt.ylabel(r"\mathcal{S}_{i \times j")
                
        #plt.tight_layout()
        #plt.show()

        avg_stab = AverageStability(model, inputs, labels_ohe, 64)
        print("### Average Stability ###")
        print(avg_stab.evaluate(gradcampp_explainer))

        np.save(os.path.join('files','explanations_Resnet_Gradcam++.npy'), explanations)
        np.save(os.path.join('files','inputs_Resnet_Gradcam++.npy'), inputs)
        np.save(os.path.join('files','labels_ohe_Resnet_Gradcam++.npy'), labels_ohe)
                
    elif m == 1:

      print(" ### EFFICIENT NET BASE RESULTS ###")

      print("GradCAM model")
      gradcam_explainer = GradCAM(model = model)
      explanations = gradcam_explainer(inputs, labels_ohe)
      explanations = np.array(explanations)

      for k, im in enumerate(cherrypick.keys()):

          gradcam_explainer = GradCAM(model = model)
          explanations_inter =  gradcam_explainer(inputs[im:im+1], labels_ohe[im:im+1])
          
          gradcam_explainer = GradCAM(model = model)
          explanations_uni = gradcam_explainer(inputs[im:im+1], labels_ohe[im:im+1])

          #plt.subplot(2, 4, k +1)
          #show(inputs[im])
          
          percentile = perc[k]
          heatmap = np.array(explanations_uni)
          #show(heatmap, cmap="jet", alpha=0.4)
          #if k == 0:
          #    plt.ylabel(r"\mathcal{S}_i")
          #plt.subplot(2, 4, k + 5)
          
          #show(inputs[im])
          heatmap = np.array(explanations_inter)
          t = np.percentile(heatmap.flatten(), 100 - percentile)
          heatmap = 0.0 + (heatmap > t) * heatmap
          #show(heatmap, cmap=red_tr, alpha=0.7)
          #if k == 0:
          #    plt.ylabel(r"\mathcal{S}_{i \times j")
              
      #plt.tight_layout()
      #plt.show()

      avg_stab = AverageStability(model, inputs, labels_ohe, 64)
      print("### Average Stability ###")
      print(avg_stab.evaluate(gradcam_explainer))

      np.save(os.path.join('files','explanations_efficientnet_Gradcam.npy'), explanations)
      np.save(os.path.join('files','inputs_efficientnet_Gradcam.npy'), inputs)
      np.save(os.path.join('files','labels_ohe_efficientnet_Gradcam.npy'), labels_ohe)


      print("GradCAM++ Model ")
      gradcampp_explainer = GradCAMPP(model = model)
      explanations = gradcampp_explainer(inputs, labels_ohe)
      explanations = np.array(explanations)

      for k, im in enumerate(cherrypick.keys()):
          
          gradcampp_explainer = GradCAMPP(model = model)
          explanations_inter =  gradcam_explainer(inputs[im:im+1], labels_ohe[im:im+1])
          
          gradcampp_explainer = GradCAMPP(model = model)
          explanations_uni = gradcampp_explainer(inputs[im:im+1], labels_ohe[im:im+1])

          #plt.subplot(2, 4, k +1)
          #show(inputs[im])
          
          percentile = perc[k]
          heatmap = np.array(explanations_uni)
          #show(heatmap, cmap="jet", alpha=0.4)
          #if k == 0:
          #    plt.ylabel(r"\mathcal{S}_i")
          #plt.subplot(2, 4, k + 5)
          
          #show(inputs[im])
          heatmap = np.array(explanations_inter)
          t = np.percentile(heatmap.flatten(), 100 - percentile)
          heatmap = 0.0 + (heatmap > t) * heatmap
          #show(heatmap, cmap=red_tr, alpha=0.7)
          #if k == 0:
          #    plt.ylabel(r"\mathcal{S}_{i \times j")
              
      #plt.tight_layout()
      #plt.show()

      avg_stab = AverageStability(model, inputs, labels_ohe, 64)
      print("### Average Stability ###")
      print(avg_stab.evaluate(gradcampp_explainer))

      np.save(os.path.join('files','explanations_efficientnet_Gradcam++.npy'), explanations)
      np.save(os.path.join('files','inputs_efficientnet_Gradcam++.npy'), inputs)
      np.save(os.path.join('files','labels_ohe_efficientnet_Gradcam++.npy'), labels_ohe)
      

                
    elif m == 2:
      print(" ### MOBILE NET BASE RESULTS ###")

      print("GradCAM model")
      gradcam_explainer = GradCAM(model = model)
      explanations = gradcam_explainer(inputs, labels_ohe)
      explanations = np.array(explanations)

      for k, im in enumerate(cherrypick.keys()):

          gradcam_explainer = GradCAM(model = model)
          explanations_inter =  gradcam_explainer(inputs[im:im+1], labels_ohe[im:im+1])
          
          gradcam_explainer = GradCAM(model = model)
          explanations_uni = gradcam_explainer(inputs[im:im+1], labels_ohe[im:im+1])

          #plt.subplot(2, 4, k +1)
          #show(inputs[im])
          
          percentile = perc[k]
          heatmap = np.array(explanations_uni)
          #show(heatmap, cmap="jet", alpha=0.4)
          #if k == 0:
          #    plt.ylabel(r"\mathcal{S}_i")
          #plt.subplot(2, 4, k + 5)
          
          #show(inputs[im])
          heatmap = np.array(explanations_inter)
          t = np.percentile(heatmap.flatten(), 100 - percentile)
          heatmap = 0.0 + (heatmap > t) * heatmap
          #show(heatmap, cmap=red_tr, alpha=0.7)
          #if k == 0:
          #    plt.ylabel(r"\mathcal{S}_{i \times j")
              
      #plt.tight_layout()
      #plt.show()

      avg_stab = AverageStability(model, inputs, labels_ohe, 64)
      print("### Average Stability ###")
      print(avg_stab.evaluate(gradcam_explainer))

      np.save(os.path.join('files','explanations_mobilenet_Gradcam.npy'), explanations)
      np.save(os.path.join('files','inputs_mobilenet_Gradcam.npy'), inputs)
      np.save(os.path.join('files','labels_ohe_mobilenet_Gradcam.npy'), labels_ohe)


      print("GradCAM++ Model ")
      gradcampp_explainer = GradCAMPP(model = model)
      explanations = gradcampp_explainer(inputs, labels_ohe)
      explanations = np.array(explanations)

      for k, im in enumerate(cherrypick.keys()):
          
          gradcampp_explainer = GradCAMPP(model = model)
          explanations_inter =  gradcam_explainer(inputs[im:im+1], labels_ohe[im:im+1])
          
          gradcampp_explainer = GradCAMPP(model = model)
          explanations_uni = gradcampp_explainer(inputs[im:im+1], labels_ohe[im:im+1])

          #plt.subplot(2, 4, k +1)
          #show(inputs[im])
          
          percentile = perc[k]
          heatmap = np.array(explanations_uni)
          #show(heatmap, cmap="jet", alpha=0.4)
          #if k == 0:
          #    plt.ylabel(r"\mathcal{S}_i")
          #plt.subplot(2, 4, k + 5)
          
          #show(inputs[im])
          heatmap = np.array(explanations_inter)
          t = np.percentile(heatmap.flatten(), 100 - percentile)
          heatmap = 0.0 + (heatmap > t) * heatmap
          #show(heatmap, cmap=red_tr, alpha=0.7)
          #if k == 0:
          #    plt.ylabel(r"\mathcal{S}_{i \times j")
              
      #plt.tight_layout()
      #plt.show()

      avg_stab = AverageStability(model, inputs, labels_ohe, 64)
      print("### Average Stability ###")
      print(avg_stab.evaluate(gradcampp_explainer))

      np.save(os.path.join('files','explanations_mobilenet_Gradcam++.npy'), explanations)
      np.save(os.path.join('files','inputs_mobilenet_Gradcam++.npy'), inputs)
      np.save(os.path.join('files','labels_ohe_mobilenet_Gradcam++.npy'), labels_ohe)


          
    else:
        print(" ### VGG-16 BASE RESULTS ###")

        print("GradCAM model")
        gradcam_explainer = GradCAM(model = model)
        explanations = gradcam_explainer(inputs, labels_ohe)
        explanations = np.array(explanations)

        for k, im in enumerate(cherrypick.keys()):

            gradcam_explainer = GradCAM(model = model)
            explanations_inter =  gradcam_explainer(inputs[im:im+1], labels_ohe[im:im+1])
            
            gradcam_explainer = GradCAM(model = model)
            explanations_uni = gradcam_explainer(inputs[im:im+1], labels_ohe[im:im+1])

            #plt.subplot(2, 4, k +1)
            #show(inputs[im])
            
            percentile = perc[k]
            heatmap = np.array(explanations_uni)
            #show(heatmap, cmap="jet", alpha=0.4)
            #if k == 0:
            #    plt.ylabel(r"\mathcal{S}_i")
            #plt.subplot(2, 4, k + 5)
            
            #show(inputs[im])
            heatmap = np.array(explanations_inter)
            t = np.percentile(heatmap.flatten(), 100 - percentile)
            heatmap = 0.0 + (heatmap > t) * heatmap
            #show(heatmap, cmap=red_tr, alpha=0.7)
            #if k == 0:
            #    plt.ylabel(r"\mathcal{S}_{i \times j")
                
        #plt.tight_layout()
        #plt.show()

        avg_stab = AverageStability(model, inputs, labels_ohe, 64)
        print("### Average Stability ###")
        print(avg_stab.evaluate(gradcam_explainer))

        np.save(os.path.join('files','explanations_vgg-16_Gradcam.npy'), explanations)
        np.save(os.path.join('files','inputs_vgg-16_Gradcam.npy'), inputs)
        np.save(os.path.join('files','labels_ohe_vgg-16_Gradcam.npy'), labels_ohe)


        print("GradCAM++ Model ")
        gradcampp_explainer = GradCAMPP(model = model)
        explanations = gradcampp_explainer(inputs, labels_ohe)
        explanations = np.array(explanations)

        for k, im in enumerate(cherrypick.keys()):
            
            gradcampp_explainer = GradCAMPP(model = model)
            explanations_inter =  gradcam_explainer(inputs[im:im+1], labels_ohe[im:im+1])
            
            gradcampp_explainer = GradCAMPP(model = model)
            explanations_uni = gradcampp_explainer(inputs[im:im+1], labels_ohe[im:im+1])

            #plt.subplot(2, 4, k +1)
            #show(inputs[im])
            
            percentile = perc[k]
            heatmap = np.array(explanations_uni)
            #show(heatmap, cmap="jet", alpha=0.4)
            #if k == 0:
            #    plt.ylabel(r"\mathcal{S}_i")
            #plt.subplot(2, 4, k + 5)
            
            #show(inputs[im])
            heatmap = np.array(explanations_inter)
            t = np.percentile(heatmap.flatten(), 100 - percentile)
            heatmap = 0.0 + (heatmap > t) * heatmap
            #show(heatmap, cmap=red_tr, alpha=0.7)
            #if k == 0:
            #    plt.ylabel(r"\mathcal{S}_{i \times j")
                
        #plt.tight_layout()
        #plt.show()

        avg_stab = AverageStability(model, inputs, labels_ohe, 64)
        print("### Average Stability ###")
        print(avg_stab.evaluate(gradcampp_explainer))

        np.save(os.path.join('files','explanations_vgg-16_Gradcam++.npy'), explanations)
        np.save(os.path.join('files','inputs_vgg-16_Gradcam++.npy'), inputs)
        np.save(os.path.join('files','labels_ohe_vgg-16_Gradcam++.npy'), labels_ohe)
            

                    