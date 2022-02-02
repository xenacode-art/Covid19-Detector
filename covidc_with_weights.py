# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 17:03:06 2020

@author: @xenacod-art
"""

import numpy as np
from sklearn import preprocessing
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input

def extract_transfer_learning_features(img_path):
    model = DenseNet121(weights='imagenet', include_top=False)
    img = image.load_img(img_path, target_size=(331, 331))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x).flatten()
    return features
def standarized_normalized_severity(features, norm='yes'):
    features=(features-np.load('trained_models/mean_covid_vs_non.npy'))/(np.load('trained_models/std_covid_vs_non.npy')+0.0001)
    #features=preprocessing.scale(features)
    if norm=='yes':
        features=preprocessing.normalize(features)
    return features
def standarized_normalized_c_vs_n(features, norm='yes'):
    features=(features-np.load('trained_models/mean_covid_vs_non.npy'))/(np.load('trained_models/std_covid_vs_non.npy')+0.0001)
    #features=preprocessing.scale(features)
    if norm=='yes':
        features=preprocessing.normalize(features)
    return features
def apply_covidc(image_path):
    feats=extract_transfer_learning_features(image_path)
    trained_model_c_vs_n_w=np.load('trained_models/weights/weight_vector_SVM_densent_covid_vs_noncovid.npy')
    trained_model_c_vs_n_b=np.load('trained_models/weights/bias_SVM_densent_covid_vs_noncovid.npy')
    
    if np.dot(trained_model_c_vs_n_w[0],standarized_normalized_c_vs_n([feats])[0])+trained_model_c_vs_n_b[0]<=0:
        print("Diagnosis: non-COVID")
    else:
        print('Diagnosis: COVID-19')
        trained_model_severity_w=np.load('trained_models/weights/weight_vector_SVM_densent_covid_severity.npy')
        trained_model_severity_b=np.load('trained_models/weights/bias_SVM_densent_covid_severity.npy')
        if np.dot(trained_model_severity_w[0],standarized_normalized_severity([feats])[0])+trained_model_severity_b[0]<=0:
            print("Severity: Mild")
        else:
            print("Severity: Severe")
    
if __name__ == "__main__":
    apply_covidc('input_image_example.png')
