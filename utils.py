import os
from multiprocessing import Pool
from timeit import default_timer as timer

import nibabel as nib
from sklearn.model_selection import train_test_split
import numpy as np
from dipy.io.image import load_nifti

from preprocess.get_subvolume import get_training_sub_volumes, get_test_subvolumes
# TODO: Condicionar la creacion de los directorios para separar train y test en carpetas diferentes
# TODO: Que en la carpeta de predicciones cuando se creen los directorios solo se creen los de test
# Por ahora estoy borrando manualmente las carpetas vacias del folder subvolumes_predict que contiene
# las predicciones y reconstrucciones
def make_dirs():
    """Construye los directorios donde se almacenan todos los datos en caso de que no existan aun.
    Como resultado retorna un diccionario con los paths.

    claves del diccionario: 
    ['SUBVOLUME_FOLDER', 'SUBVOLUME_MASK_FOLDER', 'RESULTADOS', 'DATABASE_DIR', 'SAMPLES']

    :return: diccionario con los paths 
    :rtype: dict
    """
    processed_paths = {}
    processed_sample_paths = {}
    PARENT_DIR = os.getcwd()
    DATABASE_DIR = os.path.join(PARENT_DIR, "NFBS_Dataset")
    SAMPLES_FOLDERS = next(os.walk(DATABASE_DIR))[1]
    
    processed_paths["PROCESSED_DIR"] = os.path.join(PARENT_DIR, "processed")
    PROCESSED_DIR = processed_paths["PROCESSED_DIR"]
    processed_paths["SUBVOLUME_FOLDER"] = os.path.join(PROCESSED_DIR,"subvolumes")
    processed_paths["SUBVOLUME_MASK_FOLDER"] = os.path.join(PROCESSED_DIR,"subvolumes_masks")
    processed_paths["RESULTADOS"] = os.path.join(PROCESSED_DIR,"subvolumes_predict")

    for path in processed_paths:
        dir_exists = os.path.exists(processed_paths[path])
        if not dir_exists:
            os.makedirs(processed_paths[path])

    processed_paths.pop("PROCESSED_DIR")
    
    for path in processed_paths:
        for sample in SAMPLES_FOLDERS:
            sample_dir = os.path.join(processed_paths[path], sample)
            dir_exists = os.path.exists(sample_dir)
            if not dir_exists:
                os.makedirs(sample_dir)
        processed_sample_paths[path] = processed_paths[path]
    processed_sample_paths["DATABASE_DIR"] = DATABASE_DIR
    processed_sample_paths["SAMPLES"] = SAMPLES_FOLDERS
    return processed_sample_paths

# Se crean los directorios
# Estos paths se usan en las funciones
paths = make_dirs()

def create_file_name(sample, mask=False):
    """Funcion que genera el nombre de los archivos para automatizar la lectura

    :param sample: codigo del paciente (carpeta)
    :type sample: string
    :param mask: Define si se quiere la mascara o el volumen normal, defaults to False
    :type mask: bool, optional
    :return: nombre del archivo a leer
    :rtype: path
    """
    if mask:
        file_name = "sub-"+sample+"_ses-NFB3_T1w_brainmask.nii.gz"
    else:
        file_name = "sub-"+sample+"_ses-NFB3_T1w.nii.gz"
    file_name = os.path.join(sample, file_name)
    return file_name

def get_sub_volumes_train(sample):
    """Esta funcion recibe el nombre de una muestra (carpeta) y genera los sub-volumenes.
    SOLO PARA TRAIN!!

    :param sample: nombre de la carpeta que contiene el volumen y su mascara
    :type sample: path
    """
    img = os.path.join(paths["DATABASE_DIR"], create_file_name(sample, False))
    img_mask = os.path.join(paths["DATABASE_DIR"], create_file_name(sample, True))
    img = nib.load(img)
    img_mask = nib.load(img_mask)
    image = img.get_fdata()
    image_mask = img_mask.get_fdata()
    SAVE_PATH_SUBVOLUME = os.path.join(paths["SUBVOLUME_FOLDER"], sample)
    SAVE_PATH_SUBMASK = os.path.join(paths["SUBVOLUME_MASK_FOLDER"], sample)
    get_training_sub_volumes(image, img.affine, image_mask, img_mask.affine, 
                                    SAVE_PATH_SUBVOLUME, SAVE_PATH_SUBMASK, 
                                    classes=1, 
                                    orig_x = 256, orig_y = 256, orig_z = 192, 
                                    output_x = 128, output_y = 128, output_z = 16,
                                    stride_x = 32, stride_y = 32, stride_z = 8,
                                    background_threshold=0.1)

def get_sub_volumes_test(sample):
    """Esta funcion recibe el nombre de una muestra (carpeta) y genera los sub-volumenes.
    SOLO PARA TEST!!

    NOTA: el stride de los volumenes de test es mas grande 
    debido a que genera muchas mas imagenes y no tengo espacio.

    :param sample: nombre de la carpeta que contiene el volumen y su mascara
    :type sample: path
    """
    img = os.path.join(paths["DATABASE_DIR"], create_file_name(sample, False))
    img_mask = os.path.join(paths["DATABASE_DIR"], create_file_name(sample, True))
    img = nib.load(img)
    img_mask = nib.load(img_mask)
    image = img.get_fdata()
    image_mask = img_mask.get_fdata()
    SAVE_PATH_SUBVOLUME = os.path.join(paths["SUBVOLUME_FOLDER"], sample)
    SAVE_PATH_SUBMASK = os.path.join(paths["SUBVOLUME_MASK_FOLDER"], sample)
    get_test_subvolumes(image, img.affine, image_mask, img_mask.affine, 
                                    SAVE_PATH_SUBVOLUME, SAVE_PATH_SUBMASK, 
                                    orig_x = 256, orig_y = 256, orig_z = 192, 
                                    output_x = 128, output_y = 128, output_z = 16,
                                    stride_x = 128, stride_y = 128, stride_z = 8)

train_files, val_files = train_test_split(paths["SAMPLES"], test_size=0.2, random_state=42)
test_files, val_files = train_test_split(val_files, test_size=0.5, random_state=42)
train_files.extend(val_files)

def reconstruction(orig_x = 256, orig_y = 256,orig_z = 192,
                   output_x = 128, output_y = 128, output_z = 16,
                   stride_x = 128, stride_y = 128, stride_z = 16,
                   test_files=None, path_resultados=None):
    """Esta funcion toma una lista con los nombres de las carpetas que contienen los 
    sub-volumenes de cada paciente y el path donde se almacenan los sub-volumenes de test.
    Luego realiza la reconstruccion de los volumenes a su tamaño original.

    Guarda las reconstrucciones en path_resultados con el nombre de cada muestra (el mismo de la carpeta)

    :param orig_x: tamaño original del volumen en X, defaults to 256
    :type orig_x: int, optional
    :param orig_y: tamaño original del volumen en Y, defaults to 256
    :type orig_y: int, optional
    :param orig_z: tamaño original del volumen en Z, defaults to 192
    :type orig_z: int, optional
    :param output_x: tamaño final de cada sub-volumen en X, defaults to 128
    :type output_x: int, optional
    :param output_y: tamaño final de cada sub-volumen en Y, defaults to 128
    :type output_y: int, optional
    :param output_z: tamaño final de cada sub-volumen en Z, defaults to 16
    :type output_z: int, optional
    :param stride_x: Stride en el eje X, defaults to 128
    :type stride_x: int, optional
    :param stride_y: Stride en el eje Y, defaults to 128
    :type stride_y: int, optional
    :param stride_z: Stride en el eje Z, defaults to 8
    :type stride_z: int, optional
    :param test_files: Contiene los nombres de las carpetas con los volumenes de cada paciente, defaults to None
    :type test_files: python list, optional
    :param path_resultados: Contiene el path donde estan las mascaras predichas para los sub-volumenes de test, defaults to None
    :type path_resultados: string, optional

    Ejemplo de uso:

    test_files = ['A00057965','A00055373','A00055542']
    paths["RESULTADOS"] = "c:\\Users\\Bersek\\Desktop\\proyecto_minciencias\\Minciencias_pruebas\\processed\\subvolumes_predicts"

    reconstruction(test_files=test_files, path_resultados=paths["RESULTADOS"], stride_z=8)

    """
    z = 0
    for i in range(0, orig_z-output_z+1, stride_z):
        z+=1
        y = 0
        for j in range(0, orig_y-output_y+1, stride_y):
            y+=1
            x = 0
            for k in range(0, orig_x-output_x+1, stride_x):
                x+=1
    print("X:",x, " Y:",y, " Z:",z)
    for sample in test_files:
        path = []
        flag = False
        num = 0
        tries = 0

        for subvol in os.listdir(os.path.join(path_resultados,sample)):
            path.append(os.path.join(path_resultados,sample,subvol))
            path = sorted(path, key=len)
        for i in range(0, z):
            for j in range(0, x):
                for k in range(0, y):
                    nifti = np.asarray(nib.load(path[tries]).get_fdata().astype(np.float32)) 
                    tries += 1
                    if num == 0 or num == 2:
                        array_aux = nifti 
                    elif num == 1:
                        array_nuevo = np.concatenate((array_aux, nifti[output_x-stride_x:output_x,:,:]), axis=0)
                    elif num == 3:
                        array_aux2 = np.concatenate((array_aux, nifti[output_x-stride_x:output_x,:,:]), axis=0)
                        array_nuevo2 = np.concatenate((array_nuevo, array_aux2[:,output_y-stride_y:output_y,:]), axis=1)
                    num += 1
                    if num==4 and not flag:
                        array_nuevo3 = array_nuevo2
                        flag=True
                        num = 0
                    elif num==4 and  flag:
                        array_nuevo3 = np.concatenate((array_nuevo3, array_nuevo2[:,:,output_z-stride_z:output_z]), axis=2)
                        num = 0

        _, img_affine = load_nifti(path[1])
        nii_segmented = nib.Nifti1Image(array_nuevo3, img_affine)
        nib.save(nii_segmented, os.path.join(path_resultados,sample+".nii"))

# Procesamiento de los volumenes de entrenamiento y test
# Se usa multiprocesamiento porque toma mucho tiempo
if __name__ == '__main__':
    start = timer()
    with Pool(6) as pool:
        # Comentar las lineas de abajo dependiendo de si quiere generar los volumenes o no
        pool.map(get_sub_volumes_train, train_files)
        pool.map(get_sub_volumes_test, test_files)
    end = timer()
    
    print("Elapsed time {}".format(end-start))