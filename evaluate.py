import warnings
warnings.filterwarnings('ignore')
import os
import sys

import glob
import lxml.etree as ET
from pprint import pprint
import time
import argparse

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import cv2
from PIL import Image

import tensorflow as tf
from keras.models import load_model
from keras import backend as K

from syntactical_analysis.lexer import Lexer
from syntactical_analysis.parser import Parser
from syntactical_analysis.code_gen import CodeGenerator

from utils import one_hot_encode_to_char_list, get_symbols, generate_eqns

K.set_image_data_format('channels_first')

# set env variables
os.environ['PYTHONPATH'] = r'D:\SharedLinux_D\CPSC_597\OCR_Math_Expressions\models\research;D:\SharedLinux_D\CPSC_597\OCR_Math_Expressions\models\research\slim;'
# turn off tensorflow debugging info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# import od
OD_PATH = 'models/research/object_detection/'

# append object_detection dir
sys.path.append(OD_PATH+"..")
from object_detection.utils import ops as utils_ops
# od imports
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# ---------------------------------------------------------------
# Globals
# VARIABLES
MODEL_NAME = 'faster_rcnn_resnet50_inference_graph'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = OD_PATH +'faster_rcnn_resnet50_training/'+ MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = OD_PATH + "faster_rcnn_resnet50_training/data/object-detection.pbtxt"
MODELS_PATH = "trained_models/"
PATH_TO_EVALUATE_IMAGES_DIR = "datasets/object_detection/evaluate/"


# ---------------------------------------------------------------
# Object Detection Helpers
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
              'num_detections', 'detection_boxes', 'detection_scores',
              'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                 feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
              'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


# ---------------------------------------------------------------
def evaluate(image_paths, save_xml, debug, cc_model):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    # load character classification model
    cnn = load_model(cc_model)

    # load syntactic analysis objects
    lexer = Lexer()
    parser = Parser()
    codegen = CodeGenerator()

    # load symbols
    symbols = pd.read_csv("processed_data/symbols.csv").drop(["Unnamed: 0"], axis=1)

    # local function to get label from symbol no
    def get_symbol_from_id(id_):
        return symbols[symbols['new_id']==id_].latex.values[0]

    # evaluate images
    for image_path in image_paths:
        image_path = image_path.replace("\\", "/")
        im_name = image_path.split("/")[-1]
        image = Image.open(image_path)

        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = np.asarray(image)
        if len(image_np.shape) < 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        else:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
        print('-' * 50)
        print('Evaluating Image: ', image_path)

        print('Image shape: ', image_np.shape)
        # Actual detection.
        output_dict = run_inference_for_single_image(image_np, detection_graph)

        # crop image and display
        im_width, im_height = image.size
        cropped_images = []
        box_syms = []
        for i in range(output_dict['num_detections']):
            if output_dict['detection_scores'][i] > 0.8:
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.array([output_dict['detection_boxes'][i]]),
                    np.array([output_dict['detection_classes'][i]]),
                    np.array([output_dict['detection_scores'][i]]),
                    category_index,
                    instance_masks=output_dict.get('detection_masks'),
                    use_normalized_coordinates=True,
                    line_thickness=1)
                (ymin, xmin, ymax, xmax) = output_dict['detection_boxes'][i]
                (left, right, top, bottom) = (xmin * im_width,
                                              xmax * im_width,
                                              ymin * im_height,
                                              ymax * im_height)

                if debug:
                    print('*' * 15)
                    print("Box: ")
                    print("left: ", left, " right: ", right, " top: ", top, " bottom ", bottom)

                cropped_image = image.crop((left, top, right, bottom))

                # Predict label for cropped image with CNN
                c_im = np.asarray(cropped_image)
                c_im = cv2.cvtColor(c_im, cv2.COLOR_BGRA2RGBA)
                c_im = cv2.cvtColor(c_im, cv2.COLOR_RGBA2RGB)
                c_im = cv2.cvtColor(c_im, cv2.COLOR_RGBA2GRAY)
                c_im = cv2.resize(c_im, (32, 32))
                c_im = c_im / 255.0
                # Symbol Prediction
                # CNN
                c_im = np.resize(c_im, (1, 1, 32, 32))
                res = cnn.predict(c_im)
                res = res.flatten()
                lbls = one_hot_encode_to_char_list(res, threshold=0.01, get_max=False)
                syms = []
                if debug:
                    print("CNN/ANN Predictions:")
                for j in lbls:
                    symbols_row = symbols[symbols['new_id'] == j[0]][['latex', 'old_symbol', 'new_id']]
                    latex, old_symbol, new_id = symbols_row.iloc[0].to_list()
                    conf_score = round(j[1] * 100, 2)
                    syms.append((get_symbol_from_id(new_id), conf_score))
                    if debug:
                        print('Symbol_id:', new_id, ', Latex:', latex, ', Confidence Score:', conf_score, '%')
                box_syms.append(({
                                     'x1': left,
                                     'x2': right,
                                     'y1': top,
                                     'y2': bottom
                                 },
                                 syms))
                plt.figure(figsize=(1, 1))
                plt.axis('off')
                temp_ = np.asarray(cropped_image)
                plt.imshow(temp_)
                plt.show()
                cropped_images.append(np.asarray(cropped_image))
        plt.figure(figsize=(7, 7))
        plt.axis('off')

        # Predict xml
        box_syms = sorted(box_syms, key=lambda x: x[0]['x1'])
        sym_preds = [i[1] for i in box_syms]
        identified_syms = get_symbols(sym_preds)
        eqns = generate_eqns(identified_syms)
        predicted_eqns = []
        xml_cnt = 0
        for i in eqns:
            try:
                tokens = lexer.generate_tokens(list(i))
                tree = parser.generate_tree(tokens)
                xml_tree = codegen.gen_mathml(tree)
                xml = ET.tostring(xml_tree, pretty_print=True).decode('utf-8')
                if save_xml:
                    xml_tree.write(f'{im_name.split(".")[0]}_{str(xml_cnt)}.xml', encoding='utf-8')
                predicted_eqns.append(xml)
                xml_cnt += 1
            except:
                pass

        # print generated xmls
        if len(predicted_eqns) > 0:
            print('*'*10)
            print('Generated XMLs:')
            for eq in predicted_eqns:
                print(eq)
                print('*' * 10)
        else:
            print('*'*10)
            print('Failed to get syntactically correct XML representation')
            print('*'*10)

        # show image
        temp_ = cv2.resize(image_np, (int(im_width), int(im_height)))
        cv2.imshow(im_name, temp_)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    # cli args
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-d', '--debug', type=bool, default=False,
                           help='True to print predictions')
    argparser.add_argument('-p', '--image_dir', type=str, default=PATH_TO_EVALUATE_IMAGES_DIR,
                           help='Directory containing multiple images in .png format')
    argparser.add_argument('-i', '--image', type=str, default=None,
                           help='Path to evaluate a single image file')
    argparser.add_argument('-s', '--save_xml', type=bool, default=False,
                           help='True to save generated MathML file to disk')
    argparser.add_argument('-m', '--cc_model', type=str, required=True,
                           help='Path to load Character Classfication model')
    args = argparser.parse_args()

    if args.image is not None:
        images = [args.image]
    else:
        images = list(glob.glob(args.image_dir+'*.png'))
    print('-'*50)
    print("Image paths to evaluate: ")
    for i in images:
        print(i)
    evaluate(image_paths=images, save_xml=args.save_xml, debug=args.debug, cc_model=args.cc_model)


if __name__ == '__main__':
    main()
