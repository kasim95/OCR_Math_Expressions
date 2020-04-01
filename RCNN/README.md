1. Go to https://github.com/protocolbuffers/protobuf/releases and download protoc-3.11.2-win64.zip for a 64 bit pc or protoc-3.11.2-win32.zip for 32 bit pc and extract it to {C:/Program Files} or any other secure directory


2. Open Anaconda Prompt and cd to Product_Detection/tensorflow_od_api/research/

  
3. Type this command in Anaconda prompt (one single command) and press enter:
(Make sure you have protobuff cloned into the directory in C)
`"C://Program Files/protoc-3.11.2-win64/bin/protoc.exe" object_detection/protos/*.proto --python_out=.`
  

4. Replace {enter path upto Product Detection here} with the path to Product Detection in your pc and then type this command in Anaconda prompt and press enter:

`set PYTHONPATH=D:\SharedLinux_D\CPSC_597\Object_Detection_Expressions\research;D:\SharedLinux_D\CPSC_597\Object_Detection_Expressions\research\slim`

5. Use these commands to generate tfrecords for train and test

`python generate_tfrecord.py --csv_input=data/train_labels.csv --output_path=data/train.record --image_dir=images/train/`

`python generate_tfrecord.py --csv_input=data/test_labels.csv --output_path=data/test.record --image_dir=images/test/`

6. Download config and checkpoint files from
<a>https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md</a>

7. Change PATH_TO_BE_CONFIGURED in config file

8. Copy checkpoint to research/object_detection/

9. Run `python legacy/train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_resnet50_coco.config` to train


