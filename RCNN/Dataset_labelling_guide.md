## Dataset- Create Image
#### For creating images for the dataset, I am using MS Paint owing to its simplicity.
#### For the images of characters, I am using characters from HASYv2 dataset.
You can download HASYv2 dataset using this link
https://www.kaggle.com/guru001/hasyv2

### Steps
1. Create a new image in MS Paint of size **320 x 320 px** by dragging the white workspace. (You can see the image size at the bottom)
2. Next import the images of characters you want to use from HASYv2 dataset. (If equation contains `a = b + c`, insert images for characters `a, =, b, +, c`)
3. Rearrange the characters so that they express the equation correctly. If the characters do not fit the image, resize them by maximum of 5 pixels in both x and y axis. Also make sure that there is sufficient space in between each character so that the bounding boxes for two adjacent characters that will be labelled later will not conincide.
4. Save the image with a file name as `exp{nnn}.png` where nnn is a sequential number. (For example, `exp001.png`)





---
## Dataset- Label Image
#### For labelling datasets, I am using the following tool:
https://github.com/tzutalin/labelImg

You can install the tool using PyPI with the command:
Linux and Mac: `pip3 install labelImg`
Windows: `pip install labelImg`

If it did not install properly, you can find the steps to install on their github page.

### Steps to use:

1. To use the tool, enter `labelImg` in Command Prompt on Windows or Terminal in Linux and Mac. (Note: Do not close Command Prompt or Terminal while the tool is running.)

2. Now, click on **File** menu -> **Open Dir**. Browse to directory where images are stored.

3. Now, click on **File** -> **Change Save Dir** and browse to the same directory.

4. Now, click on **File** and make sure there is an option named **PascalVOC**. (If you see **YOLO** option, click on it so that it changes to **PascalVOC**)

5. Now, click on **View** menu and check **Single Class Mode**.

6. On the bottom right of the labelImg tool, you will see a list of images. Double click on an image which you want to annotate/label.

7. Press **W** key and select a box around the character to annotate. (Note: Make sure that the box only contains features for that character. Also, the box should be small as possible without losing any character features.)

8. Enter **char** as the label name. (Note: All characters are labelled using a single class named **char**)

9. Press ***Ctrl + S*** to save the annotations. It will save a ***{Image name}.xml*** file in your images directory. It contains the annotations.

10. After you are done annotating the image, save it and double click on the next image to annotate.

Tips:
* If you want to delete an annotation, click on it and press **Delete** key.
* The github page contains a few hotkeys which do help to label quickly.



