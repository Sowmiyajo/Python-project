import tensorflow as tf
import numpy as np

from tkinter import *
import os
from tkinter import filedialog
import cv2
import argparse, sys, os
import time
from matplotlib import pyplot as plt
from tkinter import messagebox





def endprogram():
	print ("\nProgram terminated!")
	sys.exit()






def file_sucess():
    global file_success_screen
    file_success_screen = Toplevel(training_screen)
    file_success_screen.title("File Upload Success")
    file_success_screen.geometry("150x100")
    file_success_screen.configure(bg='pink')
    Label(file_success_screen, text="File Upload Success").pack()
    Button(file_success_screen, text='''ok''', font=(
        'Verdana', 15), height="2", width="30").pack()


global ttype

def training():
    global training_screen

    global clicked

    training_screen = Toplevel(main_screen)
    training_screen.title("Training")
    # login_screen.geometry("400x300")
    training_screen.geometry("600x450+650+150")
    training_screen.minsize(120, 1)
    training_screen.maxsize(1604, 881)
    training_screen.resizable(1, 1)
    training_screen.configure()
    # login_screen.title("New Toplevel")



    Label(training_screen, text='''Upload Image ''', background="#d9d9d9", disabledforeground="#a3a3a3",
          foreground="#000000",  width="300", height="2", font=("Calibri", 16)).pack()
    Label(training_screen, text="").pack()


    options = [
        "GliomaTumor",
        "MeningiomaTumor",
        "NoTumor",
        "PituitaryTumor"
    ]

    # datatype of menu text
    clicked = StringVar()


    # initial menu text
    clicked.set("select")

    # Create Dropdown menu
    drop = OptionMenu(training_screen, clicked, *options )
    drop.config(width="30")

    drop.pack()

    ttype=clicked.get()

    Button(training_screen, text='''Upload Image''', font=(
        'Verdana', 15), height="2", width="30", command=imgtraining).pack()

def vgg():
    import CnnModel as vgg



def imgtraining():
    name1 = clicked.get()

    print(name1)

    import_file_path = filedialog.askopenfilename()
    import os
    s = import_file_path
    os.path.split(s)
    os.path.split(s)[1]
    splname = os.path.split(s)[1]


    image = cv2.imread(import_file_path)
    #filename = 'Test.jpg'
    filename = 'Dataset/Testing/'+name1+'/'+splname


    cv2.imwrite(filename, image)
    print("After saving image:")

    fnm = os.path.basename(import_file_path)
    print(os.path.basename(import_file_path))

    # file_sucess()

    print("\n*********************\nImage : " + fnm + "\n*********************")
    img = cv2.imread(import_file_path)
    if img is None:
        print('no data')

    img1 = cv2.imread(import_file_path)
    print(img.shape)
    img = cv2.resize(img, ((int)(img.shape[1] / 5), (int)(img.shape[0] / 5)))
    original = img.copy()
    neworiginal = img.copy()
    cv2.imshow('original', img1)
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    img1S = cv2.resize(img1, (960, 540))

    cv2.imshow('Original image', img1S)

    dst = cv2.fastNlMeansDenoisingColored(img1, None, 10, 10, 7, 21)
    cv2.imshow("Noise Removal", dst)







def testing():
    global testing_screen
    testing_screen = Toplevel(main_screen)
    testing_screen.title("Testing")
    # login_screen.geometry("400x300")
    testing_screen.geometry("600x450+650+150")
    testing_screen.minsize(120, 1)
    testing_screen.maxsize(1604, 881)
    testing_screen.resizable(1, 1)
    testing_screen.configure()
    # login_screen.title("New Toplevel")

    Label(testing_screen, text='''Upload Image''',  width="300", height="2", font=("Palatino Linotype", 16)).pack()
    Label(testing_screen, text="").pack()
    Label(testing_screen, text="").pack()
    Label(testing_screen, text="").pack()
    Button(testing_screen, text='''Upload Image''', font=(
        'Palatino Linotype', 15), height="2", width="30", command=imgtest).pack()


global affect
def imgtest():


    import_file_path = filedialog.askopenfilename()

    image = cv2.imread(import_file_path)
    print(import_file_path)
    filename = 'Output/Out/Test.jpg'
    cv2.imwrite(filename, image)
    print("After saving image:")
    #result()

    #import_file_path = filedialog.askopenfilename()
    print(import_file_path)
    fnm = os.path.basename(import_file_path)
    print(os.path.basename(import_file_path))

   # file_sucess()

    print("\n*********************\nImage : " + fnm + "\n*********************")
    img = cv2.imread(import_file_path)
    if img is None:
        print('no data')

    img1 = cv2.imread(import_file_path)
    print(img.shape)
    img = cv2.resize(img, ((int)(img.shape[1] / 5), (int)(img.shape[0] / 5)))
    original = img.copy()
    neworiginal = img.copy()
    img1 = cv2.resize(img1, (960, 540))
    #cv2.imshow('original', img1)
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    img1S = cv2.resize(img1, (960, 540))

    cv2.imshow('Original image', img1S)
    grayS = cv2.resize(gray, (960, 540))
    cv2.imshow('Gray image', grayS)

    dst = cv2.fastNlMeansDenoisingColored(img1, None, 10, 10, 7, 21)
    dst = cv2.resize(dst, (960, 540))
    cv2.imshow("Nosie Removal", dst)
    result()




def result():
    import warnings
    warnings.filterwarnings('ignore')

    import tensorflow as tf
    classifierLoad = tf.keras.models.load_model('model.h5')

    import numpy as np
    from keras.preprocessing import image
    test_image = image.load_img('./Output/Out/Test.jpg', target_size=(150, 150))
    # test_image = image.img_to_array(test_image)
    #test_image = np.expand_dims(test_image, axis=0)
    #test_image = np.array(test_image).reshape(1,150,150, 3)

    # Add an extra dimension to simulate a batch of images
    reshaped_image = np.expand_dims(test_image, axis=0)  # Adds a dimension at index 0

    # Add time step dimension for LSTM (None, 1, 150, 150, 3)
    reshaped_image = np.expand_dims(reshaped_image, axis=1)

    result = classifierLoad.predict(reshaped_image)

    pred = np.argmax(result, axis=1)
    print(pred)
    if pred[0] == 0:
        messagebox.showinfo("Result", 'GliomaTumor')

    elif pred[0] == 1:
        messagebox.showinfo("Result", 'MeningiomaTumor')
        #messagebox.showinfo("Prescription", 'use of antibiotics, painkillers, and wound care sprays to treat symptoms')
    elif pred[0] == 2:
        messagebox.showinfo("Result", 'NoTumor')
    elif pred[0] == 3:
        messagebox.showinfo("Result", 'PituitaryTumor')








def main_account_screen():
    global main_screen
    main_screen = Tk()
    width = 600
    height = 600
    screen_width = main_screen.winfo_screenwidth()
    screen_height = main_screen.winfo_screenheight()
    x = (screen_width / 2) - (width / 2)
    y = (screen_height / 2) - (height / 2)
    main_screen.geometry("%dx%d+%d+%d" % (width, height, x, y))
    main_screen.resizable(0, 0)
    # main_screen.geometry("300x250")
    main_screen.configure()
    main_screen.title(" Brain Disease Prediction")

    Label(text="Brain Disease Prediction", width="300", height="5", font=("Calibri", 16)).pack()


    Label(text="").pack()
    Button(text="Training", font=(
        'Verdana', 15), height="2", width="30", command=vgg, highlightcolor="black").pack(side=TOP)

    Label(text="").pack()
    Button(text="Testing", font=(
        'Verdana', 15), height="2", width="30", command=testing, highlightcolor="black").pack(side=TOP)
    Label(text="").pack()


    main_screen.mainloop()


main_account_screen()

