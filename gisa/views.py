from django.http import HttpResponse
from django.shortcuts import render
from .models import Image,segment
import cv2
from . import forms
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model,model_from_json
from keras.preprocessing.image import img_to_array
from django.contrib import messages
import base64
import io
import PIL
import numpy as np
import os
from project.settings import BASE_DIR
from project.settings import MEDIA_DIR
from django.core.files.base import ContentFile
from django.core.files import File
from django.http.response import StreamingHttpResponse
import threading
import gzip
from threading import Thread, Lock
_db_lock = Lock()

def home_view(request):
    return render(request, 'home.html')

def do_segmentation(h_l, s_l, v_l, h_h, s_h, v_h, target_image,name_image):
    print('inside do_segmentation function')
    print("high values")
    print('hue : {} \n saturation: {} \n value : {}\n'.format( h_h, s_h, v_h))
    print("low values")
    print('hue : {} \n saturation: {} \n value : {}\n'.format(h_l, s_l, v_l))

    target_image = PIL.Image.open(target_image)
    target_image = np.float32(target_image)
    print('Target image type : ' + str(type(target_image)))
    print('Target_immage shape : ' + str(target_image.shape))
    hsv=cv2.cvtColor(target_image,cv2.COLOR_BGR2HSV)
    # hsv_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
    # Defining color theshold
    lower = np.float32([h_l, s_l, v_l])
    higher = np.float32([h_h, s_h, v_h])
    # print(lower,higher)
    # hsv_frame = np.float32(hsv)
    if target_image.any():
        print("Not none in parameter image")
    image_mask = cv2.inRange(hsv, lower, higher)
    if image_mask.any():
        print('Not none in image_mask')
    print("mask shape = {}".format(image_mask.shape))
    # orig_img = PIL.Image.open(orig_img)
    # orig_img = np.float32(orig_img)
    # print('orig_img shape : ' + str(orig_img))
    output1 = cv2.bitwise_and(target_image, target_image, mask = image_mask)
    if output1.any():
        print('Non-zero')
    # print(output1.shape)
    try:
        _,buffer_image1 = cv2.imencode('.jpeg', output1)
        f_image1 = buffer_image1.tobytes()
        f1 = ContentFile(f_image1)
        image_file = File(f1, name = name_image )
        return image_file
    except IndexError:
        print("Index out of range!")
        pass


def prepare(ima , category):
    if category=='Number' or category == 'Sign Language Hand Gesture Digit':
        IMG_SIZE = 100
    else:
        IMG_SIZE = 200
    print('ima shape' + str(ima.size))
    print('category: ' + category)
    # ima = img_to_array(ima)
    img_array = ima*255
    img_array=img_array/255.0  # filepathread in the image, convert to grayscale
    print('img_array shape:' + str(img_array.shape))    #(200,200,3)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    print('new_array shape:' + str(new_array.shape))      #(100,100,3)
    new_array =  new_array.reshape(-1,IMG_SIZE, IMG_SIZE,1)
    print('out of prepare')
    print('prepared image shape' + str(new_array.shape))
    return new_array

def skin_DT(image):  
    print('skin_dt image : ' + str(type(image)))
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    print('h')
    ycbcr=cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    print('f')
    b,g,r=cv2.split(image)
    y,cr,cb=cv2.split(ycbcr)
    h,s,v=cv2.split(hsv)
    # print('h = ' + str(h))
    # print('s = ' + str(s))
    # print('v = ' + str(v))
    s=s/(h+s+v)
    # h=h/(h+s+v)
    # v=v/(h+s+v)
    ans=((0<=h)*(h<=50)*(0.23<s)*(s<0.68)*(r>95)*(g>40)*(b>20)*(r>g)*(r>b)*(abs(r-g)>15))+((r>95)*(g>40)*(b>20)*(r>g)*(r>b)*(abs(r-g)>15)*(cr>135)*(cb>85)*(y>80)*(cr<=(1.5862*cb+20))*(cr>=(0.3448*cb+76.2069))*(cr>=(-4.5652*cb+234.5652))*(cr<=-1.15*cb+301.75)*(cr<=-2.2857*cb+432.85))
    b=b*ans
    g=g*ans
    r=r*ans
    new=cv2.merge((b,g,r))
    ans=ans*255
    new1=cv2.merge((ans,ans,ans))
    print('new1 shape:' + str(new1.shape))
    cv2.imwrite("media/uploaded_images/test.jpeg",new1)
    print('written')
    new1=cv2.imread("media/uploaded_images/test.jpeg")
    new1 = cv2.cvtColor(new1, cv2.COLOR_BGR2GRAY)
    print('new1 shape: ' + str(new1.shape))
    print('d')
    dist = cv2.distanceTransform(new1, cv2.DIST_L2, 5)
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    print('out of dt')
    return dist 

# def predict_image(image, category, name_image):
#     try :
#         print('Inside predict_image shape : ' + str(image.shape))
#         model_path = os.path.join(BASE_DIR,'01resnet.model')
#         model = load_model(model_path, compile=False)
#         image1 = image.copy()
#         print('Image1 shape' + str(image1.shape))
#         dist = skin_DT(image)       #(200,200)
#         prediction = model.predict(prepare(dist,category))
#         prediction=np.argmax(prediction)
#         x1=prediction
#         try:
#             cv2.putText(image1,str(x1),(60,60),cv2.FONT_HERSHEY_SIMPLEX,3.0,(255,0,0),lineType=cv2.LINE_AA)
#         except:
#             print("Variable x1 is not empty")
#         _,buffer_image1 = cv2.imencode('.jpg', image1)
#         f_image1 = buffer_image1.tobytes()
#         f1 = ContentFile(f_image1)
#         image_file = File(f1, name = name_image )
#         return image_file,x1
#     except Exception as e:
#         print(e)
def func(image):
    new1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dist = cv2.distanceTransform(new1, cv2.DIST_L2, 5)
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    print('dist shape : ' + str(dist.shape))
    return dist

def predict_image(image, category, name_image):
    try:
        print('Inside predict_image shape :' + str(image.shape))
        model_path = os.path.join(BASE_DIR, '01resnet.model')
        model = load_model(model_path, compile = False)
        image1 = image.copy()
        # print('Image1 test' + str(image1.shape))
        print('Image1 shape : ' + str(image1.shape))
        dist = func(image)
        prediction = model.predict([prepare(dist, category)])
        prediction=np.argmax(prediction)
        x1=str(prediction)
        print('x1 is : ' + x1)
        try:
            cv2.putText(image1,str(x1),(60,60),cv2.FONT_HERSHEY_SIMPLEX,3.0,(0,0,255),lineType=cv2.LINE_AA)
        except:
            print("Variable x1 is not empty")
        _,buffer_image1 = cv2.imencode('.jpeg', image1)
        f_image1 = buffer_image1.tobytes()
        f1 = ContentFile(f_image1)
        image_file = File(f1, name = name_image )
        return image_file,x1
    except Exception as e:
        print(e)



def formpage(request):
    global flag
    upload_image = Image()
    modified_image = Image()
    temp_form = forms.TempForm({'predictIt':'no'})
    image_form = forms.ImageForm()
    if request.method == "POST":                #elif->req.is_ajax   
      temp_form = forms.TempForm(request.POST)
      t_value = request.POST.get('predictIt')
      if t_value == 'yes':
          #test_image = upload_image.uploads
          img_obj = Image.objects.filter().order_by('-id')[0]
          print("image object = {}".format(img_obj))
          print("image object image = {}".format(img_obj.uploads))
          category = img_obj.category
          name_image = img_obj.uploads.name
          print(name_image)
          print(type(img_obj))
          print('retreived')
          test_image = img_obj.uploads
          image_bytes = test_image.read()
          target_image = PIL.Image.open(io.BytesIO(image_bytes))
          target_image = target_image.resize((200,200),PIL.Image.ANTIALIAS)
          print(type(target_image))
          image_array = np.array(target_image)
          image_file ,x1 = predict_image(image_array,category,name_image)
          print('Imgage_file type: ' + str(type(image_file)))
          modified_image.uploads = img_obj.uploads
          print("next step")
          modified_image.save()
          context_dict = {'form' : image_form,'temp_form' : temp_form, 'prediction' : x1, 'image_show' : modified_image}   #predicted image
      else:   
        image_form = forms.ImageForm(request.POST,request.FILES)
        if image_form.is_valid():
            print('inside form.vaid')
            category  = image_form.cleaned_data['category']
            if request.FILES.get("uploads",None) is not None:
                print('image prese')
                test_image = request.FILES['uploads']
                image_byte = test_image.read()
                target_image = PIL.Image.open(io.BytesIO(image_byte))
                target_image = target_image.resize((200,200),PIL.Image.ANTIALIAS)
            #   target_image = prepare(target_image, category)
                target_image = np.array(target_image)
                name_image = image_form.cleaned_data['uploads'].name
            #   print(type(target_image))
            #   print(target_image.shape)
            #   image_file ,x1 = predict_image(target_image,category,name_image)
            #   modified_image.uploads = image_file
            #   modified_image.save()
                flag = 1
                if 'uploads' in request.FILES:
                    print('inside function')
                    upload_image.category = image_form.cleaned_data['category']
                    upload_image.uploads = request.FILES['uploads']
                    upload_image.save()
                    print('Saved image' + str(upload_image.uploads.name))
                    upload_obj = Image.objects.filter().order_by('-id')[0]
                    image_id = upload_obj.id
                    print("image id = {}".format(image_id))
                    context_dict = {'form' : image_form, 'temp_form' : temp_form, 'image_show':upload_image}     #uploaded image
            else:
                context_dict = {'form' : image_form , 'temp_form' : temp_form}
          #return HttpResponse('The predicted class is {}'.format(x1))
          #messages.success(request,'The predicted class is {}'.format(x1))
        else:
            print(image_form.errors)


    # elif request.is_ajax():
    #     print("ajax one!")
    #     # High bar values
    #     hue_amount_h = request.GET.get('h_value_h')
    #     saturation_amount_h = request.GET.get('s_value_h')
    #     value_amount_h = request.GET.get('v_value_h')

    #     # low bar values
    #     hue_amount_l = request.GET.get('h_value_l')
    #     saturation_amount_l = request.GET.get('s_value_l')
    #     value_amount_l = request.GET.get('v_value_l')

    #     print("high values")
    #     print('hue : {} \n saturation: {} \n value : {}\n'.format( hue_amount_h, saturation_amount_h, value_amount_h))
    #     print("low values")
    #     print('hue : {} \n saturation: {} \n value : {}\n'.format(hue_amount_l, saturation_amount_l, value_amount_l))

    #     if flag:
    #         print("changing Image")
    #         im = do_segmentation(hue_amount_l, saturation_amount_l, value_amount_l,
    #                              hue_amount_h, saturation_amount_h, value_amount_h, target_image, name_image)
    #         if im:
    #             s = Image.objects.get(Id=image_id)
    #             s.uploads = im
    #             s.save()
    #             img_add = s.uploads.url
    #             return HttpResponse(img_add)
    #         else:
    #             print("Image not available")
    #             context_dict = {'segment_form': temp_form}
    #     else:
    #             print("ajax request not maintained properly")
    #             context_dict = {'segment_form': temp_form}


    else:
        image_form = forms.ImageForm()
        context_dict = {'form' : image_form, 'temp_form' : temp_form}
    print(context_dict)
    return render(request,'predict.html',context = context_dict)

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(1)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = self.frame
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()

cam = VideoCamera()

def gen(camera):
    while True:
        frame = cam.get_frame()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# @gzip.gzip_page
def livepage(request):
    try:
        print('here')
        return StreamingHttpResponse(gen(VideoCamera()), content_type="multipart/x-mixed-replace; boundary=frame")
        # return render(response, 'live.html')
        # return(resp, 'live.html')
    except:  # This is bad! replace it with proper handling
        print('Exception occurred')
        pass
    
    
def button_live(request) :
    submitbutton = request.POST.get('Submit')
    print(submitbutton)
    if submitbutton:
        context = {'submitbutton' : submitbutton}
    else:
        context = {'submitbutton' : None}
    return render(request, 'live.html', context)


def segment_it(request):
    global flag
    global image_id
    global category
    global orig_img
    sform = forms.segmentForm()
    tform = forms.TempForm({'predictIt' : 'no'})
    modified_image = Image()
    s_image = segment()
    if request.method == 'POST':
        print('inside req.method = POST')
        tform = forms.TempForm(request.POST)
        t_value = request.POST.get('predictIt')
        print('tvalue : ' + str(t_value))
        if t_value == 'yes':
            img_obj = segment.objects.filter().order_by('-id')[0]
            print("image object = {}".format(img_obj))
            print("image object image = {}".format(img_obj.uploads))
            category = img_obj.category
            name_image = img_obj.uploads.name
            print(name_image)
            print(type(img_obj))
            print('retreived')
            test_image = img_obj.uploads
            image_bytes = test_image.read()
            target_image = PIL.Image.open(io.BytesIO(image_bytes))
            target_image = target_image.resize((200,200),PIL.Image.ANTIALIAS)
            print(type(target_image))
            image_array = np.array(target_image)
            image_file ,x1 = predict_image(image_array,category,name_image)
            print('Imgage_file type: ' + str(type(image_file)))
            modified_image.uploads = img_obj.uploads
            print("next step")
            modified_image.save()
            context_dict = {'segment_form' : sform,'temp_form' : tform, 'prediction' : x1, 'image_show' : img_obj}
        else :
            sform = forms.segmentForm(request.POST, request.FILES)
            if sform.is_valid():
                print('inside form valid')
                category = sform.cleaned_data['category']
                if request.FILES.get("uploads", None) is not None:
                    print('image present')
                    test_image = request.FILES['uploads']
                    image_byte = test_image.read()
                    name_image = sform.cleaned_data['uploads'].name

                    print(name_image)
                    # hue_amount_l = 0
                    # value_amount_l = 0
                    # saturation_amount_l = 0

                    # hue_amount_h = 255
                    # value_amount_h = 255
                    # saturation_amount_h = 255

                    target_image = PIL.Image.open(io.BytesIO(image_byte))
                    target_image = target_image.resize((200,200),PIL.Image.ANTIALIAS)
                #   target_image = prepare(target_image, category)
                    target_image = np.array(target_image)
                    
                    #segmented_image_showing = do_segmentation(hue_amount_l, saturation_amount_l, value_amount_l,hue_amount_h, saturation_amount_h, value_amount_h, temp_image)
                    flag = 1
                    s_image.uploads = sform.cleaned_data['uploads']
                    s_image.category = sform.cleaned_data['category']
                    s_image.save()
                    print('image saved')
                    #img_add = s_image.Image.url
                    s_obj = segment.objects.filter().order_by('-id')[0]
                    image_id = s_obj.id
                    print("image id = {}".format(image_id))
                    context_dict = {'segment_form': sform, 'image_show':s_image, 'temp_form' : tform}
       
    elif request.is_ajax():
        print("ajax one!")
        # High bar values
        hue_amount_h = request.GET.get('h_value_h')
        saturation_amount_h = request.GET.get('s_value_h')
        value_amount_h = request.GET.get('v_value_h')
        # low bar values
        hue_amount_l = request.GET.get('h_value_l')
        saturation_amount_l = request.GET.get('s_value_l')
        value_amount_l = request.GET.get('v_value_l')

        print("high values")
        print('hue : {} \n saturation: {} \n value : {}\n'.format( hue_amount_h, saturation_amount_h, value_amount_h))
        print("low values")
        print('hue : {} \n saturation: {} \n value : {}\n'.format(hue_amount_l, saturation_amount_l, value_amount_l))

        img_obj = segment.objects.filter().order_by('-id')[0]
        target_image = img_obj.uploads
        orig_img = target_image
        if np.array(PIL.Image.open(target_image)).any() :
            print('Not none in target_image')
        name_image = img_obj.uploads.name

        if flag:
            print("changing Image")
            im = do_segmentation(hue_amount_l, saturation_amount_l, value_amount_l,
                                 hue_amount_h, saturation_amount_h, value_amount_h, target_image,name_image)
            print('Im type : ' + str(type(im)))
            if im:
                s = segment.objects.get(id=image_id)
                s.uploads = im      #overwrite image
                # s.category = category
                s.save()
                img_add = s.uploads.url
                return HttpResponse(img_add)
            else:
                print("Image not available")
                context_dict = {'segment_form': sform, 'temp_form' : tform, 'image_show' : s} 
        else:
                print("ajax request not maintained properly")
                context_dict = {'segment_form': sform, 'temp_form' : tform, 'image_show' : s}

    else:
        sform = forms.segmentForm()
        tform = forms.TempForm({'predictIt' : 'no'})
        flag = 0
        context_dict = {'segment_form': sform, 'temp_form' : tform}
    
    print("final context_dict = {}".format(context_dict))
    return render(request, 'segment.html', context_dict)


def segment_live(request):
    submitbutton = request.POST.get('Submit')
    print(submitbutton)
    if submitbutton:
        context = {'submitbutton' : submitbutton}
    else:
        context = {'submitbutton' : None}
    return render(request, 'live_segment.html', context)