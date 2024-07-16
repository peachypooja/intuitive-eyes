import cv2  
import numpy as np
import pyautogui
from time import sleep
from PIL import Image  #pip install packages
import os
import ctypes


face_cascade = cv2.CascadeClassifier('C:\\Users\\Mtronics Computers\\Desktop\\haarcascade_frontalface_alt2.xml') 
cap = cv2.VideoCapture(0)   # 0 = main camera , 1 = extra connected webcam and so on.
rec = cv2.face.LBPHFaceRecognizer_create()
#rec = cv2.face.EigenFaceRecognizer_create()


pathz = "C:\\Users\\Mtronics Computers\\Desktop\\eyes"

dir_path = f"{pathz}\\dataSet"


#___make a path named dataSet
try:
    os.mkdir(dir_path)
except OSError:
    print("Folder exists")
else:
    print("created")


#recogizer module

def recog():
    """ Recognizes people from the pretrained .yml file """
    rec.read("C:\\Users\\Mtronics Computers\\Desktop\\eyes\\Intutiveeye.yml")
    id = 0  #set id variable to zero

    font = cv2.FONT_HERSHEY_COMPLEX 
    col = (255, 0, 0)
    strk = 2 

    while True:  #This is a forever loop
        ret, frame = cap.read() #Capture frame by frame 
    
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #change color from BGR to Gray
        faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5)

        #print(faces)
        for(x, y, w, h) in faces:
            #print(x, y, w, h)
            roi_gray = gray[y: y+h, x: x+w]  #region of interest is face

            #*** Drawing Rectangle ***
            color = (255, 0, 0)
            stroke = 2
            end_cord_x = x+w
            end_cord_y = y+h
            cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), color, stroke)

            #***detect
            id, conf = rec.predict(roi_gray)
            #cv2.putText(np.array(roi_gray), str(id), font, 1, col, strk)
            #print(id) #prints the id's
            #id, conf = rec.predict(roi_gray)
            print("Recognized ID:", id, "Confidence:", conf)
        
            #if sees unauthorized person
            
            user32 = ctypes.WinDLL('user32')
            LockWorkStation = user32.LockWorkStation

            # Function to lock the system
            def lock_system():
                # Call the LockWorkStation function to lock the system
                LockWorkStation()

            # Main part of the code
            if id != 1: 
                # Execute lock command
                lock_system() 
            else:
                print("Authorized Person")
            '''if id not in [1, 2, 3]: 
                # Execute lock command
                lock_system()'''
        
    
        cv2.imshow('IntitutiveEye', frame)

        #check if user wants to quit the program (pressing 'q')
        if cv2.waitKey(10) == ord('q'):
            op = pyautogui.confirm("Close the Program 'IntutiveEye'?") 
            if op == 'OK':
                print("Out")
                break
            
                

    cap.release()
    cv2.destroyAllWindows() #remove all windows we have created




#create dataset and train the model
def data_Train():
    sampleNum = 0
    #print("Starting training")
    id = pyautogui.prompt(text="""
    Enter User ID.\n\nnote: numeric data only 1 2 3 etc.""", title='IntutiveEye', default='none')
    #check for user input
    
    #if user input is 1 2 or 3  
    if int(id) < 0:
        pyautogui.alert(text='WRONG INPUT',title='IntutiveEye',button='Back')

    elif int(id) > 0:
        while True:
            ret, img = cap.read()  
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for(x, y, w, h) in faces: #find faces
                sampleNum = sampleNum + 1  #increment sample num till 21
                cv2.imwrite(f'{pathz}\\dataSet\\User.{id}.{sampleNum}.jpg', gray[y: y+h, x: x+w]) #uncomment this
                cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 4)
                cv2.waitKey(100)

            cv2.imshow('faces', img)  #show image while capturing
            cv2.waitKey(1)
            if(sampleNum > 40): #21 sample is collected
                break   
             
    trainer()  #Train the model based on new images

    recog() #start recognizing


#Trainer 
def trainer():
    faces = []   #empty list for faces
    Ids = [] #empty list for IDs

    #path = dir_path
    path = (f'{pathz}\\dataSet')

    #gets image id with path
    def getImageWithID(path):
        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
        #print(f"{imagePaths}\n")
    
        # Iterate over image paths
        for imagePath in imagePaths:
            # Open and convert image to grayscale
            faceImg = Image.open(imagePath).convert('L')
            # Convert image to NumPy array
            faceNp = np.array(faceImg, 'uint8')
            # Extract ID from filename
            filename = os.path.basename(imagePath)
            try:
                ID_str = filename.split('.')[1]  # Extract the part containing the ID as a string
                ID = int(ID_str)  # Convert the string to an integer
                # Append face image and ID to lists
                faces.append(faceNp)
                Ids.append(ID)
            except ValueError as e:
                print(f"Error converting ID for filename '{filename}': {e}")

            cv2.waitKey(10)


        return Ids, faces


    ids, faces = getImageWithID(path)

    print(ids, faces)
    rec.train(faces, np.array(ids))
    #create a yml file at the folder. WIll be created automatically.
    rec.save(f'{pathz}\\Intutiveeye.yml')   
    pyautogui.alert("Done Saving.\nPress OK to continue")
    cv2.destroyAllWindows()



#Options checking
opt =pyautogui.confirm(text= 'Chose an option', title='IntutiveEye', buttons=['START', 'Train', 'Exit'])
if opt == 'START':
    #print("Starting the app")
    recog()
    
if opt == 'Train':
    opt = pyautogui.confirm(text="""
    Please look at the Webcam.\nTurn your head a little while capturing.\nPlease add just one face at a time.
    \nClick 'Ready' when you're ready.""", title='IntutiveEye', buttons=['Ready', 'Cancel'])
        
    if opt == 'Ready':
            #print("Starting image capture + Training")
            data_Train()
    if opt == 'Cancel':
        print("Cancelled")
        recog()
        

if opt == 'Exit':
    print("Quit the app")
    
