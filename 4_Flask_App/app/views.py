import os
import cv2
from app.face_recognition import faceRecognitionPipeline
from app.age_recognition import ageRecognitionPipeline
from flask import render_template, request
import matplotlib.image as matimg
import cv2
import face_recognition

UPLOAD_FOLDER = 'static/upload'

def index():
    return render_template('index.html')


def app():
    return render_template('app.html')


def genderapp():
    if request.method == 'POST':
        f = request.files['image_name']
        filename = f.filename
        # save our image in upload folder
        path = os.path.join(UPLOAD_FOLDER,filename)
        f.save(path) # save image into upload folder
        # get predictions
        pred_image, predictions = faceRecognitionPipeline(path)
        pred_filename = 'prediction_image.jpg'
        cv2.imwrite(f'./static/predict/{pred_filename}',pred_image)
        
        # generate report
        report = []
        for i , obj in enumerate(predictions):
            gray_image = obj['roi'] # grayscale image (array)
            eigen_image = obj['eig_img'].reshape(100,100) # eigen image (array)
            gender_name = obj['prediction_name'] # name 
            score = round(obj['score']*100,2) # probability score
            
            # save grayscale and eigne in predict folder
            gray_image_name = f'roi_{i}.jpg'
            eig_image_name = f'eigen_{i}.jpg'
            matimg.imsave(f'./static/predict/{gray_image_name}',gray_image,cmap='gray')
            matimg.imsave(f'./static/predict/{eig_image_name}',eigen_image,cmap='gray')
            
            # save report 
            report.append([gray_image_name,
                           eig_image_name,
                           gender_name,
                           score])
            
        
        return render_template('gender.html',fileupload=True,report=report) # POST REQUEST
            
    
    
    return render_template('gender.html',fileupload=False) # GET REQUEST

def ageapp():
    if request.method == 'POST':
        f = request.files['image_name']
        filename = f.filename
        # save our image in the upload folder
        path = os.path.join(UPLOAD_FOLDER, filename)
        f.save(path)  # save the image into the upload folder

        # get predictions
        pred_image, predictions = ageRecognitionPipeline(path)
        pred_filename = 'prediction_image.jpg'
        cv2.imwrite(f'./static/predict/{pred_filename}', pred_image)

        # generate report
        report = []
        for i, obj in enumerate(predictions):
            face_image = obj['face_image']  # face image (array)
            estimated_age = obj['estimated_age']  # estimated age

            # save face image in the predict folder
            face_image_name = f'face_{i}.jpg'
            matimg.imsave(f'./static/predict/{face_image_name}', face_image)

            # save report
            report.append([face_image_name, estimated_age])

        return render_template('age.html', fileupload=True, report=report)  # POST REQUEST

    return render_template('age.html', fileupload=False)  # GET REQUEST

# views.py
def facerecognition():
    # Load known faces and encodings
    known_face_encodings = []
    known_face_names = []

    # Add your known faces and encodings
    # For example:
    # known_face_encodings.append(face_recognition.face_encodings(face_recognition.load_image_file("path/to/known_face.jpg"))[0])
    # known_face_names.append("Known Person")

    
    modi_image = face_recognition.load_image_file("images/samples/modi.jpg")
    modi_face_encoding = face_recognition.face_encodings(modi_image)[0]

    trump_image = face_recognition.load_image_file("images/samples/trump.jpg")
    trump_face_encoding = face_recognition.face_encodings(trump_image)[0]

    Divyansh_image = face_recognition.load_image_file("images/samples/Divyansh.jpg")
    Divyansh_face_encoding = face_recognition.face_encodings(Divyansh_image)[0]

    Sahil_image = face_recognition.load_image_file("images/samples/Sahil.jpg")
    Sahil_face_encoding = face_recognition.face_encodings(Sahil_image)[0]

    Amit_image = face_recognition.load_image_file("images/samples/Amit Choudhary.jpg")
    Amit_face_encoding = face_recognition.face_encodings(Amit_image)[0]

    atul_image = face_recognition.load_image_file("images/samples/atul.jpg")
    atul_face_encoding = face_recognition.face_encodings(atul_image)[0]

    Rahul_image = face_recognition.load_image_file("images/samples/Rahul.png")
    Rahul_face_encoding = face_recognition.face_encodings(Rahul_image)[0]

    cheetna_image = face_recognition.load_image_file("images/samples/cheetna.jpg")
    cheetna_face_encoding = face_recognition.face_encodings(cheetna_image)[0]

    Yogesh_image = face_recognition.load_image_file("images/samples/Yogesh.jpeg")
    Yogesh_face_encoding = face_recognition.face_encodings(Yogesh_image)[0]

    Karan_image = face_recognition.load_image_file("images/samples/Karan.jpg")
    Karan_face_encoding = face_recognition.face_encodings(Karan_image)[0]

    known_face_encodings = [modi_face_encoding, trump_face_encoding, Divyansh_face_encoding, Sahil_face_encoding, Amit_face_encoding, atul_face_encoding, Rahul_face_encoding, cheetna_face_encoding, Yogesh_face_encoding, Karan_face_encoding]
    known_face_names = ['Narendra Modi', 'Donald Trump', 'Divyansh Saxena', 'Sahil Saxena', 'Amit Choudhary', 'Atul Tripathi', 'Rahul Johari', 'cheetna Hooda', 'Yogesh Kumar', 'Karan Singh']

    # Get the webcam #0 (the default one, 1, 2, etc. means other attached cams)
    webcam_video_stream = cv2.VideoCapture(0)

    # Initialize variables to hold face locations, encodings, and names
    all_face_locations = []
    all_face_encodings = []
    all_face_names = []

    # Loop through every frame in the video
    while True:
        # Get the current frame from the video as an image
        res, current_frame = webcam_video_stream.read()

        # Resize the current frame to 1/4 size to process faster
        current_frame_small = cv2.resize(current_frame, (0, 0), fx=0.25, fy=0.25)

        # Detect all faces in the image
        all_face_locations = face_recognition.face_locations(current_frame_small, number_of_times_to_upsample=2, model='hog')

        # Detect the face encoding for all the faces detected
        all_face_encodings = face_recognition.face_encodings(current_frame_small, all_face_locations)

        # Loop through the face locations and face encodings
        for current_face_location, current_face_encodings in zip(all_face_locations, all_face_encodings):
            # Check if the face matches any known faces
            all_matches = face_recognition.compare_faces(known_face_encodings, current_face_encodings)

            # String to hold the labels
            name_of_person = 'Unknown face'

            # Check if there is a match
            if True in all_matches:
                first_match_index = all_matches.index(True)
                name_of_person = known_face_names[first_match_index]

            # Draw a rectangle around the face
            top_pos, right_pos, bottom_pos, left_pos = current_face_location
            top_pos = top_pos * 4
            right_pos = right_pos * 4
            bottom_pos = bottom_pos * 4
            left_pos = left_pos * 4
            cv2.rectangle(current_frame, (left_pos, top_pos), (right_pos, bottom_pos), (255, 0, 0), 2)

            # Display the name as text in the image
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(current_frame, name_of_person, (left_pos, bottom_pos), font, 0.5, (255, 255, 255), 1)

        # Display the frame with recognized faces
        cv2.imshow("Face Recognition", current_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release webcam resources
    webcam_video_stream.release()
    cv2.destroyAllWindows()

    return render_template('facerecognition.html')
