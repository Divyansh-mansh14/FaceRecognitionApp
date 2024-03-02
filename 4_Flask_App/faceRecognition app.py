import face_recognition
import cv2

#Get the webcam #0(the default one, 1,2, etc means other attached cams)
webcam_video_stream = cv2.VideoCapture(0) # 0 means default camera (webcam)

# Load a Sample picture and extract face encodings.

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

#image_to_recognize = face_recognition.load_image_file("images/testing/trump-modi-unknown.jpg")


#intialize the array variable to hold all face locations, encodings and names in the frame
all_face_locations = []
all_face_encodings = []
all_face_names = []

#loop through every frame in the video
while True:
    #get the current frame from the video as an image
    res, current_frame = webcam_video_stream.read()
    
    
    # we are resizing the image because for processing a image by a deep learning system 
    #we donot require a full size image(Complete image with full resolution) 
    #if we will not resize it the computer has to process the complete image and for that 
    #it has to process more pixels and for which it will use huge amount of resources and time
    
    #resize the current frame to 1/4 size to process faster
    #the second parameter is designated size 
    #since we dont have to change the width or height 
    #we are trying to scale down the image so we will keep it (0,0)
    #fx and fy are scale factor
    current_frame_small = cv2.resize(current_frame, (0,0), fx=0.25, fy=0.25)
    
    #detect all faces in the image
    #arguments are image, number_of_times_to_upsample, model
    all_face_locations = face_recognition.face_locations(current_frame_small, number_of_times_to_upsample=2, model = 'hog')
    
    # detect the face encoding for all the faces detected
    all_face_encoding = face_recognition.face_encodings(current_frame_small, all_face_locations)


    #looping through the face locations and the face encoding
    for current_face_location, current_face_encodings in zip(all_face_locations, all_face_encoding):
        #splitting the tuple to get the four position values of current face
        top_pos, right_pos, bottom_pos, left_pos = current_face_location
        
        
        #We are doing it to compensate for reducing the size above
        top_pos = top_pos * 4
        right_pos = right_pos * 4
        bottom_pos = bottom_pos * 4
        left_pos = left_pos * 4
         
        all_matches = face_recognition.compare_faces(known_face_encodings, current_face_encodings)
        #String to hold the labels
        name_of_person = 'Unknown face'
        # check if the all_matches have atleast one items
        #if yes, get the index number of face that is located in the first index of all_matches
        # Get the name corresponding to the index number and save it in name of person
        if True in all_matches:
            first_match_index = all_matches.index(True)
            name_of_person = known_face_names[first_match_index]
             
        cv2.rectangle(current_frame,(left_pos,top_pos),(right_pos,bottom_pos), (255, 0, 0), 2)

        # display the name as text in the image
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame, name_of_person, (left_pos, bottom_pos), font, 0.5, (255, 255, 255), 1)
    
    
    cv2.imshow("Identified Faces", current_frame)
        
    # Press q on keyboard to break the while loop!
    # cv2.waitKey(1) will return a 32 bit key
    # after this we are performing & operation which is basically the masking operation
    # which convert all 28 bits into 0s and all what remains is 4 bit value
    # we did this because ord('q') will return unicode value in the range of 0 - 255
    # so if we compare cv2.waitKey(1) and ord('q') directly we can not do that
    # bcz cv2.waitKey(1) it is a 32 bit value while ord('q') is unicode value ranging 
    # between 0 - 255
    if cv2.waitKey(1) &  0xFF == ord('q'):
        break
# Release webcam resources
webcam_video_stream.release()
cv2.destroyAllWindows()
