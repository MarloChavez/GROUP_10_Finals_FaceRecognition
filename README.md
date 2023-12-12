# 🍟 GROUP10_FINALS_FACE RECOGNITION 🍔
![image](https://github.com/harleybelz/GROUP_10_Finals_FaceRecognition/assets/144197127/9b5372f4-02c2-477c-a6b8-4c0556392329)
 
The fast food industry thrives as a vibrant sector renowned for its swift eateries offering affordable, convenient, and immediately consumable meals. Emerging in the mid-1900s, this realm has expanded into a worldwide sensation, catering to an array of tastes globally. Identified by its prompt service, standardized menus, and frequently franchised setups, these establishments serve a variety of popular dishes, ranging from burgers, fries, and pizzas to sandwiches, fried chicken, and diverse cultural cuisines. While lauded for its convenience and speed, it remains a substantial presence in the global culinary scene, continuously adapting to shifting consumer preferences and embracing advancements in food technology. 

The team's job is to create a face recognition system related to food. They'll be identifying well-known figures who enjoy various fast foods and regular folks who also indulge in fast food

Shown below are the codes used by the group for this Face Recognition task:
# 👉 IMPORTING IMAGES AND INSTALLING FACE RECOGNITION
    !git clone https://github.com/MarloChavez/GROUP_10_Finals_FaceRecognition.git
    !pip install face_recognition
    %cd GROUP_10_Finals_FaceRecognition

# 👉 ENCODING PROFILES USING KNOWN FACE IMAGES
    import face_recognition
    import numpy as np
    from google.colab.patches import cv2_imshow
    import cv2

    # Creating the encoding profiles
    face_1 = face_recognition.load_image_file("Barack_Obama.jpg")
    face_1_encoding = face_recognition.face_encodings(face_1)[0]

    face_2 = face_recognition.load_image_file("Dwyane_Wade.jpg")
    face_2_encoding = face_recognition.face_encodings(face_2)[0]

    face_3 = face_recognition.load_image_file("Hayley_Tamaddon.jpg")
    face_3_encoding = face_recognition.face_encodings(face_3)[0]

    face_4 = face_recognition.load_image_file("Lebron_James.jpg")
    face_4_encoding = face_recognition.face_encodings(face_4)[0]

    face_5 = face_recognition.load_image_file("MP Nadine.jpg")
    face_5_encoding = face_recognition.face_encodings(face_5)[0]

    face_6 = face_recognition.load_image_file("Michael_Mosley.jpg")
    face_6_encoding = face_recognition.face_encodings(face_6)[0]

    face_7 = face_recognition.load_image_file("Peter_Andre.jpg")
    face_7_encoding = face_recognition.face_encodings(face_7)[0]

    known_face_encodings = [
                        face_1_encoding,
                        face_2_encoding,
                        face_3_encoding,
                        face_4_encoding,
                        face_5_encoding,
                        face_6_encoding,
                        face_7_encoding
    ]

    known_face_names = [
                    "Barack Obama",
                    "Dwyane Wade",
                    "Hayley Tamaddon",
                    "Lebron James",
                    "MP Nadine",
                    "Michael Mosley",
                    "Peter Andre",
    ]
    
# 👉 RUNNING OF FACE RECOGNITION ON KNOWN FAST FOOD CONSUMER
📌Facial recognition works by using machine learning libraries like OpenCV to create models that can identify faces. This usually starts with gathering a set of labeled images 
   featuring faces and preparing these images to ensure they have a consistent structure and highlight facial features. By utilizing this, the group can recognize known and unknown faces 
   of people consuming fast food. 

📌 For face recognition of known and unknown faces our group used the code below: 

    file_name = "    "
    unknown_image = face_recognition.load_image_file(file_name)
    unknown_image_to_draw = cv2.imread(file_name)

    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    for (top,right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

    name = "Unknown"

     face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
    name = known_face_names[best_match_index]
    cv2.rectangle(unknown_image_to_draw, (left, top), (right, bottom),(0,255,0),3)
    cv2.putText(unknown_image_to_draw,name, (left -80, top), cv2.FONT_HERSHEY_SIMPLEX,1,(55,255,255),2, cv2.LINE_AA)

    cv2_imshow(unknown_image_to_draw)

 






