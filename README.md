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

# 🥪 FACE RECOGNITION ON GROUPS 

📌 For face recognition in groups, codes with an image file name "G1.jpg" are used to recognize known faces.

![image](https://github.com/harleybelz/GROUP_10_Finals_FaceRecognition/assets/144197127/890cdfd4-ff45-47f5-bf6a-4ea39d76cdbd)

📌 For face recognition in groups, codes with an image file name "G2.jpg" are used to recognize known faces.

![image](https://github.com/harleybelz/GROUP_10_Finals_FaceRecognition/assets/144197127/942f5a5e-d6ee-49b7-ba9b-dde5152f9fd7)

📌 For face recognition in groups, codes with an image file name "G3.jpg" are used to recognize known faces.

![image](https://github.com/harleybelz/GROUP_10_Finals_FaceRecognition/assets/144197127/7ca2c2da-6073-408a-9acd-6dd8a9262791)

📌 For face recognition in groups, codes with an image file name "G5.jpg" are used to recognize known faces.

![image](https://github.com/harleybelz/GROUP_10_Finals_FaceRecognition/assets/144197127/0cfc406d-27a0-4ace-91c6-f918cb0053af)

# 🍕 FACE RECOGNITION OF FAMOUS FAST FOOD CONSUMER 

📖The collection of individuals known for excelling in their respective fields and gaining widespread recognition, also share an affinity for fast food. Through a facial recognition system, the team can identify and label these individuals.

![image](https://github.com/harleybelz/GROUP_10_Finals_FaceRecognition/assets/144197127/930c980e-925f-4535-ba80-c4c86b7a2f23)

⚪ LeBron James

Born on December 30, 1984, in Akron, Ohio, is a renowned American basketball player celebrated for his versatile skills. Regarded as one of the most exceptional players in basketball history, he secured NBA championships with the Miami Heat (2012, 2013), Cleveland Cavaliers (2016), and the Los Angeles Lakers (2020). His career soared in 2023 as he surpassed Kareem Abdul-Jabbar's longstanding record to become the NBA's highest all-time scorer, accumulating an impressive 38,387 points.

📌 For face recognition of famous fast food consumers, codes with an image file name "Lebron_James.jpg" are used to recognize faces.

![image](https://github.com/harleybelz/GROUP_10_Finals_FaceRecognition/assets/144197127/8b03552f-90d9-4df9-8df8-4e9b6592b330)

⚪ Barack Obama

Barack Obama served as the 44th President of the United States. His journey embodies the essence of the American narrative rooted in heartland values, raised in a middle-class family, where hard work and education were the pathways to success. He strongly believed that a life of such abundance should be dedicated to serving others.

📌 For face recognition of famous fast food consumers, codes with an image file name "Barack_Obama.jpg" are used to recognize faces.

![image](https://github.com/harleybelz/GROUP_10_Finals_FaceRecognition/assets/144197127/08724313-955c-4553-9e0a-6b211664397c)

⚪ Dwyane Wade

Dwyane Wade, a decorated NBA player with three championship titles, an Olympic Gold medal, and multiple All-Star appearances, has had a remarkable and dynamic career. He became part of the Miami HEAT after being selected as the fifth pick in the 2003 NBA Draft.

📌 For face recognition of famous fast food consumers, codes with an image file name "Dwyane_Wade.jpg" are used to recognize faces.

![image](https://github.com/harleybelz/GROUP_10_Finals_FaceRecognition/assets/144197127/b530ab11-f989-4582-9910-1f8dc61198c4)

⚪ Michael Mosley

Michael Mosley, a British TV journalist, producer, and ex-physician, has been with the BBC since 1985. He's notably recognized for hosting TV shows centered on biology and medicine, along with frequent appearances on The One Show.

📌 For face recognition of famous fast food consumers, codes with an image file name "Michael_Mosley.jpg" are used to recognize faces.

![image](https://github.com/harleybelz/GROUP_10_Finals_FaceRecognition/assets/144197127/21d064d2-762b-4329-8a40-cd968ab53e83)

⚪ Peter Andre

In 1996, Andre experienced a significant breakthrough when his song 'Mysterious Girl' gained widespread success, reaching the second position in the UK Singles Chart and ultimately ranking as the sixth best-selling single of 1996 in the UK. Subsequent singles like 'I Feel You' and 'I'm Your Man' also achieved places in the UK Top 10.

📌 For face recognition of famous fast food consumers, codes with an image file name "Peter_Andre.jpg" are used to recognize faces.

![image](https://github.com/harleybelz/GROUP_10_Finals_FaceRecognition/assets/144197127/d3a6008e-5b39-4307-819b-b7f49422ffec)

⚪ Hayley Tamaddon

Hayley Soraya Tamaddon, a British actress, gained recognition for her portrayals in ITV soap operas, notably as Del Dingle in Emmerdale and Andrea Beckett in Coronation Street. Her triumph came in winning the fifth season of Dancing on Ice on March 28, 2010, alongside her skating partner Daniel Whiston.

📌 For face recognition of famous fast food consumers, codes with an image file name "Hayley_Tamaddon.jpg" are used to recognize faces.

![image](https://github.com/harleybelz/GROUP_10_Finals_FaceRecognition/assets/144197127/f5e14336-cee4-4893-9969-91d28c643c68)

⚪ MP Nadine

Nadine Vanessa Dorries, a British writer and ex-politician, held the position of Secretary of State for Digital, Culture, Media, and Sport from 2021 to 2022. She represented Mid Bedfordshire as a Member of Parliament from 2005 to 2023 under the Conservative Party.

📌 For face recognition of famous fast food consumers, codes with an image file name "MP Nadine.jpg" are used to recognize faces.

# 🌭 FACE RECOGNITION OF UNKNOWN FAST FOOD CONSUMER

📖 The collection of unkown individuals that shares an affinity for fast food. Through a facial recognition system, the team can't identify and label these individuals. 

![image](https://github.com/ejce/GROUP_10_Finals_FaceRecognition/assets/144202790/999a40f5-3859-43f0-afc5-756999feb83f)

📌 For face recognition of unknown fast food consumers, codes with an image file name "U1.jpg" are used to recognize faces.

![image](https://github.com/ejce/GROUP_10_Finals_FaceRecognition/assets/144202790/833d7cfd-863d-480c-b82a-69105007274b)

📌 For face recognition of unknown fast food consumers, codes with an image file name "U2.jpg" are used to recognize faces.

![image](https://github.com/ejce/GROUP_10_Finals_FaceRecognition/assets/144202790/60cbc3d3-13bb-41a0-a68d-c4e339d4db84)

📌 For face recognition of unknown fast food consumers, codes with an image file name "U3.jpg" are used to recognize faces.

![image](https://github.com/ejce/GROUP_10_Finals_FaceRecognition/assets/144202790/7d34b105-302f-4942-8b6c-51766409aafa)

📌 For face recognition of unknown fast food consumers, codes with an image file name "U4.jpg" are used to recognize faces.

![image](https://github.com/ejce/GROUP_10_Finals_FaceRecognition/assets/144202790/5c6b53a5-5987-439d-be1f-14f1beea2dcf)

📌 For face recognition of unknown fast food consumers, codes with an image file name "U5.jpg" are used to recognize faces.

![image](https://github.com/ejce/GROUP_10_Finals_FaceRecognition/assets/144202790/812582fc-8463-44c3-a977-5fbc9d2255e1)

📌 For face recognition of unknown fast food consumers, codes with an image file name "U6.jpg" are used to recognize faces.

![image](https://github.com/ejce/GROUP_10_Finals_FaceRecognition/assets/144202790/6e65b128-2ab1-428d-8a14-ee064271f47e)

📌 For face recognition of unknown fast food consumers, codes with an image file name "U7.jpg" are used to recognize faces.

![image](https://github.com/ejce/GROUP_10_Finals_FaceRecognition/assets/144202790/6fbd188b-80a1-4a58-9b39-5a782b8171f9)

📌 For face recognition of unknown fast food consumers, codes with an image file name "U8.jpg" are used to recognize faces.

![image](https://github.com/ejce/GROUP_10_Finals_FaceRecognition/assets/144202790/28185718-f2c9-4c80-a64f-c2ef9bcb47a5)

📌 For face recognition of unknown fast food consumers, codes with an image file name "U9.jpg" are used to recognize faces.

![image](https://github.com/ejce/GROUP_10_Finals_FaceRecognition/assets/144202790/0ca6149b-59e3-4892-a668-30c25c93a2d5)

📌 For face recognition of unknown fast food consumers, codes with an image file name "U10.jpg" are used to recognize faces.

# REFERENCES

https://www.britannica.com/biography/LeBron-James

https://www.whitehouse.gov/about-the-white-house/presidents/barack-obama/#:~:text=Barack%20Obama%20served%20as%20the,lived%20in%20service%20to%20others.

https://dwyanewade.com/about-dwyane-wade/#:~:text=As%20a%20three%2Dtime%20NBA,of%20the%202003%20NBA%20Draft.

https://en.wikipedia.org/wiki/Michael_Mosley_(broadcaster)

https://pianity.com/peter-andre#:~:text=In%201996%2C%20Andre%20achieved%20breakthrough,UK%20Top%2010%20as%20well.

https://en.wikipedia.org/wiki/Nadine_Dorries

https://en.wikipedia.org/wiki/Hayley_Tamaddon





