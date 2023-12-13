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

🤞 BTS, also known as the Bangtan Boys, is a South Korean band formed in 2010. Comprising Jin, Suga, J-Hope, RM, Jimin, V, and Jungkook, they actively contribute to their music creation. Initially rooted in hip-hop, their musical style has evolved to embrace various genres. Their lyrics delve into themes like mental health, the challenges faced by young people in school, growth, loss, self-acceptance, individuality, and the impact of fame. Additionally, their music draws inspiration from literature, philosophy, and psychology, while their discography includes a storyline set in an alternate universe. The members of the said Korean boy band also show an affinity to various kinds of fast food since they share photos on social media that they enjoying it.
 
![image](https://github.com/harleybelz/GROUP_10_Finals_FaceRecognition/assets/144197127/c5f2b864-b8a5-45c9-bf01-755851af367e)


# 🍕 FACE RECOGNITION OF FAMOUS FAST FOOD CONSUMER 

📖The collection of individuals known for excelling in their respective fields and gaining widespread recognition, also share an affinity for fast food. Through a facial recognition system, the team can identify and label these individuals.

⚪ Kim Taehyung (V)

Kim Tae-hyung, professionally recognized as V, is a South Korean vocalist and a part of the BTS boy band. Since joining the group in 2013, V has released three solo tracks as part of BTS "Stigma" in 2016, "Singularity" in 2018, and "Inner Child" in 2020—all of which achieved chart success on South Korea's Gaon Digital Chart.

📌 For face recognition of famous fast food consumers, codes with an image file name "V.jpg" are used to recognize faces.

![image](https://github.com/harleybelz/GROUP_10_Finals_FaceRecognition/assets/144197127/9bdf1a04-daf0-46ba-b7a7-776c670db6a4)

⚪ Park Jimin

Park Ji-min, is a renowned South Korean singer and dancer known by his stage name, Jimin. He holds the roles of lead vocalist and dancer in the widely acclaimed global music sensation BTS. Initially excelling as a top student in Busan High School's modern dance department, Jimin's path shifted when a teacher suggested he audition for an entertainment company. Signing with Big Hit Entertainment, he made his debut in 2013 as a member of the South Korean boy band BTS.

📌 For face recognition of famous fast food consumers, codes with an image file name "Jimin.jpg" are used to recognize faces.

![image](https://github.com/harleybelz/GROUP_10_Finals_FaceRecognition/assets/144197127/6c790388-9f71-4278-8db5-1ed45e5f9305)

⚪ Suga

SUGA, whose real name is Min Yoon-Gi, is recognized as a skilled rapper in the K-pop group BTS. Beyond his rap talents, he's also a proficient producer and songwriter. His achievements include winning the Best Producer award at MMA in 2017. Before attaining fame, he encountered numerous challenges and hardships. His life journey serves as an inspirational tale, showcasing his resilience and unwavering determination to pursue happiness, eventually realizing his dream and achieving success.

📌 For face recognition of famous fast food consumers, codes with an image file name "Suga.jpg" are used to recognize faces.

![image](https://github.com/harleybelz/GROUP_10_Finals_FaceRecognition/assets/144197127/75cf77b5-8893-495e-a09e-81521c0ad3bf)

⚪ Jungkook

Jeon Jeong-Guk, widely recognized as BTS Jungkook, is a versatile musician hailing from South Korea. Within the renowned group Bangtan Sonyeondan (BTS), he fulfills roles as the lead singer, dancer, and rapper. Since he entered BTS in 2013, Jungkook's impact has transcended South Korea, captivating audiences across Asia through his songs and music videos. Apart from his musical talents, he's celebrated for his appealing personality and striking appearance. Elements such as his distinctive long hair, tattoos, and eyebrow piercing have contributed to his status as a heartthrob, endearing him to numerous fans.

📌 For face recognition of famous fast food consumers, codes with an image file name "Jung Kook.jpg" are used to recognize faces.

![image](https://github.com/harleybelz/GROUP_10_Finals_FaceRecognition/assets/144197127/735a8806-bbe1-4c57-957a-7484039505af)

⚪ J-Hope

Jung Ho-seok, known professionally as J-Hope, is a versatile South Korean artist excelling as a rapper, singer-songwriter, dancer, and record producer. His entry into the South Korean boy band BTS occurred in 2013 under Big Hit Entertainment. In 2018, J-Hope unveiled his inaugural solo mixtape titled "Hope World." Critically acclaimed, it reached noteworthy success by peaking at number 38 on the US Billboard 200, marking the highest chart position achieved by a solo Korean artist at that time.

📌 For face recognition of famous fast food consumers, codes with an image file name "J hope.jpg" are used to recognize faces.

![image](https://github.com/harleybelz/GROUP_10_Finals_FaceRecognition/assets/144197127/65ca9000-248d-4c94-ac14-aacdf2446992)

⚪ RM

Kim Nam-joon, professionally known as RM, is a prominent South Korean rapper, songwriter, and record producer who leads the South Korean boy band BTS. In 2015, RM introduced his inaugural solo mixtape titled "RM," followed by his second mixtape, "Mono," in 2018. "Mono" made history by becoming the highest-charting album by a Korean solo artist in the United States, reaching number 26 on the Billboard 200. RM marked his official solo debut in 2022 with the launch of his studio album "Indigo," featuring collaborations with Erykah Badu and Anderson Paak. This album marked a significant milestone in his solo career.

📌 For face recognition of famous fast food consumers, codes with an image file name "RM.jpg" are used to recognize faces.

![image](https://github.com/harleybelz/GROUP_10_Finals_FaceRecognition/assets/144197127/cfb621e8-138b-4def-9d3a-a720f86078fa)


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

https://en.wikipedia.org/wiki/BTS

https://en.wikipedia.org/wiki/V_(singer)

https://nationaltoday.com/birthday/jimin/

https://www.creatrip.com/en/blog/7138/BTS-SUGA-s-Life-Story

https://www.groovenexus.com/artist-spotlight/singer/bts-jungkook-biography/

https://en.wikipedia.org/wiki/J-Hope

https://en.wikipedia.org/wiki/RM_(musician)




