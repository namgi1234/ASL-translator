#if u run this app, type this "streamlit run app.py"
import streamlit as st
import cv2
import math
from ultralytics import YOLO
import speech_recognition as sr

def detect_web(image):
    model = YOLO("best-lite.pt")
    # object classes
    classNames = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
    confidence_data = []
    result = image
    results = model(result, stream=True)

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(result, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            #print("Confidence --->",confidence)
            confidence_data.append(confidence)
            #print(confidence_data)
            # class name
            cls = int(box.cls[0])

            # object details
            org = [x1, y1-10]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            # tag = classNames[cls] + str(confidence)
            cv2.putText(result, classNames[cls]  , org, font, fontScale, color, thickness)
            print(classNames[cls])
    return result

st.set_page_config(
    page_title="ASL translator",
    page_icon="👌",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("ASL translator : 👌")

source_radio = st.sidebar.radio("Select Source",["STT","ASL detection"])

input = None
if source_radio == "STT":
    st.header("Speach to Text")
    lang = st.selectbox("Select language",("en-EN","ko-KR","ja-JP","zh-CN"))
    if st.button("recognize"):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            notice = st.text("Say Something")
            speech = r.listen(source)
        try:
            audio = r.recognize_google(speech, language=lang)
            notice.empty()
            st.code(audio,language='txt')
        except sr.UnknownValueError:
            notice.empty()
            st.code("Your speech can not understand",language='txt')
        except sr.RequestError as e:
            notice.empty()
            st.code("Request Error!; {0}".format(e),language='txt')


elif source_radio =="ASL detection":
    st.header("ASL detection")
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)
    run = st.checkbox('Run')
    while run:
        _, img = camera.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = detect_web(img)
        FRAME_WINDOW.image(img)