# InterView
# an app for mock interview coaching
# Author: Michelle Min (michelleymin@gmail.com)

##########
#LIBRARIES
##########

import streamlit as st
import time
import glob
import os
from os import path

# for audio transcription
import speech_recognition as sr

# for visualizing and data manipulations
import matplotlib.pyplot as plt
import numpy as np
import wave

# for NLP tasks
import spacy
import spacy_streamlit as ss
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

from PIL import Image

########
#EXECUTION
#######

# disable warnings
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_option('deprecation.showfileUploaderEncoding', False)

#File downloader and animation.
def download_file(file_path):
    # Don't download the file twice. (If possible, verify the download using the file length.)
    if os.path.exists(file_path):
        if "size" not in EXTERNAL_DEPENDENCIES[file_path]:
            return
        elif os.path.getsize(file_path) == EXTERNAL_DEPENDENCIES[file_path]["size"]:
            return

    # Two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % file_path)
        progress_bar = st.progress(0)
        with open(file_path, "wb") as output_file:
            with urllib.request.urlopen(EXTERNAL_DEPENDENCIES[file_path]["url"]) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning("Downloading %s... (%6.2f/%6.2f MB)" %
                        (file_path, counter / MEGABYTES, length / MEGABYTES))
                    progress_bar.progress(min(counter / length, 1.0))

    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()


#img = Image.open('images.jpeg')
st.title('Text to Speech/AudioSpeech to Text Analytic Web App')
#st.image(img, width=650)
st.subheader("Navigate to side bar to see more options")

# re-configuring page layout to restrict users from overwriting the app configuraion

hide_streamlit_style = '''
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
'''
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.sidebar.title("Menu")
analyze = st.sidebar.selectbox(
        '', ["Home", "Audio2Text_Analytics"], index=0)


def main():

# ------- audio to text ----------------------------------------------------

    if analyze == "Audio2Text_Analytics":

        try:
            os.mkdir("temp2")   # create directory to save our audio file to work on
        except:
            pass
        st.markdown('## Upload a wav Audio File')
        audio_file = st.file_uploader("Choose an audio file to upload", type=["wav"])
        if audio_file is not None:
            st.audio(audio_file)  # enabling users to play their audio file
            with open(os.path.join("temp2", audio_file.name), "wb") as f:  # saving file to a directory
                f.write(audio_file.getbuffer())

            st.success("File Saved : {} in temp2".format(audio_file.name))

            r = sr.Recognizer()
            AUDIO_DIR = path.join(path.dirname(path.realpath(__file__)), "temp2")  # obtain the directory to saved file
            AUDIO_PATH = path.join(AUDIO_DIR, f'{audio_file.name}')     # merge path to file to directory
            with sr.AudioFile(AUDIO_PATH) as source:
               audio_text = r.record(source)    # listening to audio to match language

            # recognize_method will throw up request error if the google API is unreachable, we capture this error using try and except handling
               try:
                # using google speech recognizer
                   text_file = r.recognize_google(audio_text)

                   if st.checkbox("Look up your transcribed audio to text here..."):
                      st.write(text_file)

               except:
                    st.write('Sorry...Try Again!')

            st.markdown('## Explore Different Categories of your Audio Speech')
            fe = st.radio(label="Feature Extraction", options=(' ', 'NER of SpeechText', 'Display AudioSpeech Signal', 'Analyze Speech Sentiment'))
            if fe == "Display AudioSpeech Signal":
                wav = wave.open(AUDIO_PATH, 'rb')  # open wave file using the wave library
                raw = wav.readframes(-1)        # reading the entire wave audio frame which returns the frame as a byte object
                raw = np.frombuffer(raw, "int16")  # we use numpy to convert audio bytes into an array
                sample_rate = wav.getframerate()

                Time = np.linspace(0, len(raw)/sample_rate, num=len(raw))
                fig, ax = plt.subplots()
                plt.plot(Time, raw, color='blue')
                ax.set_xlabel('Time (seconds)')
                ax.set_ylabel('Amplitude')
                ax.set_title('Input Audio Signal')
                plt.tight_layout()
                st.pyplot(fig)

            if fe == "NER of SpeechText":
                text_file = text_file
                nlp = spacy.load('en_core_web_sm')
                docx = nlp(text_file)
                if st.button("SpeechText Attributes"):
                   ss.visualize_tokens(docx)
                   ss.visualize_ner(docx, labels=nlp.get_pipe('ner').labels)

            if fe == "Analyze Speech Sentiment":
                sia = SentimentIntensityAnalyzer()
                t = sia.polarity_scores(text_file)
                if st.button("Predict"):
                    st.write("Neutral, Positive and Negative value of your speech is:")
                    st.write(t['neu'], t['pos'], t['neg'])
                    if t['neu'] > t['pos'] and t['neu'] > t['neg'] and t['neu'] > 0.85:
                        st.markdown("Speech Text is classified as **Neutral**. :confused:")
                        st.balloons()
                    elif t['pos'] > t['neg']:
                        st.markdown("Speech Text is Classified as **Positive**. :smiley:")
                        st.balloons()
                    elif t['neg'] > t['pos']:
                        st.markdown("Speech Text is Classified as **Negative**. :disappointed:")


    st.sidebar.markdown(
            """
     ----------
    ## Project Overview
    This is an AI web app that transcribes text to speech in 6 different languages and speech to text, to extract specific features in the
    speech like NER, Sentiment and the Audio signal time frame.

    """)

    st.sidebar.header("")  # initialize empty space

    st.sidebar.markdown(
    """
    ----------
    ## Instructions
    1. For your text2Speech, select your input text language and your output speech language. Then use the convert button.
    2. Upload your wav audio file to begin your speech2text analyses
    3. If your file does not upload, make sure it's a mono-channel file, i.e having only one audio source.
    4. If your file isn't wav formated, do not worry, [click here](https://www.movavi.com/support/how-to/how-to-convert-music-to-wav.html)
       and head over to "Online Converter" to upload your mp3 file.
    5. if you are getting "Sorry...Run Again" message when you upload file, try running again when you have a stable
       network

    """)

     # preview app demo
    demo = st.sidebar.checkbox('App Demo')
    if demo == 1:
       st.sidebar.video('https://res.cloudinary.com/dfgg73dvr/video/upload/v1624127072/ezgif.com-gif-maker_k56lry.mp4', format='mp4')

    st.sidebar.header("")

    st.sidebar.markdown(
    """
    -----------
    # Let's connect

    [![Jude Leonard Ndu](https://img.shields.io/badge/Author-@JudeLeonard-gray.svg?colorA=gray&colorB=dodgergreen&logo=github)](https://www.github.com/judeleonard/)

    [![Jude Ndu](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logoColor=white)](https://www.linkedin.com/in/jude-ndu-78ab38175/)

    [![Jude Leonard](https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=gray)](https://www.twitter.judeleonard13/)
    """)

     #----- deleting files from directories so we don't overload the app------
    def remove_wav_files(n):
        wav_files = glob.glob("temp2/*wav")
        if len(wav_files) != 0:
            now = time.time()
            n_days = n * 86400
            for f in wav_files:
                if os.stat(f).st_mtime < now - n_days:
                   os.remove(f)
                   print("Deleted", f)

    def remove_mp3_files(n):
        mp3_files = glob.glob("temp/*mp3")
        if len(mp3_files) != 0:
            now = time.time()
            n_days = n * 86400
            for f in mp3_files:
                if os.stat(f).st_mtime < now - n_days:
                   os.remove(f)
                   print("Deleted", f)


    remove_mp3_files(7)   # remove mp3 files from directory

    remove_wav_files(7)   # remove wav files from directory




if __name__ == '__main__':
    main()
