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
from nltk import word_tokenize
from nltk import FreqDist
import pandas as pd
import seaborn as sns
from textblob import TextBlob

nltk.download('vader_lexicon')
nltk.download('punkt')
nlp = spacy.load('en_core_web_sm')

# for prosody analysis

# from PIL import Image

########
#EXECUTION
#######

# Start execution in main() function.
def main():

    # Add a selector for the app mode on the sidebar.
    st.markdown('___')
    st.sidebar.title("Make an impression with InterView")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Show instructions", "Record an interview", "Generate feedback"])
    if app_mode == "Show instructions":
        add_intro()
        st.sidebar.success('Record an interview and get immediate feedback.')
    elif app_mode == "Record an interview":
        run_asr()
    elif app_mode == "Generate feedback":
        run_analysis()

# Add intro text for the home/instruction page
def add_intro():
    st.markdown('# InterView Demo')
    st.markdown('Learning happens best when content is personalized to meet our needs and strengths. For this reason I created :robot_face: InterView, the AI system to coach you through mock interviews. This site is only a demo of several functionalities. You can find me on [LinkedIn] (https://www.linkedin.com/in/michellemin-1/) and [GitHub] (https://github.com/michelleymin).')
    st.markdown('👈 **Select an option from the sidebar to get started.**')

########
#ASR
#######

# Automatic speech recognition app when the user selects "Record interview".
def run_asr():
    # @st.cache

    # Create directory to save audio file
    try:
        os.mkdir("temp2")
    except:
        pass

    # Upload audio file
    audio_file = st.file_uploader("Upload audio", type=["wav"])
    if audio_file is not None:

        # Save file to directory
        with open(os.path.join("temp2", audio_file.name), "wb") as f:
            f.write(audio_file.getbuffer())

        # Save file path
        AUDIO_DIR = path.join(path.dirname(path.realpath(__file__)), "temp2")  # obtain the directory to saved file
        AUDIO_PATH = path.join(AUDIO_DIR, f'{audio_file.name}')     # merge path to file to directory

        # Save file to session state (global)
        st.session_state.audio_file = audio_file
        st.session_state.audio_name = f'{audio_file.name}'[:-4]
        st.session_state.audio_path = AUDIO_DIR

        # Load ASR
        r = sr.Recognizer()

        # Connect to Google speech recognizer API
        with sr.AudioFile(AUDIO_PATH) as source:
            audio_text = r.record(source)    # listening to audio to match language

            try:
                text_file = r.recognize_google(audio_text)

                # While wait
                with st.spinner('Analyzing...'):
                   time.sleep(5)
                   st.success('Done!')

                # Save text file to session state (global)
                st.session_state.text_file = text_file

                # Display audio play back and transcript
                st.text("")
                st.markdown('___')
                with st.beta_expander('Here\'s what we found'):
                    st.audio(audio_file)
                    st.write(text_file)

            # Failed to connect
            except:
                st.error('Sorry... try one more time!')

########
#ANALYZE
#######

# Fluency of audio file
def calc_rhythm(audio_file):
    wav = wave.open(audio_file, 'rb')  # open wave file using the wave library
    raw = wav.readframes(-1)        # reading the entire wave audio frame which returns the frame as a byte object
    raw = np.frombuffer(raw, "int16")  # we use numpy to convert audio bytes into an array
    sample_rate = wav.getframerate()

    Time = np.linspace(0, len(raw)/sample_rate, num=len(raw))
    fig, ax = plt.subplots()
    plt.plot(Time, raw, color='blue')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Amplitude')
    ax.set_title('')
    plt.tight_layout()
    st.pyplot(fig)

# Vocabulary of text file
def calc_vocab(text_file):

    # keywords
    st.markdown('## Top words')
    tokens = word_tokenize(text_file)
    corpus =nltk.Text(tokens)
    fdist = FreqDist(corpus).most_common(10)
    fdist = pd.Series(dict(fdist))

    # Setting figure, ax into variables
    fig, ax = plt.subplots(figsize=(10,5))
    ## Seaborn plotting using Pandas attributes + xtick rotation for ease of viewing
    all_plot = sns.barplot(x=fdist.index, y=fdist.values, ax=ax)
    st.pyplot(fig)


    # named entity recognition
    docx = nlp(text_file)
    ss.visualize_ner(docx, labels=nlp.get_pipe('ner').labels, show_table=False)

# Mood of text file
def calc_mood_all(text_file):
    # Sentiment analysis
    st.markdown('## Sentiment analysis')
    sia = SentimentIntensityAnalyzer()
    t = sia.polarity_scores(text_file)

    # Output
    if t['neu'] > t['pos'] and t['neu'] > t['neg'] and t['neu'] > 0.85:
        st.markdown("Speech Text is classified as **Neutral**. :confused:")
    elif t['pos'] > t['neg']:
        st.markdown("Speech Text is Classified as **Positive**. :smiley:")
    elif t['neg'] > t['pos']:
        st.markdown("Speech Text is Classified as **Negative**. :disappointed:")

    st.text("")
    col1, col2 = st.beta_columns(2)
    with col1 :
        st.write("The Positive value of your speech is:")
        st.write("The Neutral value of your speech is:")
        st.write("The Negative value of your speech is:")
    with col2:
        st.write(t['pos'])
        st.write(t['neu'])
        st.write(t['neg'])

    # COMMENT


def calc_mood_sent(text_file):
    st.text("")

    # Sentiment across each sentence in the text
    sents = word_tokenize(text_file)
    entireText = TextBlob(text_file)
    sentScores = [] #storing sentences in a list to plot
    for sent in sents:
        text = TextBlob(sent) #sentiment for each sentence
        score = text.sentiment[0] #extracting polarity of each sentence
        sentScores.append(score)

    #Plotting sentiment scores per sentencein line graph
    st.line_chart(sentScores) #using line_chart st call to plot polarity for each sentenc


# Text analysis app when user selects "Generate feedback"
def run_analysis():
    st.sidebar.subheader('Take a closer look :bulb:')
    model_select = st.sidebar.radio('',['vocabulary','rhythm','mood'])

    if st.session_state.text_file is not None and st.session_state.audio_file is not None:
        if model_select == "vocabulary":
            calc_vocab(st.session_state.text_file)
        elif model_select == "rhythm":
            calc_rhythm(st.session_state.audio_file)
        elif model_select == "mood":
            calc_mood_all(st.session_state.text_file)
            calc_mood_sent(st.session_state.text_file)
    else:
        st.error('Please reload your audio file.')

########
#MISC
#######

# Delete files to avoid overloading app
def remove_wav_files(n):
    wav_files = glob.glob("temp2/*wav")
    if len(wav_files) != 0:
        now = time.time()
        n_days = n * 86400
        for f in wav_files:
            if os.stat(f).st_mtime < now - n_days:
               os.remove(f)
               print("Deleted", f)

remove_wav_files(7)   # remove wav files from directory


if __name__ == '__main__':
    main()
