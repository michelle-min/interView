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

# for viz and data manipulations
import matplotlib.pyplot as plt
import numpy as np
import wave
import pandas as pd
import seaborn as sns

# for NLP
import spacy
import spacy_streamlit as ss
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import word_tokenize
from nltk import FreqDist
from textblob import TextBlob
import neattext as nt

# for prosody
import myprosody
import pickle

nltk.download('punkt')
nltk.download('stopwords')
nlp = spacy.load('en_core_web_sm')

########
#EXECUTION
#######

# Start execution in main() function.
def main():

    # Add a selector for the app mode on the sidebar.
    st.markdown('___')
    st.sidebar.title("Make an impression with InterView")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["", "Why internships", "What to expect", "Record an interview", "Generate feedback", "Make some changes"])
    if app_mode == "":
        add_intro()
    elif app_mode == "Why internships":
        add_internships()
    elif app_mode == "What to expect":
        st.sidebar.success('👆 Record your own answer and get immediate feedback.')
        add_questions()
    elif app_mode == "Record an interview":
        run_asr()
    elif app_mode == "Generate feedback":
        run_analysis()
    elif app_mode == "Make some changes":
        run_nextsteps()

# Add text for the home/instruction page
def add_intro():
    st.markdown('# InterView Demo')
    st.markdown('Learning happens best when content is personalized to meet our needs and strengths. For this reason I created :robot_face: InterView, the AI system to coach you through mock interviews for internships. This site is only a demo of several functionalities.')
    #You can find me on [LinkedIn] (https://www.linkedin.com/in/michellemin-1/) and [GitHub] (https://github.com/michelleymin).
    st.markdown('👈 **Select an option from the sidebar to get started.**')
    st.markdown('*What to expect.* Learn about common interview questions')
    st.markdown('*Record an interview.* Upload your question response audio')
    st.markdown('*Generate feedback.* Get immediate feedback')
    st.markdown('*Make some changes.* Figure out your next steps')

# Add text for the what to expect page
def add_questions():
    st.markdown('# So you\'re looking for an internship?')
    st.markdown('Once you start applying, here are common questions that you can expect in an interview. At any point, you can practice it yourself by selecting "Record your interview" in the left sidebar.')
    st.markdown('___')
    st.markdown('## Tell me about yourself.')
    st.markdown('“Tell me about yourself,” or questions like it, are common at the beginning of interviews as they ease both you and the interviewer into the interview. It allows the interviewer to hear a short, summed up version of your background and skills, and it gives them insight into what experience and qualifications you think are most relevant to the position you’re interviewing for.')
    st.markdown('It’s also not lost on employers that, although a common interview question, it still has the tendency to fluster or stump candidates. By answering this question well, you are setting the tone for the interview as someone who is confident, good under pressure and attentive to the qualifications of the position.')
    st.markdown('Some interviewers might approach this question as an icebreaker by using your response to spark casual conversation to get to know you better, while others may move directly into other interview questions after you respond.')
    st.markdown('Text from [indeed] (https://www.indeed.com/career-advice/interviewing/interview-question-tell-me-about-yourself)')
    st.markdown('___')
    st.markdown('## When is a time you showed leadership?')
    st.markdown('With these kinds of questions, interviewers are usually trying to learn three things: First, they want to know how you behaved in a real-world situation. Second, they want to understand the measurable value you added to that situation. Finally, they are trying to learn how you define something like “pressure at work”—a concept different people might interpret differently.')
    st.markdown('Success in a behavioral interview is all about preparation. There aren’t necessarily wrong answers. These questions are aimed at getting to know the real you. The important thing is to be honest and to practice structuring your responses in a way that communicates what you have to offer.')
    st.markdown('You can use the STAR interview method to prepare for behavioral interviews—a technique that helps you structure your response to behavioral interview questions. Using this method, you create a deliberate story arc that your interviewer can easily follow.')
    st.markdown('Text from [indeed] (https://www.indeed.com/career-advice/interviewing/how-to-prepare-for-a-behavioral-interview)')

# Add text for the why internships page
def add_internships():
    st.markdown('# Why internships?')
    st.markdown('What you learn in school may not always feel relevant to what you\'ll do after you graduate. In fact, research shows that many colleges and universities don\'t prioritize teaching their students the skills that employers want. Work-based learning is a way to bridge that gap.')
    st.markdown('Internships allow you to get hands-on practice for important skills like problem solving and teamwork. They also help you get more job offers with a higher starting salary after graduating and connect you with mentors who can give you job-search advice. They even give you more information to decide on your major and classes while you\'re in school.')

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


# Rhythm of audio file
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

    # Expander
    with st.beta_expander('Why does this matter?'):
        st.write('Varying voice intensity (volume) and pitch (up/down) can signal excitement and engagement, which are important to interviewers.')

# Vocabulary of text file
def calc_vocab(text_file):
    # keywords
    st.markdown('## Top words')
    rm_stopwords = str(nt.TextFrame(text_file).remove_stopwords())
    tokens = word_tokenize(rm_stopwords)
    corpus =nltk.Text(tokens)
    fdist = FreqDist(corpus).most_common(10)
    fdist = pd.Series(dict(fdist))

    # Setting figure, ax into variables
    fig, ax = plt.subplots(figsize=(10,5))
    ## Seaborn plotting using Pandas attributes + xtick rotation for ease of viewing
    all_plot = sns.barplot(x=fdist.index, y=fdist.values, ax=ax)
    st.pyplot(fig)

    # Expander
    st.text("")
    with st.beta_expander('Why does this matter?'):
        st.write('People who use more unique and sophisticated vocabulary are rated as performing better in interviews.')

    # Stop words list
    st.markdown('## Stop words')
    stop_w = nt.TextExtractor(text_file).extract_stopwords()
    st.text(stop_w)
    st.text("")

    # named entity recognition
    docx = nlp(text_file)
    ss.visualize_ner(docx, labels=nlp.get_pipe('ner').labels, show_table=False)

    # Expander
    # with st.beta_expander('Why does this matter?'):
        # st.write('Group pronouns like "we" and "they" are associated with higher interview ratings. However, it\'s important to showcase your accomplishments and individual contributions. Here\'s a sample response:.')


# Mood of text file
def calc_mood_all(text_file):
    # Sentiment analysis
    st.markdown('## Sentiment analysis')
    sia = SentimentIntensityAnalyzer()
    t = sia.polarity_scores(text_file)

    # Output
    if t['neu'] > t['pos'] and t['neu'] > t['neg'] and t['neu'] > 0.85:
        st.markdown("Your response is classified as **Neutral**. :confused:")
    elif t['pos'] > t['neg']:
        st.markdown("Your response is classified as **Positive**. :smiley:")
    elif t['neg'] > t['pos']:
        st.markdown("Your response is classified as **Negative**. :disappointed:")

    st.text("")
    col1, col2 = st.beta_columns(2)
    with col1 :
        st.write("The Positive value of your response is:")
        st.write("The Neutral value of your response is:")
        st.write("The Negative value of your response is:")
    with col2:
        st.write(t['pos'])
        st.write(t['neu'])
        st.write(t['neg'])

    # Expander
    st.text("")
    with st.beta_expander('Why does this matter?'):
        st.write('People who say mostly positive emotion words are rated as performing better in interviews than those who say mostly negative emotion words.')

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


########
#NEXT STEPS
#######

def run_nextsteps():
    st.markdown('Enter your answers below and we\'ll follow up with you later.')
    st.markdown('What did you learn about yourself?')
    st.text_input(' ')
    st.markdown('What will you do differently next time?')
    st.text_input('   ')
    st.markdown('When is your next practice session?')
    st.date_input('')

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
