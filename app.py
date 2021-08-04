
# InterView: mock interview coaching

##########
#lIBRARIES
##########

import streamlit as st
import streamlit.components.v1 as components
#Summarization imports
from gensim.summarization import summarize
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import readtime
import textstat
from textblob import TextBlob
#NER and imports
import spacy
from spacy import displacy
# import en_core_web_sm
# nlp = en_core_web_sm.load()
#rhythm
#import myprosody as mysp
#SR
#import speechrecognition as sr
#
# ########
# #HEADER
# #######
#
# st.set_page_config(page_title="InterView")
#
# st.markdown("<h1 style='text-align: center; color:orange;'>InterView</h1>", unsafe_allow_html=True)
# st.markdown("<h4 style='text-align: center; color:grey;'>AI-powered coaching</h4>", unsafe_allow_html=True)
# #st.markdown("<h1 style='text-align: center; color: white;'>Generate. Summarize. Paraphrase. Measure.</h1>", unsafe_allow_html=True)
#
# # """
# # [![Star](https://img.shields.io/github/stars/dlopezyse/Synthia.svg?logo=github&style=social)](https://gitHub.com/dlopezyse/Synthia)
# # &nbsp[![Follow](https://img.shields.io/twitter/follow/lopezyse?style=social)](https://www.twitter.com/lopezyse)
# # """
# # st.write('')
# # st.write(':point_left: Use the menu at left to select your task (click on > if closed).')
#
# st.markdown('___')
#
# #########
# #SIDEBAR
# ########
#
# st.sidebar.header('Take a closer look :bulb:')
# nav = st.sidebar.radio('',['fluency', 'rhythm', 'vocabulary', 'profile'])
# st.sidebar.write('')
# st.sidebar.write('')
# st.sidebar.write('')
#
# # UPLOAD
# with st.sidebar.beta_expander('Upload File'):
#     uploaded_file = st.file_uploader('',type=".wav")
#
#
# #ABOUT
# ######
# expander = st.sidebar.beta_expander('About')
# expander.write("Learning happens best when content is personalized to meet our needs and strengths. For this reason I created InterView :robot_face:, the AI system to coach you through mock interviews (this site is only a demo of several functionalities). You can find me on [LinkedIn] (https://www.linkedin.com/in/michellemin-1/) and [GitHub] (https://github.com/michelleymin).")
#
#
# #########
# #FEATURE EXTRACTION
# ########
#
# #FLUENCY
# #########
# text = st.text_input('Enter text')
#
# if nav == 'fluency':
#     #Sentiment Analysis
#
#     #Creating graph for sentiment across each sentence in the text inputted
#     sents = sent_tokenize(text)
#     entireText = TextBlob(text)
#     sentScores = [] #storing sentences in a list to plot
#     for sent in sents:
#         text = TextBlob(sent) #sentiment for each sentence
#         score = text.sentiment[0] #extracting polarity of each sentence
#         sentScores.append(score)
#
#     #Plotting sentiment scores per sentencein line graph
#     st.line_chart(sentScores) #using line_chart st call to plot polarity for each sentenc
#
#     #Polarity and Subjectivity of the entire text inputted
#     sentimentTotal = entireText.sentiment
#     st.write("The sentiment of the overall text below.")
#     st.write(sentimentTotal)
#
# #-----------------------------------------
#
# #RHYTHM
# ##########
#
# if nav == 'rhythm':
#     st.markdown("<h3 style='text-align: left; color:#F63366;'><b>Rhythm<b></h3>", unsafe_allow_html=True)
#     st.text('')
#
# # # STT
# #     if exists(file_uploader):
# #         r = sr.Recognizer()
# #         STT = sr.AudioFile(uploaded_file)
# #         with STT as source:
# #             audio = r.record(source)
#
# # Prosody
#     # p = "Title"
#     # c = uploaded_file
#
#     #mysp.myspgend(p,c)
#
#
#
# #-----------------------------------------
#
# #VOCABULARY
# ###########
#
# if nav == 'vocabulary':
#
#     st.markdown("<h3 style='text-align: left; color:#F63366;'><b>Keywords<b></h3>", unsafe_allow_html=True)
#     st.text('')
#
# # Named Entity Recognition
#     #Getting Entity and type of Entity
#     entities = [] #list for all entities
#     entityLabels = [] #list for type of entities
#     doc = nlp(text) #this call extracts all entities, make sure the spacy en library is loaded
#     #iterate through all entities
#     for ent in doc.ents:
#         entities.append(ent.text)
#         entityLabels.append(ent.label_)
#     entDict = dict(zip(entities, entityLabels)) #Creating dictionary with entity and entity types
#
#     #Function to take in dictionary of entities, type of entity, and returns specific entities of specific type
#     def entRecognizer(entDict, typeEnt):
#         entList = [ent for ent in entDict if entDict[ent] == typeEnt]
#         return entList
#
#     #Using function to create lists of entities of each type
#     entOrg = entRecognizer(entDict, "ORG")
#     entCardinal = entRecognizer(entDict, "CARDINAL")
#     entPerson = entRecognizer(entDict, "PERSON")
#     entDate = entRecognizer(entDict, "DATE")
#     entGPE = entRecognizer(entDict, "GPE")
#
#     #Displaying entities of each type
#     st.write("Organization Entities: " + str(entOrg))
#     st.write("Cardinal Entities: " + str(entCardinal))
#     st.write("Personal Entities: " + str(entPerson))
#     st.write("Date Entities: " + str(entDate))
#     st.write("GPE Entities: " + str(entGPE))
#
# # Profile
#     input_me = st.text_area("Input some text in English, and scroll down to analyze it", max_chars=5000)
#
#     if st.button('Measure'):
#         if input_me =='':
#             st.error('Please enter some text')
#         elif len(input_me) < 500:
#             st.error('Please enter a larger text')
#         else:
#             with st.spinner('Wait for it...'):
#                 nltk.download('punkt')
#                 rt = readtime.of_text(input_me)
#                 tc = textstat.flesch_reading_ease(input_me)
#                 tokenized_words = word_tokenize(input_me)
#                 lr = len(set(tokenized_words)) / len(tokenized_words)
#                 lr = round(lr,2)
#                 st.text('Reading Time')
#                 st.write(rt)
#                 st.text('Text Complexity (score from 0 (hard to read), to 100 (easy to read))')
#                 st.write(tc)
#                 st.text('Lexical Richness (distinct words over total number of words)')
#                 st.write(lr)
#
# #-----------------------------------------
#
# #DASHBOARD
# ########
#
# if nav == 'profile':
#     st.markdown("<h3 style='text-align: left; color:#F63366;'><b>Dashboard<b></h3>", unsafe_allow_html=True)
#     st.text('')
#
# #Text Summarization
#     st.markdown("<h3 style='text-align: left; color:#F63366;'><b>Summary<b></h3>", unsafe_allow_html=True)
#     summWords = summarize(text)
#     st.write(summWords)
#
#     input_su = st.text_area("Write some text or copy & paste so we can summarize it (minimum = 1000 characters)", max_chars=5000)
#
#     if st.button('Summarize'):
#         if input_su =='':
#             st.error('Please enter some text')
#         elif len(input_su) < 1000:
#             st.error('Please enter a larger text')
#         else:
#             with st.spinner('Wait for it...'):
#                 st.success(summarize(input_su, word_count=50, ratio=0.05))
#
#
#
# ####################################
