import streamlit as st
import pandas as pd

import time
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
import numpy as np

import plotly.express as px

from src.plot_graph import Methods

import plotly.graph_objects as go

# for text summarisation.


# from gensim.summarization import summarize


st.set_page_config(page_title='Staff Visualizer.', layout='wide',
                   page_icon='https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/160/twitter/322/chart-increasing_1f4c8.png')

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


class PlottingPerformance:

    c1, c2 = st.columns(2)

    def __init__(self, rows, columns, newdf):

        self.rows = rows

        self.columns = columns

        self.newdf = newdf

    def grouped_bar_chart(self, team_avg_row, a):

        with c1:

            x_axis = np.arange(len(a))

            staff_name_label = newdf['team_name'].iloc[0]

            plt.bar(x_axis - 0.2, self.rows, width=0.4, label=staff_name_label)
            plt.bar(x_axis + 0.2, team_avg_row, width=0.4, label='Global')

            plt.xticks(x_axis, a, rotation='vertical')

            plt.legend()

            st.pyplot(plt)

    def double_radar(self, team_avg_row):

        with c2:

            fig = go.Figure()

            fig.add_trace(go.Scatterpolar(
                r=team_avg_row,
                theta=self.columns,
                fill='toself',
                name='team average'
            ))

            fig.add_trace(go.Scatterpolar(
                r=self.rows,
                theta=self.columns,
                fill='toself',
                name='individual average'
            ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 5]
                    )),
                showlegend=False
            )

            st.plotly_chart(fig)

    def matplot(self):

        with c2:

            st.subheader('Bar Chart matplotlib')

            # fig, ax = plt.subplots()

            # c = ['#89CFF0', '#5F9EA0', '#6082B6', 'brown']

            # plt.bar(self.columns, height=self.rows, width=0.7, color=c,edgecolor='black')
            col = []
            for i in self.columns:

                col.append(i)

            chart_data = pd.DataFrame(
                np.random.randn(50, 6),
                columns=["a", "b", "c", "d", "e", "f"])

            st.area_chart(chart_data)


class ReadCsv:

    def __init__(self, df):

        self.df = df

    def readCsv(self):

        df = pd.read_csv(self.df)

        return df


def displayimage(df):

    img = df['profile_images'][0]


def wordcloudimage(li):

    stop_w = set(STOPWORDS)

    un_string = (" ").join(li)

    word_cloud = WordCloud(
        stopwords=stop_w, background_color="white").generate(un_string)

    img = word_cloud.to_image()

    st.image(img)


def get_the_performance_text(tot_avg_rating):

    comments_arr = ['', 'Too Much to Improve', 'Still not Good',
                    'Decent Performance', 'Good performance', 'Very Good Performance']
    if (tot_avg_rating == 1):
        st.subheader(comments_arr[1])
    if (tot_avg_rating == 2):
        st.subheader(comments_arr[2])
    if (tot_avg_rating == 3):
        st.subheader(comments_arr[3])
    if (tot_avg_rating == 4):
        st.subheader(comments_arr[4])
    if (tot_avg_rating == 5):
        st.subheader(comments_arr[5])


def smile_rating_fun(tot_avg_rating):

    st.subheader('Team capability Meter')

    if (tot_avg_rating == 1):
        # st.subheader('Bad Performance')
        st.image('images/angry.png', width=50)
    else:
        st.image('images/angrybw.png', width=50)

    if (tot_avg_rating == 2):
        st.image('images/lessangry.png', width=50)
    else:
        st.image('images/lessangrybw.png', width=50)

    if (tot_avg_rating == 3):
        st.image('images/decent.png', width=50)
    else:
        st.image('images/decentbw.png', width=50)

    if (tot_avg_rating == 4):
        st.image('images/good.png', width=50)
    else:
        st.image('images/goodbw.png', width=50)
    if (tot_avg_rating == 5):
        st.image('images/verygood.png', width=50)
    else:
        st.image('images/verygoodbw.png', width=50)

    get_the_performance_text(tot_avg_rating)


def techometer_ratings(tot_avg_rating):

    st.subheader('Technical capability Meter')
    if (tot_avg_rating == 1):

        st.image('images/techometer_images/1.png', width=50)
    else:
        st.image('images/techometer_images/1bw.png', width=50)

    if (tot_avg_rating == 2):

        st.image('images/techometer_images/2.png', width=50)
    else:
        st.image('images/techometer_images/2bw.png', width=50)

    if (tot_avg_rating == 3):

        st.image('images/techometer_images/3.png', width=50)
    else:
        st.image('images/techometer_images/3bw.png', width=50)

    if (tot_avg_rating == 4):

        st.image('images/techometer_images/4.png', width=50)
    else:
        st.image('images/techometer_images/4bw.png', width=50)
    if (tot_avg_rating == 5):

        st.image('images/techometer_images/5.png', width=50)
    else:
        st.image('images/techometer_images/5bw.png', width=50)


def autofill_search_bar(df):

    df = pd.read_csv('mock_data.csv')

    df = df.drop_duplicates('team_name')

    autofill_list = df['team_name'].tolist()

    a = st.selectbox('Enter / Select team name:', options=autofill_list)

    default_value = df['team_name'].iloc[0]

    return a

    get_the_performance_text(tot_avg_rating)


st.sidebar.title('Staff Visualizer.')


st.sidebar.subheader('Upload Your File')

uploaded_file = st.sidebar.file_uploader('Upload')
st.sidebar.caption('Enter only .csv files')

st.sidebar.subheader('You can dowload the mock_data file from the staff page.')


if uploaded_file is not None:

    readfile = ReadCsv(uploaded_file)

    df = readfile.readCsv()

    team_name_selectbox = autofill_search_bar(df)

    # selecting from dataframe feature.

    # selecting_from_dataframe_feature(df).

    if len(team_name_selectbox) != 0:

        newdf = df[(df['team_name'] == team_name_selectbox)
                   ].drop_duplicates('staff_feedback_for')

        team_name = (newdf['team_name'].iloc[0])

        st.subheader((team_name))

        obj1 = Methods(newdf)

        c3, c4, c5 = st.columns(3)

        with c3:

            # st.image('images/person1.png', width=130)

            avg_of_rating = obj1.avgofperformance('rating')

            # st.write(avg_of_rating)

            st.subheader('Rating')
            for i in range(0, avg_of_rating):

                rating_img_display = 'images/ratingstar.png'

                st.image(rating_img_display, width=20)

        with c4:

            m = st.markdown("""
                    <style>
                    div.stButton > button:first-child {
                        background-color: #4169E1;
                        color:white;
                        height:25px;
                        font-size:14px;
                        display:flex;
                        justify-content:center;
                        
                        
                    }
                    </style>""", unsafe_allow_html=True)

            b = st.button("Fun Person")

            b = st.button("Early Bird")

            b = st.button("Good Coder")

        with c5:

            st.subheader('Word api')

            list1 = []
            for i in newdf.index:

                list1.append((newdf['accomplishments'][i]))

            wordcloudimage(list1)

        c8, c9 = st.columns(2)

        with c8:

            collaboration = obj1.avgofperformance('collaboration')

            teamresponsibility = obj1.avgofperformance(
                'team_objective_responsibility')

            empathy = obj1.avgofperformance('empathy')

            approachable = obj1.avgofperformance('approachable')

            smile_rating = round(
                (teamresponsibility+empathy+approachable+collaboration) / 4)

            # st.write(smile_rating)

            smile_rating_fun(smile_rating)

        with c9:
            avgcode = obj1.avgofperformance('coding_practices')

            delivery = obj1.avgofperformance('delivery')

            independence = obj1.avgofperformance('independence')

            commitment = obj1.avgofperformance('commitment')

            techometer_avg_rating = round(
                (avgcode+delivery+independence+commitment) / 4)

            techometer_ratings(techometer_avg_rating)

        c1, c2 = st.columns(2)

        participation = obj1.avgofperformance('participation')

        ownership = obj1.avgofperformance('ownership')

        personaldevelopement = obj1.avgofperformance('personal_development')

        st.markdown('---')

        rows = [avgcode, empathy, delivery, approachable, participation, independence,
                commitment, ownership, collaboration, teamresponsibility, personaldevelopement]

        columns = ['coding', 'empathy', 'delivery', 'approachable', 'participation', 'independence',
                   'commitment', 'ownership', 'collaboration', 'teamresponsibility', 'personaldevelopement']

        a = ['coding', 'empathy', 'delivery', 'approachable', 'participation', 'independence',
             'commitement', 'ownership', 'collaboration', 'team responsibility', 'personal developement']

        plot = PlottingPerformance(rows, columns, newdf)

        j = st.markdown("""
                    <style>
                    div.stContainer > button:first-child {
                        background-color: #294562;
                        
                    }
                    </style>""", unsafe_allow_html=True)

        suggestions = newdf['suggestions'].values[0]

        with st.container():

            st.info('Suggestions Executive using text summarisation')

            with st.expander('Read All Suggestions.'):

                s = []

                for i in newdf['suggestions']:

                    s.append(i)

                st.write(s)

            # gensim_summarisation(s)

        with st.container():

            st.info('Acheivements Executive Summary')
            st.success('Acheivements')

            with st.expander('Read All Acheivement Summary.'):

                st.write('This is the summary of acheivement.')

        # st.write(newdf['group'][0])

        team_avg_deli = obj1.averages_of_team(df, 'delivery')

        team_avg_code = obj1.averages_of_team(df, 'coding_practices')

        team_avg_empathy = obj1.averages_of_team(df, 'empathy')

        team_avg_approachable = obj1.averages_of_team(df, 'approachable')

        team_avg_independence = obj1.averages_of_team(df, 'independence')

        team_avg_participation = obj1.averages_of_team(df, 'participation')

        team_avg_commitment = obj1.averages_of_team(df, 'commitment')

        team_avg_ownership = obj1.averages_of_team(df, 'ownership')

        team_avg_collaboration = obj1.averages_of_team(df, 'collaboration')

        team_avg_teamresponsibility = obj1.averages_of_team(
            df, 'team_objective_responsibility')

        team_avg_personaldevelopement = obj1.averages_of_team(
            df, 'personal_development')

        team_avg_row = [team_avg_code, team_avg_empathy, team_avg_deli, team_avg_approachable,  team_avg_participation, team_avg_independence, team_avg_commitment,
                        team_avg_ownership,
                        team_avg_collaboration, team_avg_teamresponsibility, team_avg_personaldevelopement]

        st.write(newdf)

        plot.grouped_bar_chart(team_avg_row, a)

        plot.double_radar(team_avg_row)


st.subheader('Staff Visualizer.')
