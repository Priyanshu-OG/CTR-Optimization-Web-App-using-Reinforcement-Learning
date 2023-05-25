import streamlit as st
import pandas as pd
custom_css = """
    <style>
    .stApp {
       background-color: #e5e5f7;
opacity: 0.8;
background-size: 20px 20px;
background-image:  repeating-linear-gradient(0deg, #45c4f7, #45c4f7 1px, #e5e5f7 1px, #e5e5f7);
    }

    .centered-text {
        text-align: center;
    }

    .font-red {
        color: #FF0000;
    }

    .bordered-section {
        border: 2px solid #000000;
        padding: 10px;
    }
    </style>
"""                                        


def display_reinforcement_learning():
    st.header("What is Reinforcement Learning?")
    st.markdown(
        """
        <p style="font-family: 'Arial Black', sans-serif; color: #3366ff;"><strong>Reinforcement Learning (RL)</strong></p>
        <p style="font-family: 'Calibri', sans-serif; color: #000000;">
        <strong>Reinforcement Learning (RL)</strong> is a branch of machine learning that focuses on how an agent can learn to make sequential decisions 
        in an environment to maximize a cumulative reward. It is used to solve interacting problems where the data observed up to time t is 
        considered to decide which action to take at time (t+1) to maximize our reward.
        </p>
        """,
        unsafe_allow_html=True
    )
    
   
    st.image("https://cdn-gcp.marutitech.com/wp-media/2017/04/RL1.jpg", use_column_width=True)
def display_dataset():
    dataset = pd.read_csv('C:/Users/priya/Desktop/Symposium/dataset.csv')
    st.subheader('OUR DATASET WILL LOOK LIKE')
    st.dataframe(dataset)
    st.write(dataset.describe())


def display_ensemble_learning_example():
    st.header("")
    st.markdown(
        """
        <h2 style="font-family: 'Arial Black', sans-serif; color: #171CD3;"> Maximizing Click-through Rate using Ensemble Learning</h2>
        <p style="font-family: 'Calibri', sans-serif; color: #000000;">
        Let's consider an online advertising platform that wants to maximize the click-through rate (CTR) of ads displayed to users. 
        
        Objective:The code aims to optimize ad selection using 
        an ensemble method that combines the UCB (Upper Confidence Bound) 
        and Thompson Sampling algorithms.
Algorithm Overview:
  

UCB Algorithm: It balances exploration and exploitation by selecting the arm (ad) with the highest Upper Confidence Bound (UCB) value.
 UCB incorporates the average reward and an exploration bonus based on the number of times an arm has been pulled.
<br>


Thompson Sampling Algorithm: It uses a Bayesian approach to explore and exploit arms. Each arm's reward distribution is modeled using a Beta distribution.
Arms are selected based on sampling from these distributions, favoring arms with higher expected rewards.
Ensemble Method:

The code combines the results of the UCB and Thompson Sampling algorithms using a weighted averaging approach.
Equal weights (0.5) are assigned to UCB and Thompson Sampling.
The ensemble method combines the exploration strategies and reward estimates of both algorithms to improve ad selection performance.
Data Preparation:

The code reads the dataset from a CSV file.
The dataset represents ad performance, where each row corresponds to a user and each column corresponds to an ad.
The dataset is converted to a NumPy array for easier manipulation.



Execution:

The code executes the ensemble method by calling the ensemble_method function, passing the number of arms, number of users, and the dataset.
The function returns the history of click-through rates (CTR) and the selected ads.
Visualization:

The code generates a histogram of ad selections using the Matplotlib library.
The histogram shows the number of times each ad was selected throughout the experiment.
The visualization helps analyze the performance of the ensemble method in selecting ads and provides insights into the distribution of ad selections.
</p>
 """,
        unsafe_allow_html=True
    )

st.set_page_config(
    page_title="Reinforcement Learning and Problem Statement and our dataset",
    page_icon=":rocket:",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown(custom_css, unsafe_allow_html=True)



st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ("Reinforcement Learning", "Problem Statement","Our Dataset"))

if selection == "Reinforcement Learning":
    display_reinforcement_learning()
elif selection == "Problem Statement":
    display_ensemble_learning_example()
elif selection=="Our Dataset":
    display_dataset()