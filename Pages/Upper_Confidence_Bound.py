import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import random

custom_css = """
    <style>
    .stApp {
        background-color: #e5e5f7;
        opacity: 0.8;
        background-size: 20px 20px;
        background-image: repeating-linear-gradient(0deg, #45c4f7, #45c4f7 1px, #e5e5f7 1px, #e5e5f7);
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


st.set_page_config(
    page_title="CTR Optimization using Ensemble Method",
    page_icon=":rocket:",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown(custom_css, unsafe_allow_html=True)

def upperconfidence(dataset, X):
   
    N = X
    d = 10

   
    ads_selected = []
    numbers_of_selections = [0] * d
    sums_of_rewards = [0] * d
    total_reward = 0
    noise_factor = 0.2
    ctr_values = []

    # Main loop
    for n in range(0, N):
        ad = 0
        max_upper_bound = 0
        for i in range(0, d):
            if numbers_of_selections[i] > 0:
                average_reward = sums_of_rewards[i] / numbers_of_selections[i]
                delta_i = math.sqrt(3 / 2 * math.log(n + 1) / numbers_of_selections[i])
                noise = random.uniform(-noise_factor, noise_factor)
                upper_bound = average_reward + delta_i + noise
            else:
                upper_bound = 1e400
            if upper_bound > max_upper_bound:
                max_upper_bound = upper_bound
                ad = i
        ads_selected.append(ad)
        numbers_of_selections[ad] += 1
        reward = dataset.values[n, ad]
        sums_of_rewards[ad] += reward
        total_reward += reward
        ctr_values.append(total_reward / (n + 1))  # Calculate CTR

    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.hist(ads_selected)
    ax1.set_title('Histogram of ads selections')
    ax1.set_xlabel('Ads')
    ax1.set_ylabel('Number of times each ad was selected')

   
    rounds = range(1, N + 1)
    ax2.plot(rounds, ctr_values)
    ax2.set_title('Click-Through Rate (CTR) over time')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('CTR')
    ax2.set_ylim([0, 1])

    
    st.pyplot(fig)


def main():
    st.title("Upper Confidence Bound (UCB) for Ad Selection")
    st.write("This app applies the UCB algorithm to select the best ad based on historical user data.")

    dataset = pd.read_csv('C:/Users/priya/Desktop/Symposium/dataset.csv')

    X = st.number_input("Enter the number of rows of the dataset you want to use:", min_value=1, max_value=len(dataset))
    X = int(X)

    if st.button("Apply Upper Confidence Bound"):
        upperconfidence(dataset, X)

    st.header("Advantages of Upper Confidence Bound (UCB)")
    st.markdown("""
        - Provides a balance between exploration and exploitation, allowing the algorithm to try new ads while exploiting the best-performing ads.
        - Adapts to changes in the reward distribution over time, allowing it to quickly adjust to new trends or patterns.
        - Guarantees logarithmic regret, meaning that the cumulative regret of the algorithm grows at most logarithmically with the number of rounds.
    """)

    st.header("Disadvantages of Upper Confidence Bound (UCB)")
    st.markdown("""
        - Requires knowledge of the upper bound on the rewards, which may not always be available or easy to estimate accurately.
        - Can be sensitive to noise in the reward observations, leading to suboptimal decisions.
        - May require a larger number of rounds to converge to the optimal solution compared to other algorithms.
    """)

    st.header("Mathematical Formulation of Upper Confidence Bound (UCB)")
    st.latex(r'''
        \text{Upper Confidence Bound (UCB)} = \bar{X}_i + \sqrt{\frac{3}{2} \cdot \frac{\ln(n+1)}{N_i}}
    ''')


if __name__ == '__main__':
    main()
