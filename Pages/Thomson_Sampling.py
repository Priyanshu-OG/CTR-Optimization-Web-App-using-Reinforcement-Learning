import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

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

# Rest of the code...

# Streamlit app configuration
st.set_page_config(
    page_title="CTR Optimization using Ensemble Method",
    page_icon=":rocket:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Streamlit app styling
st.markdown(custom_css, unsafe_allow_html=True)

# Function to apply Thompson Sampling
def thompson_sampling(dataset, X):
    # Define the number of rounds and ads
    N = X
    d = 10
    
    # Initialize variables
    ads_selected = []
    numbers_of_rewards_1 = [0] * d
    numbers_of_rewards_0 = [0] * d
    total_reward = 0
    noise_factor = 0.2
    ctr_values = []
    
    # Main loop
    for n in range(0, N):
        ad = 0
        max_random = 0
        for i in range(0, d):
            random_beta = np.random.beta(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
            random_beta_with_noise = random_beta + np.random.uniform(-noise_factor, noise_factor)
            if random_beta_with_noise > max_random:
                max_random = random_beta_with_noise
                ad = i
        ads_selected.append(ad)
        reward = dataset.values[n, ad]
        if reward == 1:
            numbers_of_rewards_1[ad] += 1
        else:
            numbers_of_rewards_0[ad] += 1
        total_reward += reward
        ctr_values.append(total_reward / (n + 1))  # Calculate CTR
    
    # Create a histogram of the ad selections
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.hist(ads_selected, bins=d)
    ax1.set_title('Histogram of ads selections')
    ax1.set_xlabel('Ads')
    ax1.set_ylabel('Number of times each ad was selected')
    
    # Create a line chart of CTR
    rounds = range(1, N + 1)
    ax2.plot(rounds, ctr_values)
    ax2.set_title('Click-Through Rate (CTR) over time')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('CTR')
    ax2.set_ylim([0, 1])
    
    # Display the plots in Streamlit
    st.pyplot(fig)

# Streamlit app
def main():
    st.title("Thompson Sampling for Ad Selection")
    st.write("This app applies the Thompson Sampling algorithm to select the best ad based on historical user data.")
    
    dataset = pd.read_csv('C:/Users/priya/Desktop/Symposium/dataset.csv')
    
    X = st.number_input("Enter the number of rows of the dataset you want to use:", min_value=1, max_value=len(dataset))
    X = int(X)
    
    if st.button("Apply Thompson Sampling"):
        thompson_sampling(dataset, X)
    
    st.header("Advantages of Thompson Sampling")
    st.markdown("""
        - Allows for exploration and exploitation simultaneously, providing a balance between trying new ads and exploiting the best performing ads.
        - Adapts to changes in the reward distribution over time, allowing it to quickly adapt to new trends or patterns.
        - Provides a probabilistic guarantee of finding the optimal ad with increasing number of rounds.
    """)
    
    st.header("Disadvantages of Thompson Sampling")
    st.markdown("""
        - Requires knowledge of the reward distribution for each ad, which may not always be available or easy to estimate accurately.
        - Can be computationally expensive if the number of ads or rounds is large.
        - Relies on the assumption of independent and identically distributed (IID) rewards, which may not hold in all scenarios.
    """)
    
    st.header("Mathematical Derivation of Thompson Sampling")
    st.markdown("""
        The Thompson Sampling algorithm uses Bayesian inference to estimate the reward probabilities of each ad and select the best ad based on these estimates. The algorithm can be mathematically derived as follows:
        
        1. Assume a prior distribution for the reward probability of each ad. In Thompson Sampling, a Beta distribution is commonly used as the prior due to its conjugate properties with the Bernoulli distribution.
        
        2. For each round, sample a reward probability for each ad from its respective Beta distribution using the current posterior parameters.
        
        3. Select the ad with the highest sampled reward probability.
        
        4. Observe the actual reward and update the posterior parameters of the selected ad's Beta distribution accordingly.
        
        5. Repeat steps 2-4 for the desired number of rounds.
        
        By updating the posterior parameters based on the observed rewards, the algorithm gradually learns and focuses on the ads with higher expected rewards. The exploration-exploitation trade-off is achieved through the stochastic sampling process, allowing the algorithm to explore less promising ads while exploiting the potentially better-performing ads.
    """)

if __name__ == '__main__':
    main()
