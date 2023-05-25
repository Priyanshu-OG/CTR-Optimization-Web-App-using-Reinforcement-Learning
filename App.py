import streamlit as st

# Custom CSS styles
custom_css = """
    <style>
   .stApp {
       background-color: #e5e5f7;
opacity: 0.8;
background-size: 20px 20px;
background-image:  repeating-linear-gradient(0deg, #45c4f7, #45c4f7 1px, #e5e5f7 1px, #e5e5f7);
   }

   .centered-image {
       display: flex;
       justify-content: center;
       align-items: center;
       height: 200px;
       text-align: center;
       color:red
   }

   .project-title {
       font-family: 'Montserrat', sans-serif;
       text-align: center;
       color: red;
       font-size: 25px;
       margin-bottom: 5px;
       text-decoration: none;
   }

   .research-title {
       font-family: 'Poppins', sans-serif;
       text-align: center;
       color: red;
       font-size: 18px;
       border-bottom: 2px solid #FF0000;
       padding-bottom: 5px;
       margin-bottom: 15px;
       text-decoration: none;
   }

   .about-section {
       font-family: 'Poppins', sans-serif;
       text-align: center;
       font-size: 14px;
       color: #333333;
       margin-top: 15px;
       margin-bottom: 15px;
   }
   </style>
"""

# Display custom CSS styles
st.markdown(custom_css, unsafe_allow_html=True)

# Display the project title
st.markdown("<h1 class='project-title'>SOA SYMPOSIUM 2023</h1>", unsafe_allow_html=True)

# Display the logo image
image_path = 'https://www.nuaodisha.com/images/Contents/NuaOdisha-127142-SOA-University.png'
image_width = 200;
st.markdown("<div class='centered-image'><img src='" + image_path + "' width='" + str(image_width) + "'></div>", unsafe_allow_html=True)

# Display the research title
st.markdown("<h2 class='research-title'><strong>Title: Reinforcement Learning: Recent Developments and Real-world Applications<br>(Upper Confidence Bound (UCB) and Thomson Sampling algorithm)</strong></h2>", unsafe_allow_html=True)

# Display the about section
st.markdown("<h4 class='about-section'>Presented by <br> <strong>Ashna Eshika of CSE-C<br>Reg No: 2041005002<br><br>Priyanshu Kumar of CSE-C<br>Reg No: 2041011062<br><br>Supervised By-<br>Dr. Susmita Panda</strong></h4>", unsafe_allow_html=True)