import streamlit as st
import hashlib
import pymongo

# MongoDB Connection
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["stock_market_db"]
users_collection = db["user_info"]

# Hash function for password security
def hash_password(password, salt="random_salt"):
    return hashlib.sha256((password + salt).encode()).hexdigest()

# Hide left sidebar (hamburger menu)
st.set_page_config(initial_sidebar_state="collapsed")

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "show_signup" not in st.session_state:
    st.session_state["show_signup"] = False  # Default to Login page

st.title("Stock Market App")

# **Toggle between Login & Signup**
if st.session_state["show_signup"]:
    st.subheader("Sign Up")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Sign Up"):
        # Check if email already exists
        if users_collection.find_one({"email": email}):
            st.error("Email already registered!")
        else:
            hashed_password = hash_password(password)
            users_collection.insert_one({"email": email, "password": hashed_password})
            st.success("Account created successfully! Please log in.")
            st.session_state["show_signup"] = False  # Switch to login

    st.markdown("Already have an account?", unsafe_allow_html=True)
    if st.button("Login Instead"):
        st.session_state["show_signup"] = False
        st.rerun()

else:
    st.subheader("Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user = users_collection.find_one({"email": email})
        if user and user["password"] == hash_password(password):
            st.session_state["logged_in"] = True
            st.session_state["user_email"] = email
            st.success("Login successful! Redirecting...")
            st.rerun()
        else:
            st.error("Invalid credentials!")

    st.markdown("Don't have an account?", unsafe_allow_html=True)
    if st.button("Sign Up Instead"):
        st.session_state["show_signup"] = True
        st.rerun()

# Redirect to the main app if logged in
if st.session_state["logged_in"]:
    st.switch_page("pages/new app.py")
