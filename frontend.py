import streamlit as st
import requests
import jwt  # PyJWT, you might need to install this with pip

# API URL configuration
API_BASE_URL = "http://127.0.0.1:5000"

def main():
    st.title("Wanderly Chat Service")

    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False

    if st.session_state['authenticated']:
        menu = ["Chat with Bot", "User Chat History", "Offers for Users", "Update User Information"]
        choice = st.sidebar.selectbox("Menu", menu)
    else:
        menu = ["Login", "SignUp"]
        choice = st.sidebar.selectbox("Menu", menu)

    if choice == "SignUp":
        sign_up_form()
    elif choice == "Login":
        login_form()
    elif choice == "Update User Information":
        update_user_form()
    elif choice == "Chat with Bot":
        chat_interface()
    elif choice == "User Chat History":
        display_chat_history()
    elif choice == "Offers for Users":
        display_offers()

def sign_up_form():
    with st.form("Register Form", clear_on_submit=True):
        email = st.text_input("Email")
        password = st.text_input("Password", type='password')
        name = st.text_input("Name")
        gender = st.text_input("Gender")
        current_city = st.text_input("Current City")
        age = st.number_input("Age", step=1)
        submit = st.form_submit_button("Register")

        if submit:
            user_data = {
                "email": email,
                "password": password,
                "name": name,
                "gender": gender,
                "current_city": current_city,
                "age": age
            }
            response = requests.post(f"{API_BASE_URL}/register", json=user_data)
            if response.status_code == 201:
                st.success("You have successfully registered.")
            else:
                st.error(f"Failed to register: {response.json().get('error')}")

def login_form():
    with st.form("Login Form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type='password')
        submit = st.form_submit_button("Login")

        if submit:
            login_data = {
                "email": email,
                "password": password
            }
            response = requests.post(f"{API_BASE_URL}/login", json=login_data)
            if response.status_code == 200:
                st.session_state['token'] = response.json().get('access_token')
                st.session_state['user_id'] = response.json().get('user_id')
                st.session_state['authenticated'] = True
                st.success("Logged in successfully.")
            else:
                st.error("Failed to login.")

def update_user_form():
    if 'token' in st.session_state:
        with st.form("Update User Form"):
            new_email = st.text_input("New Email", placeholder="Type Email to update")
            new_password = st.text_input("New Password", type='password', placeholder="Type Password to update")
            new_name = st.text_input("New Name", placeholder="Type Name to update")
            new_gender = st.text_input("New Gender", placeholder="Type Gender to update")
            new_city = st.text_input("New Current City", placeholder="Type City to update")
            new_age = st.number_input("New Age", step=1, min_value=0, placeholder="Type Age to update", format="%d")
            submit = st.form_submit_button("Update")

            if submit:
                update_data = {"user_id": st.session_state['user_id']}  # Start with user_id
                if new_email:
                    update_data["email"] = new_email
                if new_password:
                    update_data["password"] = new_password
                if new_name:
                    update_data["name"] = new_name
                if new_gender:
                    update_data["gender"] = new_gender
                if new_city:
                    update_data["current_city"] = new_city
                if new_age > 0:
                    update_data["age"] = new_age

                headers = {"Authorization": f"Bearer {st.session_state['token']}"}
                response = requests.put(f"{API_BASE_URL}/update_user", json=update_data, headers=headers)
                if response.ok:
                    st.success("User updated successfully.")
                else:
                    st.error(f"Failed to update user: {response.json().get('error')}")

def display_token():
    if 'token' in st.session_state:
        st.text_area("JWT Token", st.session_state['token'], height=100)
        decoded_token = jwt.decode(st.session_state['token'], options={"verify_signature": False})
        st.json(decoded_token)
    else:
        st.error("You need to login to see the token.")

def display_chat_history():
    if 'token' in st.session_state:
        headers = {"Authorization": f"Bearer {st.session_state['token']}"}
        response = requests.get(f"{API_BASE_URL}/get_chat_history?user_id={st.session_state['user_id']}", headers=headers)
        if response.ok:
            chat_history = response.json().get('chat_history')
            st.write("Chat History:", chat_history)
        else:
            st.error("Failed to fetch chat history.")

def display_offers():
    if 'token' in st.session_state:
        headers = {"Authorization": f"Bearer {st.session_state['token']}"}
        response = requests.get(f"{API_BASE_URL}/get_offers?user_id={st.session_state['user_id']}", headers=headers)
        if response.ok:
            offers = response.json().get('offers')
            st.write("Offers:", offers)
        else:
            st.error("Failed to fetch offers.")


def chat_interface():
    if 'token' in st.session_state:
        user_input = st.text_input("You:", key="chat_input")
        send_button = st.button("Send")
        if send_button:
            send_chat_message(user_input)

        # Display chat history
        if st.session_state['chat_history']:
            for msg in st.session_state['chat_history']:
                st.text(msg)
    else:
        st.warning("Please login to chat.")

def send_chat_message(message):
    """Send message to backend and append response to chat history."""
    headers = {"Authorization": f"Bearer {st.session_state['token']}"}
    chat_data = {
        "user_id": st.session_state['user_id'],
        "message": message
    }
    response = requests.post(f"{API_BASE_URL}/chat", json=chat_data, headers=headers)
    if response.ok:
        chat_response = response.json().get('answer')
        st.session_state['chat_history'].append(f"You: {message}")
        st.session_state['chat_history'].append(f"Bot: {chat_response}")
    else:
        st.error("Failed to send message: " + response.json().get('error', ''))

if __name__ == "__main__":
    main()
