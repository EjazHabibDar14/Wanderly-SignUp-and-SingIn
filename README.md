
# Wanderly SignUp and SignIn Application

This project includes both backend and frontend components to demonstrate a functional sign-up and sign-in system. It utilizes Python for the backend and Streamlit for the frontend interface.

## Getting Started

Follow these instructions to get the project running on your local machine for development and testing purposes.

#### 1. Clone the Repository

Clone this repository to your local machine using the following command:

```bash
git clone https://github.com/EjazHabibDar14/Wanderly-SignUp-and-SingIn.git
cd Wanderly-SignUp-and-SingIn
```

#### 2. Set Up a Virtual Environment

Create and activate a virtual environment to manage the project's dependencies:

- **Create the virtual environment:**

```bash
python -m venv myenv
```

- **Activate the virtual environment:**

  - On Windows:
  
  ```bash
  myenv\Scripts\activate
  ```

  - On macOS and Linux:
  
  ```bash
  source myenv/bin/activate
  ```

#### 3. Install Dependencies

Install the required Python packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

#### 4. Create a .env File

Create a `.env` file in the root directory of the project and add your OpenAI API key:

```plaintext
OPENAI_API_KEY=your_openai_api_key_here
```

Replace `your_openai_api_key_here` with your actual OpenAI API key.

### Running the Application

#### Run the Backend Server

Execute the following command to start the backend server:

```bash
python app.py
```

#### Run the Streamlit Frontend

To start the frontend interface using Streamlit, run:

```bash
streamlit run frontend.py
```

Navigate to the URL provided by Streamlit in your web browser to interact with the application.
