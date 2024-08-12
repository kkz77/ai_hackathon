# International Ai Hackathon 2024
# AI Assistant Safety App
## Description
AI-Powered Safety Assistant is an advanced application designed to enhance occupational safety and reduce workplace accidents. Developed during a hackathon in Thailand, this project leverages state-of-the-art AI technologies to create a robust safety assistant that can proactively identify and mitigate potential hazards in various work environments.

Key Features:
- AI-Driven Object Detection: Utilized OpenCV and YOLO to implement real-time object detection, ensuring prompt identification of potential safety risks in the workplace.
- Generative AI Integration: Integrated cutting-edge AI models like Claude 3 Sonnet and Amazon Titan-Embed-Text to deliver generative capabilities, enhancing the system's ability to analyze and respond to various safety scenarios.
- Retrieval Augmented Generation (RAG): Implemented RAG to streamline document retrieval and language generation processes, enabling accurate, context-aware responses that improve decision-making and safety assessments.
This AI-powered safety assistant serves as a valuable tool for enhancing workplace safety by combining real-time detection, advanced AI-driven analysis, and responsive communication, ultimately reducing the risk of accidents and improving overall safety protocols.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/kkz77/ai_hackathon.git
    ```
2. Navigate to the project directory:
    ```bash
    cd project
    ```
    
3. Create a virtual environment
     ```bash
     python3 -m venv venv

4. Activate the virtual environment:
  - On Windows:
    ```bash
    venv\Scripts\activate
    ```
    
  - On macOS and Linux:
    ```bash
    source venv/bin/activate
    ```

5. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
6. ## Environment Setup
    
    To run this project, you need to configure your environment with the necessary AWS credentials and API keys. Follow these steps:
    
    1. Create a `.env` file in the root directory of your project.
    2. Add your environment variables to the `.env` file in the following format:
    
        ```plaintext
        GOOGLE_API_KEY='Your API KEY'
        ```
    
    3. If you're using AWS services, add the following environment variables:
    
        ```plaintext
        export AWS_DEFAULT_REGION=""
        export AWS_ACCESS_KEY_ID=""
        export AWS_SECRET_ACCESS_KEY=""
        export AWS_SESSION_TOKEN=""
        ```
    
        Replace the empty strings with your actual AWS credentials.
    
    4. Save the `.env` file.
    
    Make sure that all keys and tokens are filled in with your correct credentials.


## Running the Project

### Option 1: Using the Command Line

To run the project from the command line, use the Python interpreter from your virtual environment to execute the `main.py` script. The command will look something like this:

```bash
/path/to/your/virtualenv/bin/python /path/to/your/project/ai_hackathon/main.py
```

### Option 2: Using Visual Studio Code (VS Code)

- Open the main.py file in VS Code.
- Ensure your virtual environment is activated. If not, select it from the Python interpreter list in the bottom-left corner of VS Code.
- Click on the "Run Python File" button in the top-right corner of the editor, or press F5 to start debugging/running the script.

  
## Notes

### Command Line

When running commands from the command line, replace `/path/to/your/virtualenv/bin/python` with the path to the Python interpreter in your virtual environment.

### Visual Studio Code

Ensure that your virtual environment is activated in Visual Studio Code to use the correct Python interpreter. You can select the interpreter by opening the Command Palette (Ctrl+Shift+P or Cmd+Shift+P) and typing `Python: Select Interpreter`, then choosing the appropriate environment.

