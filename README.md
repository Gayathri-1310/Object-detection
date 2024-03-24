# Object-detection
# Vehicle Detection Web Application

This web application allows users to upload images containing vehicles and performs vehicle detection using a pre-trained deep learning model. The detected vehicles are highlighted, and the count of different types of vehicles (cars, trucks, buses, motorcycles, bicycles) is provided.

## Installation

1. **Clone the Repository:**
   git clone <repository-url>

  1. Navigate to the Project Directory:
*cd vehicle-detection-web-app**

2. Install Dependencies:
*pip install -r requirements.txt**

3. Install Node.js and npm:
Ensure Node.js and npm are installed on your system. You can download and install them from here.

4. Install JavaScript Dependencies:
*cd image-processing
npm install**

**2.Usage**
**Running the Application**
**Start the Flask Application:**
**cd flask_py
flask run*

This will launch the Flask application locally.

Start the Image Processing Server:
Open a new terminal window.
**cd image-processing
npm start*

This will launch the image processing server.

**Access the Application:**
Open your web browser and navigate to http://127.0.0.1:5000/ to access the web application.

**Testing**
To test the web application, follow these steps:

1. Launch the Flask application and the image processing server as described above.
2. Open your web browser and go to http://127.0.0.1:5000/.
3. Upload an image containing vehicles using the provided interface.
4. Wait for the application to process the image and display the results.
5. Verify that the vehicles are correctly detected and highlighted in the image, and check the count of different types of vehicles displayed.
