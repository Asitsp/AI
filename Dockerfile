
# Use an official Python runtime as a parent image
FROM python:3.8

# Set the working directory
WORKDIR /app

# Copy the Python script and dataset into the container
COPY aisc2013.py /app/

# Install required dependencies
RUN pip install numpy pandas matplotlib seaborn scikit-learn

# Command to run the script
CMD ["python", "aisc2013.py"]
