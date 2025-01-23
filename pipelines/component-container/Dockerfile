# Use a specific base image for Python
FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/base-cu121.py310

# Set the working directory
WORKDIR /app

# Install only the necessary system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ \
        cmake wget unzip ffmpeg libsm6 libxext6&& \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Download and unpack sources
# RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/refs/tags/4.8.1.zip && \
#     unzip opencv.zip 

# RUN mkdir -p /app/opencv-4.8.1/build && \
#     cd /app/opencv-4.8.1/build && \
#     cmake \
#     -DOPENCV_GENERATE_PKGCONFIG=ON \
#     -DWITH_CUDA=OFF \
#     -DBUILD_TESTS=OFF \
#     -DBUILD_PERF_TESTS=OFF \
#     -DBUILD_EXAMPLES=OFF \
#     -DBUILD_JPEG=ON \ 
#     -DBUILD_PNG=ON \
#     -DCMAKE_BUILD_TYPE=Release \
#     -DBUILD_LIST=core,imgcodecs,imgproc \
#     /app/opencv-4.8.1 && \
#     cmake --build . -- -j64 && \
#     make install 

# RUN pkg-config --modversion opencv4

RUN pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

RUN pip install ultralytics

RUN pip install opencv-python

RUN pip install onnx>=1.12.0

RUN pip install onnxslim onnxruntime-gpu

RUN pip install tensorrt>7.0.0,!=10.1.0

# Copy the application code
COPY . .

# Set the Python path
ENV PYTHONPATH=/app

# Run Python
CMD ["python", "test.py"]