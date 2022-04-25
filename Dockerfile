FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

MAINTAINER Atabak Dehban <adehban@isr.tecnico.ulisboa.pt>


RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    curl \
    git \
    libcurl3-dev \
    libfreetype6-dev \
    libhdf5-serial-dev \
    libpng12-dev \
    libzmq3-dev \
    pkg-config \
    python-dev \
    rsync \
    tmux \
    software-properties-common \
    unzip \
    zip \
    zlib1g-dev \
    wget \
    python-tk \
    g++-4.8 \
    gcc-4.8 \
    python-numpy \
    vim \
    && \
    rm -rf /var/lib/apt/lists/*

RUN rm /usr/bin/g++ && \ 
    rm /usr/bin/gcc && \
    ln -s /usr/bin/g++-4.8 /usr/bin/g++ && \
    ln -s /usr/bin/gcc-4.8  /usr/bin/gcc

# Eigen
RUN git clone --branch=3.3.0 https://github.com/eigenteam/eigen-git-mirror.git && \
    cd eigen-git-mirror && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make && \
    make install

WORKDIR /

RUN curl -fSsL -O https://bootstrap.pypa.io/pip/2.7/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN pip --no-cache-dir install \
    Pillow \
    matplotlib \
    numpy \
    cython \
    easydict \
    scipy \
    enum34 \
    transforms3d \
    pyyaml \
    pillow

# opencv	
RUN git clone --branch=3.4.1 https://github.com/opencv/opencv.git && \
    cd opencv && \
    mkdir build && \
    cd build && \
    cmake -DBUILD_TIFF=ON -DBUILD_opencv_java=OFF -DWITH_CUDA=ON -DENABLE_AVX=ON -DWITH_OPENGL=OFF \
    -DWITH_OPENCL=ON -DWITH_TBB=ON -DWITH_EIGEN=ON -DWITH_V4L=ON -DWITH_VTK=OFF -DWITH_FFMPEG=OFF \
    -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DCMAKE_BUILD_TYPE=RELEASE -DINSTALL_PYTHON_EXAMPLES=ON \
    -DWITH_IPP=OFF -DWITH_ITT=OFF -DCUDA_ARCH_BIN=6.1 .. && \
    make -j && \
    make install && \
    ldconfig

WORKDIR /

# Set up Bazel.

# Running bazel inside a `docker build` command causes trouble, cf:
#   https://github.com/bazelbuild/bazel/issues/134
# The easiest solution is to set up a bazelrc file forcing --batch.
RUN echo "startup --batch" >>/etc/bazel.bazelrc
# Similarly, we need to workaround sandboxing issues:
#   https://github.com/bazelbuild/bazel/issues/418
RUN echo "build --spawn_strategy=standalone --genrule_strategy=standalone" \
    >>/etc/bazel.bazelrc
# Install the most recent bazel release.
ENV BAZEL_VERSION 0.10.0
WORKDIR /
RUN mkdir /bazel && \
    cd /bazel && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -o /bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE && \
    chmod +x bazel-*.sh && \
    ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    cd / && \
    rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh

# Download and build TensorFlow.
WORKDIR /tensorflow
RUN git clone --branch=r1.8 --depth=1 https://github.com/tensorflow/tensorflow.git .

# Configure the build for our CUDA configuration.
ENV CI_BUILD_PYTHON python
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ENV TF_NEED_CUDA 1
ENV TF_CUDA_COMPUTE_CAPABILITIES=6.1
ENV TF_CUDA_VERSION=9.0
ENV TF_CUDNN_VERSION=7

RUN pip --no-cache-dir install \
    h5py \
    ipykernel \
    keras_applications \
    keras_preprocessing \
    mock \
    scipy \
    sklearn \
    pandas \
    && \
    python -m ipykernel.kernelspec


RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH} \
    tensorflow/tools/ci_build/builds/configured GPU \
    bazel build -c opt --config=cuda \
    tensorflow/tools/pip_package:build_pip_package && \
    rm /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/pip && \
    pip --no-cache-dir install --upgrade /tmp/pip/tensorflow-*.whl && \
    rm -rf /tmp/pip && \
    rm -rf /.cache
# Clean up pip wheel and Bazel cache when done.
WORKDIR /

# Install nlopt
RUN git clone https://github.com/stevengj/nlopt.git && \ 
    cd nlopt && \
    git checkout 74e647b667f7c4500cdb4f37653e59c29deb9ee2 && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make install

WORKDIR /

RUN echo hello

# Install PoseCNN
RUN	git clone https://github.com/atabakd/PoseCNN.git 

# download VGG-16.npy
RUN curl -c /tmp/cookie.txt -s -L "https://drive.google.com/uc?export=download&id=1UdmOKrr9t4IetMubX-y-Pcn7AVaWJ2bL" |grep confirm |  sed -e\
    "s/^.*confirm=\(.*\)&amp;id=.*$/\1/" | xargs -I{} curl -b /tmp/cookie.txt  -L -o vgg16.npy "https://drive.google.com/uc?confirm={}&export=d\
    ownload&id=1UdmOKrr9t4IetMubX-y-Pcn7AVaWJ2bL"

RUN mv vgg16.npy /PoseCNN/data/imagenet_models/

# dwonload model
RUN curl -c /tmp/cookie.txt -s -L "https://drive.google.com/uc?export=download&id=1UNJ56Za6--bHGgD3lbteZtXLC2E-liWz" |grep confirm |  sed -e\
    "s/^.*confirm=\(.*\)&amp;id=.*$/\1/" | xargs -I{} curl -b /tmp/cookie.txt  -L -o demo_models.zip "https://drive.google.com/uc?confirm={}&ex\
    port=download&id=1UNJ56Za6--bHGgD3lbteZtXLC2E-liWz"
RUN mv demo_models.zip /PoseCNN/data/demo_models && \
    unzip /PoseCNN/data/demo_models/demo_models.zip -d /PoseCNN/data/demo_models/ && \
    rm /PoseCNN/data/demo_models/demo_models.zip

RUN mkdir /PoseCNN/data/LOV/data

WORKDIR /PoseCNN/lib


RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH} &&  \
    sh make.sh

# from https://github.com/Kaju-Bubanja/PoseCNN

RUN rm /usr/bin/g++ && \
    ln -s /usr/bin/g++-5* /usr/bin/g++ && \
    sed -i '64s/^/\/\//' /usr/local/cuda/include/crt/common_functions.h && \
    python setup.py build_ext --inplace && \
    sed -i '64s/^\/\///' /usr/local/cuda/include/crt/common_functions.h && \
    rm /usr/bin/g++ && \
    ln -s /usr/bin/g++-4.8 /usr/bin/g++


ENV PYTHONPATH /PoseCNN/lib:$PYTHONPATH
WORKDIR /PoseCNN
#RUN LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH} &&  \
CMD	./tools/demo.py --gpu 0 --network vgg16_convs --model data/demo_models/vgg16_fcn_color_single_frame_2d_pose_add_lov_iter_160000.ckpt --imdb lov_keyframe --cfg experiments/cfgs/lov_color_2d.yml --rig data/#LOV/camera.json --cad data/LOV/models.txt --pose data/LOV/poses.txt --background data/cache/backgrounds.pkl







































