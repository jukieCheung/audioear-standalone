from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="audioear-standalone",
    version="1.0.0",
    author="AudioEar Project",
    description="Standalone AudioEar 3D ear reconstruction (no PyTorch3D)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/audioear-standalone",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
        "torchvision>=0.11.0",
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "Pillow>=8.0.0",
    ],
    entry_points={
        "console_scripts": [
            "audioear-inference=scripts.run_inference:main",
        ],
    },
)
