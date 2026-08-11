from setuptools import setup, find_packages

setup(
    name="regalabs-tts",
    version="1.0.0",
    description="RegaLabs-TTS: CosyVoice 3 Central Kurdish (Sorani) Text-to-Speech Adaptation Tools",
    author="RegaLabs",
    license="Apache-2.0",
    url="https://github.com/RegaLabs/RegaLabs-TTS",
    packages=find_packages(),
    install_requires=[
        "torch",
        "soundfile",
        "numpy",
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Audio :: Speech Synthesis",
    ],
    python_requires=">=3.8",
)
