from setuptools import setup, find_packages

setup(
    name="vae-molecular-cg",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'mdtraj',
        'torch',
        'matplotlib',
        'scikit-learn',
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Molecular dynamics coarse-graining using variational autoencoders and other methods",
    keywords="molecular dynamics, coarse-graining, vae, pca, machine learning",
    url="https://github.com/yourusername/vae-molecular-cg",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Physics',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.7',
)
