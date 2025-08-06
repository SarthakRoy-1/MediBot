from setuptools import setup, find_packages

setup(
    name='MediBot',
    version='1.0.0',
    description='AI-powered medical chatbot using Flask and Pinecone',
    author='Sarthak Roy',
    author_email='your-email@example.com',  # <-- replace if needed
    packages=find_packages(),
    include_package_data=True,
    install_requires=[],  # Requirements handled by requirements.txt
)
