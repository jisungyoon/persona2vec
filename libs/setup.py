from setuptools import setup, find_packages


setup(name='persona2vec',
      version='0.1',
      url='https://github.com/jisungyoon/persona2vec',
      license='MIT',
      entry_points = {
        'console_scripts': ['persona2vec=persona2vec.command_line:main'],
    },

	  author='Jisung Yoon, Kaicheng Yang and Yong-Yeol Ahn',
      author_email='jisung.yoon92@gmail.com',
      description='Persona2Vec library',
      packages=find_packages(),
      zip_safe=False)
