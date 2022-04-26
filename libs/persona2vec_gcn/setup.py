from setuptools import setup, find_packages


setup(name='persona2vec_gcn',
      version='0.0',
      url='https://github.com/ashutosh1919/persona2vec',
      license='MIT',
      entry_points={
          'console_scripts': ['persona2vec=persona2vec.command_line:main'],
      },

      author='Ashutosh Hathidara, Sadamori Kojaku, Jisung Yoon, and Yong-Yeol Ahn',
      author_email='@gmail.com',
      description='GCN Persona2Vec library',
      packages=find_packages(),
      zip_safe=False)
