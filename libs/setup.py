from setuptools import setup, find_packages


setup(name='persona2vec',
      version='0.1',
      url='(anon),
      license='MIT',
      entry_points={
          'console_scripts': ['persona2vec=persona2vec.command_line:main'],
      },

      author='Anonymous Authors',
      author_email='anon',
      description='Persona2Vec library',
      packages=find_packages(),
      zip_safe=False)
