from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    name="l2d2",
    version="0.1",
    rust_extensions=[RustExtension("xtrie", binding=Binding.PyO3)],
    packages=["l2d2"],
    zip_safe=False,
    install_requires=[
        'jax>=0.1.68',
        'jaxlib>=0.1.47',
        'pytest==5.4.2',
    ],
)
