
def silence_tf():
    import os, logging

    # original: https://pypi.org/project/silence-tensorflow/
    # the original code seemed to be producing a report automatically that seemed suspicious and was blocking the code



    """Silence every unnecessary warning from tensorflow."""
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    os.environ["KMP_AFFINITY"] = "noverbose"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # We wrap this inside a try-except block
    # because we do not want to be the one package
    # that crashes when TensorFlow is not installed
    # when we are the only package that requires it
    # in a given Jupyter Notebook, such as when the
    # package import is simply copy-pasted.
    try:
        import tensorflow as tf

        tf.get_logger().setLevel('ERROR')
        tf.autograph.set_verbosity(3)
    except ModuleNotFoundError:
        pass

    import warnings

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['AUTOGRAPH_VERBOSITY'] = '1'
    warnings.filterwarnings('ignore')