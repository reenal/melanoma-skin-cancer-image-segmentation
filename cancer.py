import os
import numpy as np
from glob import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split
import cv2

tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)