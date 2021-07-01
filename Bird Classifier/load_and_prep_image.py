import tensorflow as tf

def load_and_prep_image(img, img_shape=224):

  img = tf.image.decode_image(img, channels=3)

  img = tf.image.resize(img, size = [img_shape, img_shape])
  
  return img