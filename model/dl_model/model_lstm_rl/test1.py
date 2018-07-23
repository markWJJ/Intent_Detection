import tensorflow as tf



s=tf.placeholder(dtype=tf.int32)



with tf.Session() as sess:

    print(sess.run(s,feed_dict={s:1}))