import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

m,n = 28,28

n_input = m*n # 28x28 images
n_output = 10 # Classes

# Training a Linear classifier: Xw + b = y

X = tf.placeholder(tf.float32,shape=[None, n_input],name="X") # input type and shape
y = tf.placeholder(tf.float32,shape=[None,n_output],name="y") # output type and shape
w = tf.Variable(tf.random_normal(shape=[784, 10], stddev=0.01), name="weights")
b = tf.Variable(tf.zeros([1,10]), name='bias')

# compute scores (logits)
logits = tf.matmul(X, w) + b

# Entropy and loss functions definition
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
loss = tf.reduce_mean(entropy)

# Use GD to minimise loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.3).minimize(loss)

num_epochs = 10
batch_size = 100

init = tf.global_variables_initializer()

loss_values = []
total_correct_predicts = 0

with tf.Session() as session:
    session.run(init)

    # Training
    n_batches = int(mnist.train.num_examples / batch_size)
    for epoch_i in range(num_epochs):
        for batch in range(n_batches):
            batch_x,batch_y = mnist.train.next_batch(batch_size=batch_size)
            _,loss_value = session.run(
                [optimizer,loss],feed_dict={
                X : batch_x,
                y:batch_y},
            )
        loss_values.append(loss_value)
        print("Epoch: {0} Loss: {1}".format(epoch_i,loss_value))

    print("Loss values in training: {}".format(loss_values))

    n_validation_batches = int(mnist.validation.num_examples/batch_size)

    for batch in range(n_validation_batches):
        batch_x,batch_y = mnist.validation.next_batch(batch_size=batch_size)
        _, loss_batch, logits_batch = session.run([optimizer, loss, logits], feed_dict={X: batch_x, y: batch_y})
        preds = tf.nn.softmax(logits_batch)
        correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(batch_y, 1))
        accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
        total_correct_predicts += session.run(accuracy)

    validation_accuracy = total_correct_predicts / mnist.validation.num_examples
    print("Validation Accuracy: {}".format(validation_accuracy))

    # Testing
    total_correct_predicts = 0
    n_batches = int(mnist.test.num_examples / batch_size)

    for mini_batch in range(n_batches):
        batch_x, batch_y = mnist.test.next_batch(batch_size=batch_size)

        logits_batch = session.run(logits,feed_dict={
            X : batch_x,
            y: batch_y
        })

        predictions = tf.nn.softmax(logits=logits_batch)
        correct_predicts = tf.equal(tf.argmax(predictions,1),tf.argmax(batch_y,1))
        accuracy = tf.reduce_sum(tf.cast(correct_predicts,tf.float32)) # sum efficiently
        total_correct_predicts = total_correct_predicts + session.run(accuracy)

    print("Model evaluation completed. Accuracy: {}".format(total_correct_predicts/mnist.test.num_examples))

    # Uncomment the following code to test a single pattern
    '''logits_batch = session.run(logits, feed_dict={
        X: [mnist.test.images[10]],
        y : [mnist.test.labels[10]]
    })

    plt.imshow((np.reshape(mnist.test.images[1],(28,28)) * 255))
    plt.show()
    print(logits_batch)
    predictions = tf.nn.softmax(logits=logits_batch)
    tf.Print(input_=predictions,data=[predictions],message="Testing Prediction: ")'''