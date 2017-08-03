from create_feature_sets import create_feature_sets_and_labels
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import pickle
import numpy as np
import progressbar

n_nodes_hl1 = 1500
n_nodes_hl2 = 1500
n_nodes_hl3 = 1500

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

n_classes = 10
batch_size = 100
hm_epochs = 10

train_file = 'poker_hands_data/poker-hand-training-true.data'
test_file = 'poker_hands_data/poker-hand-testing.data'

train_x,train_y,test_x,test_y = create_feature_sets_and_labels(train_file,test_file)


def model(data):
    
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
              'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
          'biases':tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])

    return output

def train_neural_network(x):
    
    prediction = model(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

    hm_epochs = 10

    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            i=0
            bar = progressbar.ProgressBar(max_value=len(train_x))
            
            while i < len(train_x):
                start = i
                if(i+batch_size <= len(train_x)):
                    end = i+batch_size
                else:
                    end = len(train_x)

                batch_x = np.array(train_x[int(start):int(end)])
                batch_y = np.array(train_y[int(start):int(end)])

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,y: batch_y})
                epoch_loss += c
                
                i+=batch_size
                if(i < len(train_x)):
                    bar.update(i)
                
            print('epoch',epoch,'out of',hm_epochs,'loss:',epoch_loss)
            saver.save(sess, "/tmp/model.ckpt")
                         
        correct = tf.equal(tf.arg_max(prediction,1),tf.arg_max(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))

        print('accuracy:',accuracy.eval({x:test_x,y:test_y}))

train_neural_network(x)
