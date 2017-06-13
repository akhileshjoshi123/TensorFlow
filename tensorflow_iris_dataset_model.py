import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names=['f1','f2','f3','f4','f5']
df = pd.read_csv(url, names=names)
print('dataset: {0}'.format(df))


# Shuffle Pandas data frame
import sklearn.utils
df = sklearn.utils.shuffle(df)
print('\n\ndf: {0}'.format(df))

# Then you can use df.reset_index() to reset the index column, if needs to be:
df = df.reset_index(drop=True)
print('\n\ndf: {0}'.format(df))

g=sns.pairplot(df, hue="f5", size= 2.5)


one_to_hundred = pd.Series(range(1,4))
df1 = pd.DataFrame([one_to_hundred]).transpose()
df1.columns= ['class']
df1['class']=df1['class'].astype('category')
one_hot = pd.DataFrame(pd.get_dummies(df1))


mat = one_hot.as_matrix()

#map data into arrays
s=np.asarray(mat[0])
ve=np.asarray(mat[1])
vi=np.asarray(mat[2])
df['f5'] = df['f5'].map({'Iris-setosa': s, 'Iris-versicolor': ve,'Iris-virginica':vi})



x_input=df.ix[:,['f1','f2','f3','f4']]
y_input=df['f5']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_input, y_input, test_size=0.33, random_state=42)


x_input= X_train
y_input= y_train

#placeholders and variables. input has 4 features and output has 3 classes
x=tf.placeholder(tf.float32,shape=[None,4])
y_=tf.placeholder(tf.float32,shape=[None, 3])
#weight and bias
W=tf.Variable(tf.random_normal([4,3]))

'''
W looks like
array([[ 0.,  0.,  0.],
       [ 0.,  0.,  0.],
       [ 0.,  0.,  0.],
       [ 0.,  0.,  0.]], dtype=float32)
'''

b=tf.Variable(tf.random_normal([3]))

# b looks like array([ 0.,  0.,  0.], dtype=float32)


# model 
#softmax function for multiclass classification
y = tf.nn.softmax(tf.matmul(x, W) + b)

#y_ original y
#loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))


#optimiser -
train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
#calculating accuracy of our model 
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



#session parameters
sess = tf.InteractiveSession()
#initialising variables
init = tf.initialize_all_variables()
sess.run(init)
#number of interations
epoch=2000


for step in range(epoch):
   _, c=sess.run([train_step,cross_entropy], feed_dict={x: x_input, y_:[t for t in y_input.as_matrix()]})
   if step%500==0 :
       print (c)

print ("Accuracy is " , sess.run(accuracy,feed_dict={x: X_test, y_:[t for t in y_test.as_matrix()]}))




