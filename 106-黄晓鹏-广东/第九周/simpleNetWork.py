import numpy
import scipy.special

class SimpleNetWork:
    def __init__(self,input_nodes,hidden_nodes,output_nodes,learning_rate):
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        self.lr = learning_rate

        self.wih = numpy.random.rand(self.hnodes, self.inodes) - 0.5
        self.who = numpy.random.rand(self.onodes, self.hnodes) - 0.5

        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self,inputs_list, targets_list):
        # input  to  matric
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)

        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)

        final_outputs = self.activation_function(final_inputs)

        final_outputs = self.activation_function(final_inputs)

        # err
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors * final_outputs * (1 - final_outputs))

        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1 - final_outputs)),
                                        numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                                        numpy.transpose(inputs))

        pass


    def qry(self,inputs):
        #
        hidden_input = numpy.dot(self.wih,inputs)
        hidden_output = self.activation_function(hidden_input)
        final_input = numpy.dot(self.who, hidden_output)
        final_output = self.activation_function(final_input)
        return final_output



# init
input_nodes = 28*28
hidden_nodes = 180
output_nodes = 10
learning_rate = 0.15
net = SimpleNetWork(input_nodes,hidden_nodes,output_nodes,learning_rate)

#load data
data_src = open("dataset/mnist_train.csv",'r')
data_list = data_src.readlines()
data_src.close()

#set epoch train
epochs = 20
for e in range(epochs):
    for data in data_list:
        values = data.split(',')
        #normalize
        input_data = (numpy.asfarray(values[1:]))/255 * 0.99 + 0.01
        #初始化全 0.01 数组
        targets_data = numpy.zeros(output_nodes) + 0.01
        #data 第一位是标签也就是数值，数值是多少就对应的把 输出数组的对应索引设为最大0.99
        targets_data[int(values[0])] = 0.99
        net.train(input_data,targets_data)

test_data_src = open("dataset/mnist_test.csv",'r')
test_data_list = test_data_src.readlines()
test_data_src.close()


print("=======test=====")
score = []
for data_list in test_data_list:
    values = data_list.split(',')
    correct_number = int(values[0])
    print("正确号码：",correct_number)
    input_data = (numpy.asfarray(values[1:])) / 255.0 * 0.99 + 0.01
    out_put = net.qry(input_data)
    label = numpy.argmax(out_put)
    if label == correct_number:
        score.append(1)
    else:
        score.append(0)

print("score:",score)
print("正确率(1t/0f)",numpy.asarray(score).sum() / numpy.asarray(score).size)

