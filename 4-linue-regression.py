import numpy as np
# 思路1：
# 计算所有点与现在在w，b值所拟的直线与真实值的差值和，返回其均值
def computer_loss(b, w, points):
    total_error  = 0
    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        total_error += (y - (w * x + b)) ** 2
    return total_error / float(len(points))

# 计算所有点与此直线的梯度，并求均值，移动更新w和b的值
def step_gradient(points, b, w, lr):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        w_gradient += 2 * x * (w * x + b -y) / N
        b_gradient += 2 * (w * x + b -y) / N
    new_w = w - lr * w_gradient
    new_b = b - lr * b_gradient
    return [new_w, new_b]

# 传入数据，并进行梯度下降优化
def gradient_decent_runner(points, starting_w, starting_b, lr, num_iterations):
    w = starting_w
    b = starting_b
    for i in range(num_iterations):
        w, b = step_gradient(points, w, b, lr)
    return [w, b]

def run():
    points = np.genfromtxt("./data/data.csv", delimiter=",")
    learning_rate = 0.00001
    initial_b = 0 # initial y-intercept guess
    initial_m = 0 # initial slope guess
    num_iterations = 100000
    print("Starting gradient descent at b = {0}, m = {1}, error = {2}"
          .format(initial_b, initial_m,
                  computer_loss(initial_b, initial_m, points))
          )
    print("Running...")
    [b, m] = gradient_decent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print("After {0} iterations b = {1}, m = {2}, error = {3}".
          format(num_iterations, b, m,
                 computer_loss(b, m, points))
          )

if __name__ == '__main__':
    run()
