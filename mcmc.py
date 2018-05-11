import random
import matplotlib.pyplot as plt
import math
from collections import deque

epochs = 1
iterations = 3000
borderline = 0.001
alphabet = []
alphabet_to_num = {}
probs = []
text = ""
real_text = ""

# transition[i][j] means x_{k-1} = j to x_{k} = i
transition = []


def read_file(loc, is_float, is_two):
    tokens = []
    with open(loc) as f:
        for line in f:
            if len(line.split(',')) != 0:
                tokens.append(line.split(','))
    if len(tokens) == 1:
        tokens = tokens[0]
    if is_float:
        if is_two:
            for i in range(len(tokens)):
                for j in range(len(tokens[i])):
                    tokens[i][j] = float(tokens[i][j])
        else:
            for i in range(len(tokens)):
                tokens[i] = float(tokens[i])
    return tokens

def read_everything():
    alphabet_loc = 'project_part_I/alphabet.csv'
    letter_trans_loc = 'project_part_I/letter_transition_matrix.csv'
    letter_prob_loc = 'project_part_I/letter_probabilities.csv'
    text_loc = 'project_part_I/ciphertext.txt'
    real_text_loc = 'project_part_I/plaintext.txt'

    alphabet_to_num = {}

    alphabet = read_file(alphabet_loc, False, False)
    probs = read_file(letter_prob_loc, True, False)
    transition = read_file(letter_trans_loc, True, True)
    text = read_file(text_loc, False, False)[0]
    real_text = read_file(real_text_loc, False, False)[0]

    for i in range(0, len(alphabet)):
        alphabet_to_num[alphabet[i]] = i
    
    return alphabet, probs, transition, text, alphabet_to_num, real_text

def preprocess(m, borderline):
    m_new = list(m)
    for i in range(0, len(m_new)):
        for j in range(0, len(m_new[i])):
            if m_new[i][j] == 0:
                m_new[i][j] = borderline
    return m_new

def random_func(alphabet):
    starts = list(alphabet)
    targets = list(alphabet)
    func = {}
    while len(starts) != 0:
        x = starts[int(random.random() * len(starts))]
        y = targets[int(random.random() * len(targets))]
        func[x] = y
        starts.remove(x)
        targets.remove(y)
    return func

def inverse_func(func):
    inverse_func = {}
    for key in func:
        inverse_func[func[key]] = key
    return inverse_func

def log_likelihood(func, inverse_func, text, probs, transition, alphabet_to_num):
    # off by factor of 1/m!
    likelihood = math.log(probs[alphabet_to_num[inverse_func[text[0]]]])

    for i in range(len(alphabet_to_num)):
        likelihood += math.log(i+1)

    prev_letter = inverse_func[text[0]]
    for i in range(1, len(text)):
        cur_letter = inverse_func[text[i]]
        likelihood += math.log(transition[alphabet_to_num[cur_letter]][alphabet_to_num[prev_letter]])
        prev_letter = cur_letter
    return likelihood 


def mcmc(alphabet, probs, transition, text, alphabet_to_num, iterations):
    global real_text
    func = random_func(alphabet)
    inv_func = inverse_func(func)
    cur_likelihood = log_likelihood(func, inv_func, text, probs, transition, alphabet_to_num)

    x = []
    y = []

    for i in range(0, iterations):
        # if i % 1000 == 0:
        #     print i, cur_likelihood
        new_func = dict(func)
        x1 = alphabet[int(random.random() * len(alphabet))]
        x2 = alphabet[int(random.random() * len(alphabet))]
        while x1 == x2:
            x1 = alphabet[int(random.random() * len(alphabet))]
            x2 = alphabet[int(random.random() * len(alphabet))]
        y1 = func[x1]
        y2 = func[x2]
        new_func[x1] = y2
        new_func[x2] = y1
        new_inv_func = inverse_func(new_func)
        new_likelihood = log_likelihood(new_func, new_inv_func, text, probs, transition, alphabet_to_num)
        
        if math.log(random.random()) < new_likelihood - cur_likelihood:
            func = dict(new_func)
            inv_func = dict(new_inv_func)
            cur_likelihood = new_likelihood

        # x.append(i+1)
        # y.append(score(func, real_text, text))

    return func, x, y

def score(func, real_text, ciphertext):
    inv_func = inverse_func(func)
    correct = 0
    for i in range(len(real_text)):
        if inv_func[ciphertext[i]] == real_text[i]:
            correct += 1
    return float(correct) / float(len(real_text))

alphabet, probs, transition, text, alphabet_to_num, real_text = read_everything()
transition = preprocess(transition, borderline)

total_ct = 0

for i in range(epochs):
    func, x, y = mcmc(alphabet, probs, transition, text, alphabet_to_num, iterations)
    lg_like = log_likelihood(func, inverse_func(func), text, probs, transition, alphabet_to_num)
    sc = score(func, real_text, text)
    print 'Epoch', i+1, 'of', epochs
    print '\tLog-likelihood:', lg_like
    print '\tPercentage correct:', sc
    if sc == 1:
        total_ct += 1
    # plt.plot(x, y)
    # plt.show()

print 'Total accuracy:', float(total_ct) / float(epochs)
