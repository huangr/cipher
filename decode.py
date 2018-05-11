import random
import matplotlib.pyplot as plt
import math
from collections import deque
from multiprocessing.dummy import Pool as ThreadPool

epochs = 10
bag_num = 7
iterations = 4000
borderline = 0.0001
alphabet = []
alphabet_to_num = {}
probs = []
text = ""
real_text = ""
pool = ThreadPool(7)

# transition[i][j] means x_{k-1} = j to x_{k} = i
transition = []

# Read in a file into the desired array format
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

# Read in all the necessary files
def read_everything():
    alphabet_loc = 'project_part_I/alphabet.csv'
    letter_trans_loc = 'project_part_I/letter_transition_matrix.csv'
    letter_prob_loc = 'project_part_I/letter_probabilities.csv'
    # text_loc = 'project_part_II/ciphertext_warandpeace.txt'
    # real_text_loc = 'project_part_II/plaintext_warandpeace.txt'

    alphabet_to_num = {}

    alphabet = read_file(alphabet_loc, False, False)
    probs = read_file(letter_prob_loc, True, False)
    transition = read_file(letter_trans_loc, True, True)

    for i in range(0, len(alphabet)):
        alphabet_to_num[alphabet[i]] = i
    
    return alphabet, probs, transition, alphabet_to_num

# Preprocess transition matrix so there are no zero probabilities for updating
def preprocess(m, borderline):
    m_new = list(m)
    for i in range(0, len(m_new)):
        for j in range(0, len(m_new[i])):
            if m_new[i][j] == 0:
                m_new[i][j] = borderline
    return m_new

# Find a random substitution cipher function
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

# Find a good initial seed based on text frequencies
def initial_func(text, alphabet, probs, alphabet_to_num, total_ct):
    alphabet_sorted = []
    alphabet_counts = {}
    for i in range(len(alphabet)):
        alphabet_sorted.append((probs[i], alphabet[i]))
        alphabet_counts[alphabet[i]] = 0
    alphabet_sorted = sorted(alphabet_sorted, key=lambda x: x[0])
    for i in range(total_ct):
        alphabet_counts[text[i]] = alphabet_counts[text[i]] + 1

    matching_alpha = []
    for key in alphabet_counts:
        matching_alpha.append((float(alphabet_counts[key]) / float(total_ct), key))
    matching_alpha = sorted(matching_alpha, key = lambda x: x[0])
    func = {}
    for i in range(len(alphabet_sorted)):
        func[alphabet_sorted[i][1]] = matching_alpha[i][1]
    return func

# Find a good random substring to train on
def random_substring(text, max_cst, max_prop):
    start = 0
    end = int(len(text) * (1-max_prop))
    max_len = max(max_cst, int(len(text) * max_prop))
    start = max(min(random.randint(start, end), len(text) - max_cst), 0)
    end = min(start + max_len, len(text))
    return text[start:end]

# Find the inverse function of a given cipher function
def inverse_func(func):
    inverse_func = {}
    for key in func:
        inverse_func[func[key]] = key
    return inverse_func

# Find the log likelihood given a piece of text and functions
def log_likelihood(func, inverse_func, text, probs, transition, alphabet_to_num):
    # off by factor of 1/m!
    likelihood = math.log(probs[alphabet_to_num[inverse_func[text[0]]]])

    prev_letter = inverse_func[text[0]]
    for i in range(1, len(text)):
        cur_letter = inverse_func[text[i]]
        likelihood += math.log(transition[alphabet_to_num[cur_letter]][alphabet_to_num[prev_letter]])
        prev_letter = cur_letter
    return likelihood 

# Run a MCMC simulation
def mcmc(inputs):
    (alphabet, probs, transition, text, alphabet_to_num, iterations) = inputs
    text_substr = random_substring(text, 5000, 0.1)
    func = initial_func(text_substr, alphabet, probs, alphabet_to_num, len(text) / 10)
    inv_func = inverse_func(func)
    best_func = func
    best_inv_func = inv_func
    cur_likelihood = log_likelihood(func, inv_func, text_substr, probs, transition, alphabet_to_num)
    best_likelihood = cur_likelihood 
    last_update = 0

    x = []
    y = []

    for i in range(0, iterations):
        # if i % 500 == 0:
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
        new_likelihood = log_likelihood(new_func, new_inv_func, text_substr, probs, transition, alphabet_to_num)
        
        if math.log(random.random()) < new_likelihood - cur_likelihood:
            func = dict(new_func)
            inv_func = dict(new_inv_func)
            cur_likelihood = new_likelihood

        if new_likelihood > best_likelihood:
            best_func = new_func
            best_inv_func = new_inv_func
            best_likelihood = new_likelihood
            last_update = i

        if i - last_update > 450:
            # print "\t Iterations made:", i
            break

        # x.append(i+1)
        # y.append(score(func, real_text, text))

    return best_func

def bag(alphabet, probs, transition, text, alphabet_to_num, iterations, bag_num):
    global pool
    input_arr = [(alphabet, probs, transition, text, alphabet_to_num, iterations) for i in range(bag_num)]
    results = pool.map(mcmc, input_arr)
    func = results[0]
    text_substr = random_substring(text, 12000, 0.2)
    likelihood = log_likelihood(func, inverse_func(func), text_substr, probs, transition, alphabet_to_num)
    for i in range(bag_num - 1):
        new_func = results[i + 1]
        new_likelihood = log_likelihood(new_func, inverse_func(new_func), text_substr, probs, transition, alphabet_to_num)
        if new_likelihood > likelihood:
            func = new_func
            likelihood = new_likelihood
    return func

def score(func, real_text, ciphertext):
    inv_func = inverse_func(func)
    correct = 0
    for i in range(len(real_text)):
        if inv_func[ciphertext[i]] == real_text[i]:
            correct += 1
    return float(correct) / float(len(real_text))

def decode(ciphertext, output_file_name):
    global epochs, bag_num, iterations, borderline, pool
    alphabet, probs, transition, alphabet_to_num = read_everything()
    transition = preprocess(transition, borderline)
    text = ciphertext
    real_text = ""
    func = bag(alphabet, probs, transition, text, alphabet_to_num, iterations, bag_num)
    inv_func = inverse_func(func)
    for char in text:
        real_text += inv_func[char]
    write_file = open(output_file_name, "w")
    write_file.write(real_text)
    write_file.close()

    return

# for i in range(epochs):
#     func = bag(alphabet, probs, transition, text, alphabet_to_num, iterations, bag_num)
#     lg_like = log_likelihood(func, inverse_func(func), text, probs, transition, alphabet_to_num)
#     sc = score(func, real_text, text)
#     print 'Epoch', i+1, 'of', epochs
#     print '\tLog-likelihood:', lg_like
#     print '\tPercentage correct:', sc
#     if sc == 1:
#         total_ct += 1
#     # plt.plot(x, y)
#     # plt.show()

# print 'Total accuracy:', float(total_ct) / float(epochs)
