import random
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
def read_write_everything():
    alphabet_loc = 'project_part_I/alphabet.csv'
    real_text_loc = []
    real_text = []
    cipher_loc = []
    cipher_text_loc = []
    real_text_loc.append('project_part_II/plaintext_feynman.txt')
    real_text_loc.append('project_part_II/plaintext_paradiselost.txt')
    real_text_loc.append('project_part_II/plaintext_warandpeace.txt')
    cipher_text_loc.append('project_part_II/ciphertext_feynman.txt')
    cipher_text_loc.append('project_part_II/ciphertext_paradiselost.txt')
    cipher_text_loc.append('project_part_II/ciphertext_warandpeace.txt')

    alphabet_to_num = {}

    real_text.append(read_file(real_text_loc[0], False, False)[0])
    real_text.append(read_file(real_text_loc[1], False, False)[0])
    real_text.append(read_file(real_text_loc[2], False, False)[0])

    alphabet = read_file(alphabet_loc, False, False)
    
    cipher_loc.append(open(cipher_text_loc[0], "w"))
    cipher_loc.append(open(cipher_text_loc[1], "w"))
    cipher_loc.append(open(cipher_text_loc[2], "w"))

    for i in range(3):
        cipher_str = ""
        func = random_func(alphabet)
        for j in range(len(real_text[i]) - 1):
            cipher_str += func[real_text[i][j]]
        cipher_loc[i].write(cipher_str + "\n")
        cipher_loc[i].close()

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

read_write_everything()