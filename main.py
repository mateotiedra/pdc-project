import random
import numpy as np

MESS_LEN = 40  # Length of the message to be transmitted
ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ."
assert len(ALPHABET) == 64, "Alphabet must contain exactly 64 characters."


### Codewords construction ###
def create_Mr(r):
    if r == 0:
        return np.ones((1, 1))
    Mr_minus_1 = create_Mr(r - 1)
    top = np.hstack((Mr_minus_1, Mr_minus_1))
    bottom = np.hstack((Mr_minus_1, -Mr_minus_1))
    return np.vstack((top, bottom))


G = 10
SIGMA2 = 10
r = 13
c_len = 2 ** (r + 1)  # c_len = 64
Mr = create_Mr(r)  # (32, 32)
Br = np.vstack((Mr, -Mr))  # (64, 32)
codewords = np.hstack((Br, Br))  # (64, 64)

# Scale the codewords to not have energy greater than 2000
total_length = MESS_LEN * (2 * c_len)
energy_per_codeword = c_len
alpha = 2000 / (energy_per_codeword * MESS_LEN + 1e-9)  # Avoid division by zero
scale = np.sqrt(alpha)
codewords = scale * codewords

THRES = 190.0  # THREShold for decoding


### Transmitter function ###
def transmitter(i: str):
    assert len(i) == MESS_LEN
    idx = np.array([ALPHABET.index(c) for c in i])  # (MESS_LEN,)
    signal = codewords[idx].flatten()  # (MESS_LEN, 64) => (2560,)
    return signal


### Channel simulation function ###
def channel(x: np.ndarray):
    assert x.size <= 1000000
    assert np.sum(x**2) <= 2000

    s = random.choice([1, 2])
    n = x.size
    Y = np.random.normal(0, np.sqrt(SIGMA2), n)
    if s == 1:
        x_even = np.array(x[::2]) * np.sqrt(G)
        x_odd = x[1::2]
    else:
        x_even = np.array(x[::2])
        x_odd = x[1::2] * np.sqrt(G)
    Y[::2] += x_even
    Y[1::2] += x_odd
    return Y


### Receiver function ###
def receiver(y: np.ndarray):
    decoded_idx_1 = []
    decoded_idx_2 = []

    for i in range(MESS_LEN):
        # Extract the received chunk for the i-th codeword
        start = i * c_len
        end = start + c_len
        codeword = y[start:end]

        y_p = codeword[: c_len // 2]
        y_pp = codeword[c_len // 2 :]

        scores = np.zeros(len(ALPHABET))
        for j in range(len(ALPHABET)):
            c = codewords[j]
            c_p = c[: c_len // 2]
            c_pp = c[c_len // 2 :]

            p_p = np.dot(y_p, c_p)
            p_pp = np.dot(y_pp, c_pp)

            score_1 = np.sqrt(G) * p_p + p_pp
            score_2 = p_p + np.sqrt(G) * p_pp

            scores[j] = max(score_1, score_2)

        # Find the index of the maximum score
        decoded_idx_1.append(np.argmax(scores))

        # Filter all elements that are greater than THRES
        above_thres_idx = [score for _, score in enumerate(scores) if score > THRES]
        if len(above_thres_idx) > 1:
            decoded_idx_2.append(0)
        else:
            decoded_idx_2.append(np.argmax(scores))

    # Convert indices back to characters
    decoded_message_1 = "".join(ALPHABET[idx] for idx in decoded_idx_1)
    decoded_message_2 = "".join(ALPHABET[idx] for idx in decoded_idx_2)

    return [decoded_message_1, decoded_message_2]


def calculate_similarity(original, decoded):
    """
    Calculate the percentage of similarity between two strings.
    Returns a value between 0 (completely different) and 100 (identical).
    """
    if not original and not decoded:
        return 100.0

    if not original or not decoded:
        return 0.0

    min_len = min(len(original), len(decoded))

    matches = sum(1 for i in range(min_len) if original[i] == decoded[i])

    max_len = max(len(original), len(decoded))
    similarity = matches / max_len

    return similarity


if __name__ == "__main__":

    print(f"Testing with r = {r}")

    # Generate a random text message of 40 characters
    plain_message = "".join(random.choices(ALPHABET, k=MESS_LEN))
    # print("Text message: ", plain_message)

    transmitted_message = transmitter(plain_message)

    received_message = channel(transmitted_message)

    decoded_messages = receiver(received_message)

    # print("Decoded message 1:", decoded_messages[0])
    print(
        "Similarity :",
        str(calculate_similarity(plain_message, decoded_messages[0]) * 40) + "/40",
    )
