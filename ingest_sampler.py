from ingest import State


_SAMPLED_SENTINEL = object()


def sample_minibatches_no_replacement(state: State, minibatch_size: int, length: int):
    sampled = []
    targets = []
    while True:
        for i in range(minibatch_size):
            sample, target = sample_instruction_no_replacement(state)
            # Fill the rest of the bytes with random noise.
            while len(sample) < length:
                sample.append(random.randrange(0x100))
            sampled.append(sample)
        yield sampled
        sampled, targets = [], []

    while len(sampled) < length:
        sampled.append([0] * length)
    yield sampled
