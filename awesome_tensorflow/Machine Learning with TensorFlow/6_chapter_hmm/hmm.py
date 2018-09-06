import numpy as np


class HMM():

    def __init__(self, start_probability, transition_probability, emission_probability):
        self.initial_prob = start_probability
        self.transition_probability = transition_probability
        self.emission_probability = emission_probability

    def get_prop(self, observations):
        ob = observations[0]
        prop = self.initial_prob * self.emission_probability[:, ob]
        for ob in observations[1:]:
            state_prop = self.next_state_prop(prop)
            prop = state_prop * self.emission_probability[:, ob]
        return np.sum(prop)

    def viterbi(self, observations):
        ob = observations[0]
        state = []
        state_prop = self.initial_prob
        s = np.argmax(state_prop * self.emission_probability[:, ob])
        state.append(s)
        for ob in observations[1:]:
            state_prop = self.next_state_prop(state_prop)
            prop = state_prop * self.emission_probability[:, ob]
            print(prop)
            s = np.argmax(prop)
            state.append(s)
        return state

    def next_state_prop(self, state_prop):
        return np.sum(np.expand_dims(state_prop, 1) * self.transition_probability, axis=0)


if __name__ == '__main__':
    # states = ('Rainy', 'Sunny')
    #
    # observations = ('walk', 'shop', 'clean')
    #
    # start_probability = {'Rainy': 0.6, 'Sunny': 0.4}
    #
    # transition_probability = {
    #     'Rainy': {'Rainy': 0.7, 'Sunny': 0.3},
    #     'Sunny': {'Rainy': 0.4, 'Sunny': 0.6},
    # }
    #
    # emission_probability = {
    #     'Rainy': {'walk': 0.1, 'shop': 0.4, 'clean': 0.5},
    #     'Sunny': {'walk': 0.6, 'shop': 0.3, 'clean': 0.1},
    # }
    initial_prob = np.array([0.6, 0.4])
    trans_prob = np.array([[0.7, 0.3], [0.4, 0.6]])
    emi_prob = np.array([[0.5, 0.4, 0.1], [0.1, 0.3, 0.6]])
    observations = [0, 1, 1, 2, 1]
    hmm = HMM(initial_prob, trans_prob, emi_prob)
    print(hmm.get_prop(observations))
    print(hmm.viterbi(observations))
