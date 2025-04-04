def viterbi_decode(obs, states, start_prob, trans_prob, emit_prob):
    V, path = [{}], {}
    for state in states:
        V[0][state] = start_prob[state] * emit_prob[state].get(obs[0], 0)
        path[state] = [state]
    
    for t in range(1, len(obs)):
        V.append({})
        new_path = {}
        for state in states:
            (prob, prev_state) = max(
                (V[t-1][prev] * trans_prob[prev].get(state, 0) * emit_prob[state].get(obs[t], 0), prev) 
                for prev in states)
            V[t][state] = prob
            new_path[state] = path[prev_state] + [state]
        path = new_path
    
    final_state = max((V[len(obs)-1][state], state) for state in states)
    return path[final_state[1]]

# Example
states = ['DT', 'NN', 'VB']
observations = ['The', 'dog', 'barks']
start_prob = {'DT': 0.6, 'NN': 0.3, 'VB': 0.1}
trans_prob = {'DT': {'NN': 0.5}, 'NN': {'VB': 0.4}, 'VB': {'DT': 0.1}}
emit_prob = {'DT': {'The': 0.9}, 'NN': {'dog': 0.8}, 'VB': {'barks': 0.7}}

tags = viterbi_decode(observations, states, start_prob, trans_prob, emit_prob)

# Output the sentence with tags
tagged_sentence = list(zip(observations, tags))
print("Tagged Sentence:", tagged_sentence)
