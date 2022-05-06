import pandas as pd
import os
import numpy as np


def viterbi_algorithm(Y, S, pi, A, B):
    """Implement a Viterbi algorithm to find most likely sequence of states (S)
    for the given obsesrvation sequence (Y) and parameters of the HMM (pi, A, B)
    Your code should return both the viterbi matrix 
    with the maximum probability value of the paths ending in each state across time 
    and the path or sequence of S having maximum probability. Inputs and outputs are explained below:

    Args:
        Y: observation sequence, shape (T,).
        S: states space, shape (k,)
        pi：initial probabilities of different states, shape (k,)
        A: transition matrix, shape (k,k)
        B: emission matrix, shape (k,N) (N: length of the observation space)  


    Returns:
        1) V(t,s) a pandas data-frame, with max probability over all length t sequences ending in states s, shape (T,k).
            E.g.,         Healthy    Fever
                      0  0.30000  0.04000
                      1  0.08400  0.02700
                      2  0.00588  0.01512
        2) most likely sequence of states, shape(T,). E.g., ['Healthy', 'Healthy', 'Fever'] if T=3
    """
    # set previous state and running probability varaibles, set ViterbiArray and running probability
    viterbiArray = []
    most_likely_states_sequence = []
    previousState = ""
    runningProbability = 0

    #get the start state
    healthyState = B["Healthy"][Y[0]] * pi["Healthy"]
    feverState = B["Fever"][Y[0]] * pi["Fever"]

    #set the start state
    if (healthyState > feverState):
        previousState = "Healthy"
        runningProbability = healthyState
    else:
        previousState = "Fever"
        runningProbability = feverState

    #update most likely state sequence
    most_likely_states_sequence.append(previousState)
    viterbiArray.append([healthyState, feverState])

    for i in range(len(Y) - 1):
        #find the lowest probability of the next state
        #find the emission probability per health state
        probHealthy = B['Healthy'][Y[i + 1]]
        probFever = B['Fever'][Y[i + 1]]

        #find the probability per previous state
        healthyState = A[previousState]["Healthy"] * probHealthy * runningProbability
        feverState = A[previousState]["Fever"] * probFever * runningProbability

        #form the V matrix
        viterbiArray.append([healthyState,feverState])

        #update the runningProbability
        if (feverState > healthyState):
            runningProbability = feverState
            previousState = "Fever"
        else:
            runningProbability = healthyState
            previousState = "Healthy"

        #update most likely state sequence
        most_likely_states_sequence.append(previousState)

    ## Please rewrite following codes to get correct returned values.
    V = pd.DataFrame(np.array(viterbiArray), columns=["Healthy", "Fever"])
    return V, most_likely_states_sequence


if __name__ == '__main__':
    
    # Observation space O = {"normal", "cold", "dizzy"}, N=3
    O = ["normal", "cold", "dizzy"]

    # Sequences of Y Y = (y1, y2, …., yT). Observation space O = {"normal", "cold", "dizzy"}, N=3
    # Be mindful of the difference between observation space and observation sequence. 
    Y = [O[0], O[1], O[2],O[2]]
    
    # State space S = {s1,s2,….,sk}, k=2
    S = ["Healthy", "Fever"]
    
    # An array consisting of initial probabilities of different states 
    # pi = (pi1,pi2….pik) where pij stores the probability p(x1=sj), k=2
    pi = dict(zip(S,[0.6,0.4]))
   
    # Transition matrix A of size K x K where A[i,j] stores the transition probability of transiting from state Si to Sj
    # E.g., A['Healthy']['Fever'] - represents the probability of transitioning from 'Healthy' state to 'Fever' state
    A = dict(zip(S,list(
        map(lambda values:dict(zip(S,values)),[[0.7,0.3],[0.4,0.6]])
        )))

    
    # Emission matrix B of size K x N where B[i,j] stores the probability of observing Oj from state Si
    # E.g., B['Healthy']['normal'] - represents the probability of emitting 'normal' observation in 'Healthy' state
    B = dict(zip(S,list(
        map(lambda values:dict(zip(O,values)),[[0.5,0.4,0.1],[0.1,0.3,0.6]])
        )))

    
    V, states = viterbi_algorithm(Y, S, pi, A, B)

    print(V)
    # Name your script with your matric number, e.g., A0222374L.py.
    # Your codes will be run automatically, 
    # those scripts fail to generate outputs or are named in wrong matric number will get zero mark.
    cols=['Matric_Number']\
        +['V_%s_%d'%(s,t) for t in range(len(Y)) for s in S]\
        +["Most_likely_states_sequence"]
    result_row=[os.path.basename(__file__).replace('.py','')]+V.to_numpy().flatten().tolist()+[("-").join(states)]
    result_df = pd.DataFrame([result_row],columns=cols)
    print(result_df)
    result_df.to_csv('result.csv', mode='a', index=False, header=False)
