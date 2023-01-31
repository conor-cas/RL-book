import math

#2 jobs plus wage_0,
wages = [4, 11, 7] # list, can adjust to see how policy changes, first is unemployment wage
gamma =.9 #float, future discount
pi = [0, .5, .5] #list of floats,  probability of job offer for each wage, 0 is unemployment wage
alpha = .2
states = [(0,1), (0,2),(1,1),(2,2)] # list of tuples
accept_rewards = {(0,1): wages[1], (0,2): wages[2], (1,1): wages[1], (2,2): wages[2]} # dict
# "rejections rewards set to zero for employed states so that policy will always "accept"
# they will lose job with the correct prob regardless of accept or reject
reject_rewards = {(0,1): wages[0], (0,2): wages[0], (1,1): 0, (2,2): 0} # dict,
#dict of Dicts
acc_probs = {(0,1): {(1,1): 1},
             (0,2): {(2,2): 1},
             (1,1): {(1,1): 1-alpha, (0,1): alpha * pi[1], (0,2): alpha * pi[2]},
             (2,2): {(2,2): 1-alpha, (0,1): alpha * pi[1], (0,2): alpha * pi[2]}
             }
#dict of dicts,
rej_probs = {(0,1): {(0,1): pi[1], (0,2): pi[2]},
             (0,2): {(0,1): pi[1], (0,2): pi[2]},
             (1,1): {(1,1): 1-alpha, (0,1): alpha * pi[1], (0,2): alpha * pi[2]},
             (2,2): {(2,2): 1-alpha, (0,1): alpha * pi[1], (0,2): alpha * pi[2]}
             }
actions = [0,1] #accept or reject
#all dicts
V = {(0,1): 10, (0,2): 20, (1,1): 10, (2,2): 20}
V_temp = {(0,1): 0, (0,2): 0, (1,1): 0, (2,2): 0}
pi_k = {(0,1): 0, (0,2): 0, (1,1): 0, (2,2): 0}


while(True): #run until small enough updates in V
    for state in states:
        q_accept = accept_rewards[state] #reward from first day wage
        for key in acc_probs[state].keys():
            q_accept += gamma * acc_probs[state][key] * V[key] # value times prob of moving there discounted by gamma
        q_reject = reject_rewards[state]
        for key in rej_probs[state].keys():
            q_reject += gamma * rej_probs[state][key] * V[key] # value times prob of moving there discounted by gamme
        q_accept = math.log(q_accept)
        q_reject = math.log(q_reject)
        if q_accept >= q_reject: # choose larger one
            V_temp[state] = q_accept
            pi_k[state] = "accept"
        else:
            V_temp[state] = q_reject
            pi_k[state] = "reject"

        V = V_temp #update value
    #proximity check
    if((V_temp[(0,1)] - V[(0,1)]) ** 2 < .01
    and (V_temp[(0,2)] - V[(0,2)]) ** 2 < .01
    and (V_temp[(1,1)] - V[(1,1)]) ** 2 < .01
    and (V_temp[(2,2)] - V[(2,2)]) ** 2 < .01):
        break

print(V)
print(pi_k)