
import numpy as np
import matplotlib.pyplot as plt

Final_PnL = []
Final_inventory = []
Final_mid = []
bid_avg = []
ask_avg = []
each_hits = []
each_lifts = []
Utility = []
for j in range(1000):
    #parameters
    S = 100
    T = 1
    delta_t = 0.005
    gamma = 0.01
    sigma = 2
    sigmasq = sigma ** 2
    I = 0
    k = 1.5
    c = 200
    W = 0
    bid_price = 99.26
    ask_price = 100.75
    spread = ask_price - bid_price
    #simulation,  track trading PnL, the Inventory, the OB Mid Price,
    # the Bid Price, the Ask Price, the number of hits and lifts
    W_trace = []
    I_trace = []
    price_ask_trace = []
    price_bid_trace = []
    OBmid_trace = []

    hits = 0
    lifts = 0
    for i in np.linspace(0, T, 200):
        #calculate the optimal action (bid-ask spread, and midpoint)
        delta_bid = spread/2
        delta_ask = spread/2
        price_ask = S + spread/2
        price_ask_trace.append(price_ask)
        price_bid = S - spread/2
        price_bid_trace.append(price_bid)
        #randomly increment the inventory by 1 or -1, and PnL accordingly ( can use uniform, or catergorical with correct weights)
        p_lift = c * np.exp(-k * delta_ask) * delta_t
        p_hit = c * np.exp(-k * delta_bid) * delta_t
        p_no_change = 1 - p_lift - p_hit
        print("lift p", p_lift)
        print("hit p", p_hit)
        print("p stay", p_no_change)
        outcome = np.random.choice([-1, 1, 0], p = [p_lift, p_hit, p_no_change])
        I = I + outcome
        I_trace.append(I)
        if outcome == 1:
            W = W - price_bid
            hits += 1
        elif outcome == -1:
            W = W + price_ask
            lifts += 1
        else:
            W = W
        W_trace.append(W)
        #increment midprice (independent of inventory)
        print("prev S", S)
        S = S + np.random.choice([-1, 1], p = [0.5, 0.5]) * sigma * np.sqrt(delta_t)
        print("new S", S)
        OBmid_trace.append(S)

    Utility.append(W + I * S)
    Final_PnL.append(W_trace[199])
    Final_inventory.append(I_trace[199])
    Final_mid.append(OBmid_trace[199])
    bid_avg.append(np.mean(price_bid_trace))
    ask_avg.append(np.mean(price_ask_trace))
    each_hits.append(hits)
    each_lifts.append(lifts)

final_avg_bid = np.mean(bid_avg)
final_avg_ask = np.mean(ask_avg)
print("final avg bid", final_avg_bid)
print("final avg ask", final_avg_ask)

#graphs
plt.hist(Utility, bins = 25)
plt.savefig("Utility.png")
hits_graph = plt.figure()
plt.hist(each_hits, bins = 25)
plt.savefig("hits.png")
lifts_graph = plt.figure()
plt.hist(each_lifts, bins = 25)
plt.savefig("lifts.png")
inventory_graph = plt.figure()
plt.hist(Final_inventory, bins = 25)
plt.savefig("inventory.png")
mid_graph = plt.figure()
plt.hist(Final_mid, bins = 25)
plt.savefig("mid.png")

#trace example
trace_pnl = plt.figure()
plt.plot(W_trace)
plt.savefig("W_trace.png")
trace_inventory = plt.figure()
plt.plot(I_trace)
plt.savefig("I_trace.png")
trace_mid = plt.figure()
plt.plot(OBmid_trace)
plt.savefig("mid_trace.png")

#stats
print("mean Utility", np.mean(Utility))
print("std Utility", np.std(Utility))
