from rl.chapter9.order_book import *
import numpy as np
import matplotlib.pyplot as plt
#create a new order book
bids: PriceSizePairs = [DollarsAndShares(
dollars=x,
shares=poisson(100. - (100 - x) * 10)
) for x in range(100, 90, -1)]
asks: PriceSizePairs = [DollarsAndShares(
dollars=x,
shares=poisson(100. - (x - 105) * 10)
) for x in range(105, 115, 1)]
ob0: OrderBook = OrderBook(descending_bids=bids, ascending_asks=asks)

#limit sell order
d_s1, ob1 = ob0.sell_limit_order(107, 40)

#market sell order
d_s2, ob2 = ob1.sell_market_order(120)
#limit buy order
d_s3, ob3 = ob2.buy_limit_order(100, 80)
# market buy order
d_s4, ob4 = ob3.sell_limit_order(104, 60)
#parameters
S = 100
T = 1
delta_t = 0.005
gamma = 0.1
sigma = 2
sigmasq = sigma ** 2
I = 0
k = 1.5
c = 140
W = 0

#simulation,  track trading PnL, the Inventory, the OB Mid Price,
# the Bid Price, the Ask Price, the number of hits and lifts
W_trace = []
I_trace = []
price_ask_trace = []
price_bid_trace = []
mid_trace = []
hits = 0
lifts = 0
for i in range(T/delta_t):
    #calculate the optimal action (bid-ask spread, and midpoint)
    delta_bid = ((2 * I + 1) * gamma * sigmasq * (T - i)) + 1 / gamma * np.log((1 + gamma) / k)
    delta_ask = ((1 - 2 * I) * gamma * sigmasq * (T - i)) + 1 / gamma * np.log((1 + gamma) / k)
    mid = S - I * gamma * sigmasq * (T - i)
    price_ask = delta_ask + S
    price_ask_trace.append(price_ask)
    price_bid = S - delta_bid
    price_ask_trace.append(price_bid)
    #randomly increment the inventory by 1 or -1, and PnL accordingly ( can use uniform, or catergorical with correct weights)
    p_lift = c * np.exp(-k * delta_ask)
    p_hit = c * np.exp(-k * delta_bid)
    p_no_change = 1 - p_lift - p_hit
    outcome = np.random.choice([-1, 1, 0], p = [p_lift, p_hit, p_no_change])
    I = I + outcome
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
    S = S + np.random.choice([-1, 1], p = [0.5, 0.5]) * sigma * np.sqrt(delta_t)
    mid_trace.append(S)
print("hi")
plt.plot(W_trace)
plt.show()
plt.plot(I_trace)
plt.plot(mid_trace)
