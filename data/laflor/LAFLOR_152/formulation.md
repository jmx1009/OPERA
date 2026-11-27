### Step 1: Decision Variables Definition

Let stations be V={A,H,B}, markets I={1,...,6}, and let P_i be the ordered list of movement-queue events on market i's physical UAM itinerary. Thus P_1=(A,H,H,B), P_2=(B,H,H,A), P_3=(A,H), P_4=(H,B), P_5=(B,H), and P_6=(H,A). Choose p in {55,70,85} and n_j in {1,2,3} for j in V. Let q_i be expected captured UAM trips, lambda_j the movement-request rate at station j, and W_j its expected queueing delay. Let D_i, g_i, t_i, and c_i be the supplied potential demand, ground time, pre-wait UAM time, and variable cost.

### Step 2: Objective Function Construction

Maximize

Z=sum_{i in I}(p-c_i)q_i-sum_{j in V}K_j(n_j).

### Step 3: Constraint Construction

Let m_ij be the number of occurrences of station j in P_i. Counting occurrences rather than distinct station labels gives

lambda_j=sum_{i in I}m_ij*q_i   for every j in V.

In particular,

lambda_A=q_1+q_2+q_3+q_6,

lambda_H=2q_1+2q_2+q_3+q_4+q_5+q_6,

lambda_B=q_1+q_2+q_4+q_5.

For every station j, require

0<=lambda_j<n_j*mu,

W_j=30*C(n_j,lambda_j/mu)/(n_j*mu-lambda_j),

where a_j=lambda_j/mu, rho_j=lambda_j/(n_j*mu), and

C(n_j,a_j)=[a_j^{n_j}/(n_j!*(1-rho_j))]/[sum_{k=0}^{n_j-1}a_j^k/k!+a_j^{n_j}/(n_j!*(1-rho_j))].

Market i accumulates one expected wait for each occurrence in P_i, so

T_i=t_i+sum_{j in V}m_ij*W_j.

The binary-Logit capture equations are

q_i=D_i*exp[-0.035p-0.18T_i]/[exp(-0.18g_i)+exp(-0.035p-0.18T_i)]   for every i in I.

For each of the 81 discrete (n_A,n_H,n_B,p) combinations, solve the capture, station-load, and wait equations simultaneously; verify stability at A, H, and B; evaluate Z; and select the largest value.
