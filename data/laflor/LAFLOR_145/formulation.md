### Sets:

- $ N_c = \{1, 2, 3\} $: Customer nodes.
- $ N_o = \{0\} \cup N_c $: Departure nodes.
- $ N_e = N_c \cup \{4\} $: Arrival nodes.
- $ N_d = \{1, 2, 3\} $: Drone-eligible customers.
- $ A = \{(i,j) \mid i \in N_o, j \in N_e, i \neq j\} $: Arcs.

### Parameters:

- $ K = 1 $, $ Q = 10 $ kg, $ W = 3 $ kg, $ E = 504 $ Wh, $ w_0 = 6 $ kg
- $ \alpha = 3.5 $ Wh/(kg·km), $ c_0 = \$22 $, $ P = 1008 $ W
- $ c_T = \$0.78 $/km, $ c_F = \$0.00248 $/Wh
- $ d_i, p_i $: delivery/pickup demands; $ r_{ij} $: full-precision Euclidean distances. Customer 3 is at $(4,5)$; all other coordinates and parameters are given in the task description.

### Decision Variables:

- $ x^{0}_{ij}, x^{1}_{ij}, x^{2}_{ij} \in \{0,1\} $: routing (truck+drone, truck alone, drone alone)
- $ z^{1}_i, z^{2}_i \in \{0,1\} $: service assignment
- $ s^{1}_i, s'^{1}_i, s^{2}_i, s'^{2}_i \geq 0 $: timing
- $ e_{ij}, e_i \in [0, E] $: drone energy
- $ w_i, w^d_i \in [0, W] $: total and remaining delivery payload after service, or before departure at a launch node
- $ v_i \in [0, Q] $: truck load

### Objective Function:

Minimize:

$$
c_T \sum_{(i,j) \in A} (x^{0}_{ij} + x^{1}_{ij}) r_{ij} + c_F \sum_{(i,j) \in A} e_{ij} + c_0 (1 - x^{0}_{0,4})
$$

### Key Constraints:

1. Route start/end at depot
2. Arc usage ≤ 1
3. Flow conservation for truck and drone
4. Each customer served exactly once: $ z^1_j + z^2_j = 1 $
5. Drone energy and payload limits
6. Truck load capacity
7. Time synchronization

### Added Constraint (Drone Service Mandate):

Customer 3 must be served by drone:

$$
z^{2}_3 = 1
$$

### Selected-Arc Payload and Flight Energy:

For a selected drone arc entering a customer served by the drone,

$$
x^2_{ij}=z^2_j=1\quad\Longrightarrow\quad
w_j=w_i-d_j+p_j,\qquad w^d_j=w^d_i-d_j.
$$

A sortie starts with its delivery parcels and no pickup parcels. No delivery parcels remain when it is recovered. Payload limits apply before and after each customer service. Selected-arc energy is tied to the actual outgoing payload:

$$
x^2_{ij}=1\quad\Longrightarrow\quad e_{ij}=\alpha(w_0+w_i)r_{ij},
\qquad x^2_{ij}=0\quad\Longrightarrow\quad e_{ij}=0.
$$

Energy accumulates along each selected sortie and resets upon battery replacement. For every sortie $S$, $\sum_{(i,j)\in S} e_{ij}\leq E$. Bounds based only on empty and fully loaded aircraft weights are insufficient to impose the energy equality.

### Timing Convention:

Flight time on a selected arc is $e_{ij}/P$ hours. Customer service and waiting contribute to timing and synchronization but not to battery consumption or energy cost. The truck can wait at the depot until final drone recovery. There is no route-duration limit or delivery deadline.
