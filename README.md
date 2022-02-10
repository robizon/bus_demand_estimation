# Simulation Dode Discription


For our simulations we simulate a short bus route with 6 bus stops.
For simplicity, and without any loss of generality, this is a pickup
only route, i.e., one that every passenger is getting off at the terminal
stop (Stop 6). The number of passengers boarding on each stop follows a
non-homogeneous Poisson process. In particular, Poisson arrival rate takes two
values, &lambda; and 2*&lambda; respectibely off-peak and peak hours.

We simulate 1,000,000 trips for this bus route. Each trip is randomly
assigned to a peak or off-peak hour based on the ratio of
the peak hours in the specific simulation. Once we simulate the
process, we explore three different ways of building a passenger
demand model.


For detailed description of simulation and estimation please see section 3.3 in the paper. 
