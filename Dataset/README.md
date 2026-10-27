# VeReMi Extension

VeReMi Extension is a dataset for evaluating Misbehavior Detection Systems (MDSs) in Vehicular Ad hoc Networks (VANETs). It provides **simulated** V2X message logs and a broad set of misbehavior and attack scenarios. Compared to the original VeReMi dataset, VeReMi Extension introduces realistic sensor error models, additional attacks, larger datasets, and benchmark detection results. VeReMi NextGen is part of a published paper, submitted to the [International Conference on Communications 2020](https://icc2020.ieee-icc.org/index.html).

This website provides a brief overview of the VeReMi Extension dataset. For more detailed information, please refer to the corresponding paper.


## Simulation Setup

VeReMi Extension was generated using the [Framework for Misbehavior Detection (F2MD)](https://github.com/josephkamel/f2md). F2MD is an extension of VEINS and enables the simulation and detection of different misbehavior detection use cases. The simulation setup is based on [VEINS](https://veins.car2x.org/), [OMNeT++ (Version 5.6.1)](https://omnetpp.org/), and [SUMO (Version1.5.0)](https://sumo.dlr.de/docs/index.html).

The dataset uses the [Luxembourg SUMO Traffic (LuST) scenario](https://github.com/lcodeca/lustscenario). LuST is a synthetic traffic scenario of Luxembourg City that is based on realistic traffic data. VeReMi Extension uses a subsection of the LuST network with an area of approximately 1.61 km².


## Dataset Structure

VeReMi Extension provides multiple dataset subsets for different time periods and traffic densities.
For each type of misbehavior, two subsets are provided:

| Time period | Description |
|---|---|
| 07:00–09:00 | Rush-hour scenario with higher vehicle density |
| 14:00–16:00 | Low-traffic scenario with lower vehicle density |

In addition, one mixed test-bench subset is provided:

| Dataset | Time period | Description |
|---|---|---|
| MixAll_0024 | 00:00–24:00 | Full-day scenario containing a mixture of all described misbehavior attacks |

The attacker penetration rate is set to 30% in all simulations.


## Attack Types

VeReMi Extension includes several attack types that affect different message attributes. The dataset distinguishes between non-malicious malfunctions and malicious attacks.

| Category | Attack types |
|---|---|
| Time-related | Delayed Messages |
| Position-related | Constant Position, Random Position, Constant Position Offset, Random Position Offset |
| Speed-related | Constant Speed, Random Speed, Constant Speed Offset, Random Speed Offset |
| Network-related | DoS Attack, DoS Random |
| Replay-based | Data Replay, Disruptive Attack |
| Identity-related | Traffic Congestion Sybil, DoS Random Sybil, Data Replay Sybil, DoS Disruptive Sybil |
| Multi-parameter | Eventual Stop |

VeReMi Extension extends the original VeReMi dataset by introducing a broader range of attack types, including speed-related attacks, delayed messages, replay-based attacks, and Sybil-based attack variants. The attacks cover multiple message attributes and enable the evaluation of Misbehavior Detection Systems under more diverse and realistic conditions.


## Acknowledgement

The dataset was primarily put together by Joseph Kamel at the Institute of Distributed Systems, part of Ulm University. Please contact Joseph if you have any questions, comments or criticism. 

The VeReMi Extension work was carried out within the French Program Investissements d'avenir at the Technological Research Institute SystemX and in collaboration with the SecForCARs project funded by the German Federal Ministry of Education and Research.

## License for Dataset
Creator: Institute of Distributed Systems, Ulm University

CC-BY 4.0 Creative Commons Attribution 4.0 International

This license requires that reusers give credit to the creator. It allows reusers to distribute, remix, adapt, and build upon the material in any medium or format, even for commercial purposes.