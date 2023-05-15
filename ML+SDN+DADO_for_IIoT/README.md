# Short reference for DADO CSV dataset

## Short explanation of the CSV files

The DADO framework has inputs that include:

- The network topology as a graph.
  - It contains the nodes of the network, whether the node is a host or a computing resource and its characteristics such as available RAM or CPU clock speed.
  - It also contains the links that join the nodes, as well as their characteristics (latency, capacity, etc.).
- A specification of microservices. Shortly, this is simply a declaration of the IDs of the different microservices that can be used and their characteristics, such as cycles, required RAM, input and output. It is done separately so it is reusable.
- A specification of the workflows. This includes the microservices required by the workflow and which host requests it.

Each of those is a single "scenario" file (that you can trace back to a set of latency, host load, link load, response time, TCAM usage and optimization time reports in the results ZIP). In this dataset, each scenario file is divided into five CSV files:

- `ComputingNodes.csv`, including the specifications of all the computing nodes.
- `Switches.csv`, including the specifications of all the switches.
- `Links.csv`, including the specifications of all the links.
- `Microservices.csv`, including the specifications of all the microservices.
- `Workflows.csv`, including the specifications of all the workflows.

## Name format

DADO's files, both results and the CSV files from the scenarios, have the following name convention:

```
MEC<Number of IIoT nodes>iiot<Number of fog nodes>fog<Number of SDN controllers>controllers<Number of workflows per device>wfpd<Length of the workflows>len<Microservices cycles (HI/MED/LO)>pw<Hardware ID used>hw<Additional info on the file>.csv
```

The hardware IDs are as follows:
| Hardware ID | Hardware kind |
| --- | --- |
| 0 | Non-computing |
| 1 | Arduino |
| 2 | Raspberry Pi |

Moreover, the sizes of the topologies are:
| IIoT nodes | Fog nodes | Topology size |
| --- | --- | --- |
| 10 | 10 | Small |
| 25 | 15 | Medium |
| 50 | 25 | Large |

And the microservice cycles are as follows:
| Microservice power ID | MCycles |
| --- | --- |
| LO | 100 |
| MED | 500 |
| HI | 1000 |

Finally, the additional info suffixes are the following:
| Suffix | File kind |
| --- | --- |
| `_lat` | Latency report |
| `_host_ld` | Host load report |
| `_lnk_ld` | Link load report |
| `_resp_time` | Response time report |
| `_tcam` | TCAM usage report |
| `_time` | Optimization time report |
| `ComputingNodes` | Computing nodes specification |
| `Switches` | Switches specification |
| `Links` | Links specification |
| `Microservices` | Microservices specification |
| `Workflows` | Workflows specification |

For instance, the computing nodes of the scenario with:

- Small topology (10 IIoT nodes, 10 fog nodes).
- 1 SDN controller.
- 1 workflow per device.
- 1 microservice per workflow
- 1000 MCycles.
- Non-computing hardware.

Are defined in `MEC10iiot10fog1controllers1wfpd1lenHIpw0hwComputingNodes.csv`.

## CSV files' columns

### Computing nodes

| Column | Description |
| --- | --- |
| id | ID of the computing node |
| power | Clock speed of the computing node (MHz) |
| memory | RAM of the computing node (MB) |

### Switches

| Column | Description |
| --- | --- |
| id | ID of the computing node |

### Links

| Column | Description |
| --- | --- |
| source | ID of the node that is the source of the link |
| destination | ID of the node that is the destination of the link |
| latency | Latency of the link (s) |
| capacity | Capacity of the link (MB) |

### Microservices

| Column | Description |
| --- | --- |
| id | ID of the microservice |
| cycles | Execution cycles of the microservice (MCycles) |
| input | Input size of the microservice (MB) |
| output | Output size of the microservice (MB) |
| memory | RAM required by the microservice (MB) |

### Workflows

| Column | Description |
| --- | --- |
| id | ID of the workflow |
| chain | List of the IDs of the microservices requested in the workflow. They are requested in the same order as shown in this column |
| starter | ID of the computing node requesting the workflow |
| response | Whether the workflow should have a response (always true) |
