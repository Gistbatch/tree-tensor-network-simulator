MPS Circuit Simulation
======================

Quantum circuit simulation using a MPS representation of the quantum state. Run `python3 example.py` for a simple demonstration.

The code supports general "qudits".

Christian B. Mendl


TTN Circuit Simulation
======================

Quantum circuit simulation using a TTN representation of the quantum state. Run `python3 example.py` for a simple demonstration.
Includes finding of a fitting tree structure.

Install
======================

Install dependencies with `pip3 install - f requirements.txt`
To use gpu functionality, cuda has to be set up correctly.
Then you can use `pip3 install cupy`.
As an alternative you can build a docker container.
```
docker build --tag mps_circuit_sim .
docker run mps_circuit_sim
```
Running the data generation from the docker file is not recommended.

Data
======================

To recreate the data and plots run `python3 recreate_data.py`
