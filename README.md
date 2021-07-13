# R2D2

A thin wrapper that makes the the R2D2 algorithm from [DeepMind's Acme
framework](https://github.com/deepmind/acme/) compatible with
[AgentOS](https://agentos.org/latest/).

## Installation

Requires Python 3.6 to 3.8.

* Create a virtualenv, e.g. `virtualenv -p python3.8 env`
* Activate your virtualenv: `source env/bin/activate`
* Clone the latest [AgentOS](https://github.com/agentos-project/agentos) master
* `pip install -e [path/to/agentos/clone/]`
* `pip install -r requirements.txt`
* Run the demo script (ported from acme-test repo): `python demo.py`
* Run basic tests of the policy: `python test.py`

