# Apptainer Environment

The conda environment is often the easiest option. Use Apptainer when you want a
reproducible Linux container with the exact robotics stack.

## Install Apptainer

```bash
sudo apt update
sudo apt upgrade
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:apptainer/ppa
sudo apt update
sudo apt install -y apptainer
```

## Build The Container

Run this from the repository root so `environment.yml` is visible to the
definition file.

```bash
apptainer build duck-collector.sif containers/duck-collector.def
```

## Start A Shell

```bash
apptainer shell --cleanenv --containall --bind "$(pwd)":/workspace duck-collector.sif
cd /workspace
pip install -e .
```

For headless simulation or video rendering, wrap commands with `xvfb-run`:

```bash
xvfb-run -s "-screen 0 1280x720x24" python scripts/evaluate_policy.py --room 3
```

To launch an interactive Python shell inside the container:

```bash
python
```
