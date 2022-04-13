# AlphaRTC Gym example

This repository is an example for [AlphaRTC Gym](https://github.com/OpenNetLab/gym). By converting the raw stats of packet traces to some features and leveraging the PPO algorithms, this example trained a simple bandwidth estimator.

## Try this example

1. Fetch all submodule

```bash
git submodule init
git submodule update
```

2. Please visit Gym link for the instructions of [AlphaRTC Gym](https://github.com/OpenNetLab/gym) to install AlphaRTC Gym in `alphartc_gym`

3. Install example dependencies OpenAI GYM

```bash
python3 -m pip install -r requirements.txt
```

4. Run this example

```bash
python3 main.py
```

If you see something like
```
Episode 0        Average policy loss, value loss, reward -0.001012914622997811, 1749.7713505035483, -0.5917726188465685
Episode 1        Average policy loss, value loss, reward -0.003164666119424294, 1643.378693441069, -0.5696511401323291
Episode 2        Average policy loss, value loss, reward -0.0006242794975365944, 1503.4368403712722, -0.5418709073877055
Episode 3        Average policy loss, value loss, reward -0.0013577909024748813, 1396.8935836247986, -0.5149282318393551
Episode 4        Average policy loss, value loss, reward -0.0002334891452391077, 1349.4928827780047, -0.5091586053527624

```
which means this example has readied in your environment. And you can find your model under the folder `data`.
