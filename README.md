# GTA VI Twitch CCU Forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/vhigor/gta6-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/vhigor/gta6-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

A kernel-weighted log-OLS forecaster for GTA VI launch-day Twitch viewers, hour by hour. It anchors on the hour-1 print from comparable big-opening streams and projects the day's curve from it.

Practice loss (mean pinball): **0.177**, vs the house baseline's 0.387.

## Run

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

`main.ts` (Node 22, standard library only) reads the sealed inputs and writes the hourly forecast to `out/`.

---

Built for the [make.fun](https://make.fun/bounty/gta6-twitch-ccu-forecast-01) gta6-twitch-ccu-forecast-01 bounty. Referral code: `p26b6feq`.
