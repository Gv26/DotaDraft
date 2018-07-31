# DotaDraft
Machine learning application for drafting Captains Mode matches in Dota 2.

## Getting started

You can configure DotaDraft by editing the parameters in `config.py`.

### Fetching training data
In order to fetch new match data for training, `STEAM_API_KEY` must be set to your [Steam Web API key](https://steamcommunity.com/dev/apikey) in `config.py`.
Run `matches.py` and new matches will begin to be fetched (this is very slow).
