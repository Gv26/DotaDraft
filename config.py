# Your Steam Web API key. These may be acquired from:
# https://steamcommunity.com/dev/apikey
STEAM_API_KEY = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'

# Filtering parameters for fetching match data. Values as stated at:
# https://wiki.teamfortress.com/wiki/WebAPI/GetMatchDetails#Result_data
game_mode = 2  # Captains mode
lobby_type = 7  # Ranked
human_players = 10

# Match IDs to search. Set to None for first match of patch and most recent match respectively.
start_match_id = 4032019767  # Patch 7.19 begins on match 4032019767.
end_match_id = 4036846609
