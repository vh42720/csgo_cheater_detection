# Detecting cheaters in Counter Strike: Global Offensive (CS:GO)

### Intro

What is CS:GO? Why is this a problem? Professional + Steam problems

### Goals

This project will apply machine learning to predict cheaters in CS:GO.

### Data

Registered Steam users are identified by a unique Steam ID, which can be
used to retrieve public player information, in-game metrics, and instances
of bans due to cheating detected by Valve Anti-Cheat (VAC).

Steps:
1. find steam user ids that are known for cheating: feed user with cheating instances
to vacbanned.com. However, this also label players who cheat for other game which is caught by VAC.
2. find steam user playing CSGO that is known for not cheating. This turns out
to be incredible hard since players range vary and not all VAC detected.
Choosing a list of professional players might be the best approach for this
as they have less incentives to cheat (or more??)
3. compile the list and query into API => get steam data for CSGO => write to CSV

Notes:
Exclude steam ID without CS:GO or VAC banned before CS:GO release (or not? since this
can become another feature => cheat_before). We also exclude new players.

### Steam APIs

There are four APIs method calls that will be used here:
1. GetPlayerSummaries
2. GetOwnedGames
3. GetPlayerBans
4. GetUserStatsForGame

### Features

The API can returns over 250 features for each user in CS:GO. This means that
some thoughtful cleaning must be done. But can deep learning do this by itself?

Ex: accuracy_pistol = total_hit_pistol/total_shot_pistol

### Models

Apply ML or deep learning

While this is a classification problem, outputs in the form of probability is more
useful here for 2 practical reasons:
1. threshold for cheater label can be manually adjusted. Banning a non cheater
is much much worse than not banning a cheater in this case. Thus, False Positive
rate must be very low for the model to be considered successful.
2. if the probability is high, manual review of the player is better than
blindly VAC banned.

## Initialize

To start collecting data, first run the init.py file to create the 
steamids file which contain the list of all steam ids. 

