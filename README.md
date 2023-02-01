# Brick Breaker (Game + AI)

![Brick Breaker demonstration](https://user-images.githubusercontent.com/48833786/216108673-f3823999-e3e9-4415-84eb-81ef17070209.png)

This project features a clone of a popular 'brick breaker' game variant, 
characterised by shooting a volley of 'balls' at an angle of choice.

Along with the game, I have attempted two approaches at an AI for the game, of which one turned out successful.

## Game Rules
- Rows of bricks are generated on every level.
- You may shoot your volley of balls at an angle of choice.
 
  To do so, you:

  1. move your mouse cursor to select angle,
  2. and click to fire your volley.
- Collecting green circles increases your ball count.

  Note that green circles that reach the bottom row are  automatically collected.
- When a brick reaches the bottom row, you lose!
 
  Try to last as long as possible.
 
  The final level you reach before you lose is your score.
- The position from which you shoot the balls in the next level is determined as the position where the first ball lands in the current level.
- Holding down [A] triggers the 'AI Assist' mode.

  The arrow will turn yellow to indicate the AI is calculating the optimal move.

  The arrow will turn red once the AI has a move ready for you. Click to fire!

## How to Run the Game
- Run `game.py` from the repository root using Python 3.
- Package requirements are specified in `requirements.txt`.
  - Pytorch and matplotlib related packages may not be necessary to run the game.
### Game Options

You may edit variables in `constants.py` to achieve various effects.

For example, the `EARLY_TERMINATION` variable can be set to `False` to not skip ball animation.

(The default behaviour when `True` skips animation when possible to reduce wait time.)

You may set `TRAIL` to `True` to add a visual trail to the balls.

WARNING: Some changes may break the game.

## What About the Game AI?
Two attempts at the AI have been made.

### Genetic Algorithm
- Initially, I tried to optain a network to output the optimal angle, given the game state as input.
- As there is no labeled data, I ran a Genetic Algorithm based approach.
- The results were unsatisfactory after 10 days of computation. Scores remained in the 65-80 range.
- As the search space was too large, this approach was abandoned.
 
### Simulation Based Explorer-Evaluator
- Given we have an 'evaluator' network to evaluate how bad(or good) the current game situation is,
we can run simulations for different shooting angles(explorer), check and evaluate the resulting game states,
and choose the best option.
- Given we have sufficient gameplay history, we can train an evaluator network from our history of games.

  The closer to game-over a state was, the worse.
- We have a loop! First, I initialized a 0th generation random evaluator network to play 1200 games.
- Then I trained a 1st gen evaluator from the 0th generation's game history.
- I planned to repeat this process for more generations of AI with increasingly better performance, but... the 1st gen AI was already too good at the game.

  It wasn't plausible to run the 1st gen AI multiple times, as games just went on and on.
- So I stopped at the 1st generation.
  #### Hard Coded Evaluator
  - Instead of an evaluator trained on previous game history, I also tried implementing a hard-coded weight matrix for the evaluator.
  - This turned out to have decent performance, and seemed more stable than our trained evaluator.
  
    It also tries to collect more green circles.
  - Ironically this hard coded AI performed the best. This is the AI currently configured for the 'AI Assist' mode.

## Blog Posts
- Several blog posts have been made regarding this project on my personal blog.
- If you can read Korean, you might find them informative:
    1. [Regarding game physics](https://syphon.tistory.com/123)
    2. [Regarding game AI](https://syphon.tistory.com/70)

