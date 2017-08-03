# Poker hand classifier
Uses tensorflow to create and use a three hidden layer, convolutional neural network. Must train before use, data and formatter provided.

## Input format for use
five cards in one input string: (suit1,card1,suit2,card2,suit3,card3,suit4,card4,suit5,card5)

### Suits
1-4, not programmed to recognize specific suit value (ex. no suit heirarchy)

### Card (rank)
0-13 (A-K))

## Output format
      0: Nothing
      1: One pair
      2: Two pairs
      3: Three of a kind
      4: Straight
      5: Flush
      6: Full house
      7: Four of a kind
      8: Straight flush
      9: Royal flush

@author Jack Stettner
