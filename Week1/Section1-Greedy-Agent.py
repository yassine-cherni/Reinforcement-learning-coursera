# [Graded_agent]
def argmax(q_values):
    """
    Takes in a list of q_values and returns the index
    of the item with the highest value. Breaks ties randomly.
    returns: int - the index of the highest value in q_values
    """
    top = float("-inf")
    ties = []
    
    for i in range(len(q_values)):
        # if a value in q_values is greater than the highest value, then update top and reset ties to zero
        # if a value is equal to top value, then add the index to ties (hint: do this no matter what)
        # return a random selection from ties. (hint: look at np.random.choice)
        ### START CODE HERE ###
        if q_values[i] > top:
            top = q_values[i]
            ties = [i]
        elif q_values[i] == top:
            ties.append(i)
        ### END CODE HERE ###
    return ties[np.random.choice(len(ties))]
  # Test argmax implentation
test_array = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
assert argmax(test_array) == 8, "Check your argmax implementation returns the index of the largest value"

test_array = [1, 0, 0, 1]
total = 0
for i in range(100):
    total += argmax(test_array)

assert total > 0, "Make sure your argmax implementation randomly choooses among the largest values."
assert total != 300, "Make sure your argmax implementation randomly choooses among the largest values."
