import numpy as np

# helper function to parse the number and letter into the one-hot encoded format
def parseForSubmission(number, letter):
  result = ""
  for i in range(10):
    if (i == 9 - number):
      result += "1"
    else:
      result += "0"

  letter_position = 0
  # if input letter was lowercase
  if (ord(letter) > ord("Z")):
    letter_position = ord(letter) - ord("a")
  else:
    letter_position = ord(letter) - ord("A")

  for i in range(26):
    if (i == 25 - letter_position):
      result += "1"
    else:
      result += "0"
  return result

# function to turn an np.array of the predictions into the submission file
def parseFromPredictions(predictions):
  f = open("submission.csv", "w")
  for i in range(predictions.shape[0]):
    f.write(f"{i}, {parseForSubmission(int(predictions[i,0]), str(predictions[i,1]))}\n")
  f.close()

# example of getting the submission file
predictions = np.array([[1, 'a'], [1, 'b'], [2, 'a']])
parseFromPredictions(predictions)