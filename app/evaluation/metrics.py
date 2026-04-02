from jiwer import wer

reference = "where is my order"
prediction = "where is my order"

error = wer(reference, prediction)

print("Word Error Rate:", error)