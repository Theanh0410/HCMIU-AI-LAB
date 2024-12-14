import pandas as pd
import numpy as np

class NaiveBayesFilter:
    def __init__(self):
        self.data = pd.DataFrame()
        self.vocabulary = set()
        self.p_spam = 0  # P(C=spam)
        self.p_ham = 0  # P(C=ham)
        self.parameters_spam = {}
        self.parameters_ham = {}

    def fit(self, X, y):
        # Create a DataFrame with word counts
        self.data = pd.DataFrame({'SMS': X, 'Label': y})
        self.data['SMS'] = self.data['SMS'].apply(lambda x: ' '.join(x))
        self.data['SMS'] = self.data['SMS'].str.split()

        # Calculate prior probabilities
        spam_messages = self.data[self.data['Label'] == 'spam']
        ham_messages = self.data[self.data['Label'] == 'ham']

        self.p_spam = len(spam_messages) / len(self.data)
        self.p_ham = len(ham_messages) / len(self.data)

        # Build the vocabulary and calculate word counts
        all_words = [word for message in self.data['SMS'] for word in message]
        self.vocabulary = set(all_words)
        
        all_words = []
        for message in self.data['SMS']:  # Iterate through each message in the 'SMS' column
            for word in message:          # For each word in the current message
                all_words.append(word)    # Add the word to the all_words list


        spam_word_counts = {word: 0 for word in self.vocabulary}
        ham_word_counts = {word: 0 for word in self.vocabulary}

        for message in spam_messages['SMS']:
            for word in message:
                spam_word_counts[word] += 1

        for message in ham_messages['SMS']:
            for word in message:
                ham_word_counts[word] += 1

        # Calculate conditional probabilities
        spam_total_words = sum(spam_word_counts.values())
        ham_total_words = sum(ham_word_counts.values())

        self.parameters_spam = {
            word: (spam_word_counts[word] + 1) / (spam_total_words + len(self.vocabulary))
            for word in self.vocabulary
        }

        self.parameters_ham = {
            word: (ham_word_counts[word] + 1) / (ham_total_words + len(self.vocabulary))
            for word in self.vocabulary
        }

    def predict_proba(self, X):
        proba = []
        for message in X:
            p_spam_given_message = self.p_spam
            p_ham_given_message = self.p_ham

            for word in message:
                if word in self.parameters_spam:
                    p_spam_given_message *= self.parameters_spam[word]
                if word in self.parameters_ham:
                    p_ham_given_message *= self.parameters_ham[word]

            proba.append([p_ham_given_message, p_spam_given_message])
        return proba
      
    def predict_proba(self, X):
    # Initialize a list to store the probabilities for each message
      proba = []

      # Iterate through each message in the input data
      for message in X:
          # Start with the prior probabilities of spam and ham
          p_spam_given_message = self.p_spam
          p_ham_given_message = self.p_ham

          # Multiply by the likelihood of each word in the message
          for word in message:
              if word in self.parameters_spam:  # Check if the word exists in the spam dictionary
                  p_spam_given_message *= self.parameters_spam[word]
              if word in self.parameters_ham:  # Check if the word exists in the ham dictionary
                  p_ham_given_message *= self.parameters_ham[word]

          # Append the calculated probabilities for this message to the list
          proba.append([p_ham_given_message, p_spam_given_message])

      # Return a list of [P(H|x), P(S|x)] for all messages
      return proba


    def predict(self, X):
        proba = self.predict_proba(X)
        return ['ham' if p[0] > p[1] else 'spam' for p in proba]

    def score(self, true_labels, predict_labels):
        correct = sum([1 for true, pred in zip(true_labels, predict_labels) if true == pred])
        return correct / len(true_labels)

    def score(self, true_labels, predict_labels):
        # Calculate True Positives (TP) and False Negatives (FN)
        tp = sum(1 for true, pred in zip(true_labels, predict_labels) if true == 'spam' and pred == 'spam')
        fn = sum(1 for true, pred in zip(true_labels, predict_labels) if true == 'spam' and pred == 'ham')
        
        # Calculate True Positives (TP): messages that are actually 'spam' and predicted as 'spam'
        true_positives = 0
        for actual, predicted in zip(true_labels, predict_labels):
            if actual == 'spam' and predicted == 'spam':
                true_positives += 1

        # Calculate False Negatives (FN): messages that are actually 'spam' but predicted as 'ham'
        false_negatives = 0
        for actual, predicted in zip(true_labels, predict_labels):
            if actual == 'spam' and predicted == 'ham':
                false_negatives += 1

        
        # Avoid division by zero
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        return recall
      
    def predict(self, X):
        # Initialize a list to store predictions
        predictions = []

        # Iterate through each message in the input data
        for message in X:
            # Get probabilities for spam and ham for the current message
            prob_ham, prob_spam = self.predict_proba([message])[0]  # Extract the first (and only) result for this message

            # Assign the label based on the higher probability
            if prob_spam > prob_ham:
                predictions.append('spam')
            else:
                predictions.append('ham')

        # Return the list of predictions
        return predictions


def fit(self, X, y):
    # Create DataFrame
    self.data = pd.DataFrame({'SMS': X, 'Label': y})
    self.data['SMS'] = self.data['SMS'].apply(lambda x: ' '.join(x))  # Join list of words into a string
    self.data['SMS'] = self.data['SMS'].str.split()  # Split back into a list of words

    # Calculate prior probabilities
    spam_mes = self.data[self.data['Label'] == 'spam']
    ham_mes = self.data[self.data['Label'] == 'ham']

    # Equations 1.3 and 1.4
    self.p_spam = len(spam_mes) / len(self.data)
    self.p_ham = len(ham_mes) / len(self.data)

    # Build vocabulary (use set to avoid duplicates)
    self.vocabulary = set()
    for message in self.data['SMS']:
        for word in message:
            self.vocabulary.add(word)  # Add words to vocabulary set

    # Initialize word counts for spam and ham messages
    spam_word_counts = {word: 0 for word in self.vocabulary}
    ham_word_counts = {word: 0 for word in self.vocabulary}

    # Count word occurrences for spam messages
    for mes in spam_mes['SMS']:
        for word in mes:
            spam_word_counts[word] += 1

    # Count word occurrences for ham messages
    for mes in ham_mes['SMS']:
        for word in mes:
            ham_word_counts[word] += 1

    # Calculate conditional probabilities (apply Laplace smoothing)
    spam_total_words = sum(spam_word_counts.values())
    ham_total_words = sum(ham_word_counts.values())

    self.parameters_spam = {
        word: (spam_word_counts[word] + 1) / (spam_total_words + len(self.vocabulary))  # Laplace smoothing
        for word in self.vocabulary
    }

    self.parameters_ham = {
        word: (ham_word_counts[word] + 1) / (ham_total_words + len(self.vocabulary))  # Laplace smoothing
        for word in self.vocabulary
    }

    # Optional: Print the parameters to check if they're populated
    print("parameters_spam: ", self.parameters_spam)
    print("parameters_ham: ", self.parameters_ham)

def score(self, true_labels, predict_labels):
    recall = 0
    true_pos = 0
    false_neg = 0

    for true, pred in zip(true_labels, predict_labels):
        if true == 'spam' and pred == 'spam':  # True positive: spam correctly predicted as spam
            true_pos += 1
        if true == 'spam' and pred == 'ham':  # False negative: spam incorrectly predicted as ham
            false_neg += 1

    if (true_pos + false_neg) > 0:
        recall = true_pos / (true_pos + false_neg)
    else:
        recall = 0  # If no spam messages exist, recall is 0

    return recall


import pandas as pd
import numpy as np

class NaiveBayesFilter:
    def __init__(self):
        self.data = pd.DataFrame()  # DataFrame to store word counts and labels
        self.vocabulary = []  # List to store unique words
        self.p_spam = 0  # Probability of Spam (P(S))
        self.p_ham = 0  # Probability of Ham (P(H))
        self.parameters_spam = {}  # Probability of each word given Spam (P(word|spam))
        self.parameters_ham = {}  # Probability of each word given Ham (P(word|ham))

    def fit(self, X, y):
        """
        This method trains the Naive Bayes classifier by calculating word frequencies
        and probabilities.
        """
        # Create DataFrame with messages and corresponding labels
        self.data = pd.DataFrame({'SMS': X, 'Label': y})
        
        # Preprocess the messages: split into words
        self.data['SMS'] = self.data['SMS'].apply(lambda x: x.split())
        
        # Calculate prior probabilities P(S) and P(H)
        spam_messages = self.data[self.data['Label'] == 'spam']
        ham_messages = self.data[self.data['Label'] == 'ham']
        
        self.p_spam = len(spam_messages) / len(self.data)
        self.p_ham = len(ham_messages) / len(self.data)

        # Build the vocabulary and calculate word counts
        all_words = [word for message in self.data['SMS'] for word in message]
        self.vocabulary = list(set(all_words))  # Create vocabulary with unique words
        
        # Initialize dictionaries for word counts for spam and ham
        spam_word_counts = {word: 0 for word in self.vocabulary}
        ham_word_counts = {word: 0 for word in self.vocabulary}
        
        # Count words for spam messages
        for message in spam_messages['SMS']:
            for word in message:
                spam_word_counts[word] += 1
        
        # Count words for ham messages
        for message in ham_messages['SMS']:
            for word in message:
                ham_word_counts[word] += 1
        
        # Calculate conditional probabilities with Laplace smoothing
        total_spam_words = sum(spam_word_counts.values())
        total_ham_words = sum(ham_word_counts.values())
        
        self.parameters_spam = {word: (spam_word_counts[word] + 1) / (total_spam_words + len(self.vocabulary))
                                for word in self.vocabulary}
        self.parameters_ham = {word: (ham_word_counts[word] + 1) / (total_ham_words + len(self.vocabulary))
                               for word in self.vocabulary}
        
        # Optionally, print out the learned parameters for debugging
        print('parameters_spam:', self.parameters_spam)
        print('parameters_ham:', self.parameters_ham)

    def predict_proba(self, X):
        """
        For each message, calculate P(H|x) and P(S|x), and return the probabilities.
        """
        proba = []
        
        for message in X:
            # Initialize probabilities with prior probabilities
            p_spam_given_message = self.p_spam
            p_ham_given_message = self.p_ham
            
            # Multiply the likelihood of each word in the message for spam and ham
            for word in message.split():  # Split the message into words
                if word in self.parameters_spam:
                    p_spam_given_message *= self.parameters_spam[word]
                if word in self.parameters_ham:
                    p_ham_given_message *= self.parameters_ham[word]
            
            # Append the probabilities for each message (P(H|x), P(S|x))
            proba.append([p_ham_given_message, p_spam_given_message])
        
        return proba

    def predict(self, X):
        """
        Predict the label for each message by comparing P(H|x) and P(S|x).
        """
        prob = []
        proba = self.predict_proba(X)  # Get probabilities for all messages
        
        for p_ham, p_spam in proba:
            if p_spam > p_ham:
                prob.append('spam')
            else:
                prob.append('ham')
        
        return prob

    def score(self, true_labels, predict_labels):
        """
        Calculate the recall score by comparing true labels and predicted labels.
        Recall = TP / (TP + FN), where TP = True Positives, FN = False Negatives.
        """
        true_pos = 0
        false_neg = 0
        
        # Count True Positives and False Negatives
        for true, pred in zip(true_labels, predict_labels):
            if true == 'spam' and pred == 'spam':
                true_pos += 1
            elif true == 'spam' and pred == 'ham':
                false_neg += 1
        
        # Calculate recall
        if (true_pos + false_neg) > 0:
            recall = true_pos / (true_pos + false_neg)
        else:
            recall = 0
        
        return recall
